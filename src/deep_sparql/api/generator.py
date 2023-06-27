from io import TextIOWrapper
import os
import time
import sys
from typing import Any, Dict, List, Tuple, Optional, Union, Iterator, Callable

import torch
from torch import nn

from deep_sparql.model import PretrainedEncoderDecoder, model_from_config

from text_correction_utils import data, tokenization, prefix
from text_correction_utils.api.corrector import ModelInfo
from text_correction_utils.api import corrector
from text_correction_utils.api.utils import device_info
from text_correction_utils.inference import (
    IdxSelectFn,
    beam_select_fn,
    eos_stop_fn,
    greedy_select_fn,
    sample_select_fn,
    search,
    beam_search
)

from deep_sparql.utils import postprocess_output, prepare_sparql_query

_BASE_URL = "https://ad-publications.informatik.uni-freiburg.de/" \
    "ACL_whitespace_correction_transformer_BHW_2023.materials"
_NAME_TO_ZIP = {
}


class DecodingState:
    def __init__(
        self,
        initial_token_ids: List[int],
        ent_start_id: int,
        ent_stop_id: int,
        prop_start_id: int,
        prop_stop_id: int,
    ):
        assert len(initial_token_ids) > 0
        self._token_ids = initial_token_ids
        self._state: Optional[str] = None
        self._start_idx = 0
        self._ent_start_id = ent_start_id
        self._ent_stop_id = ent_stop_id
        self._prop_start_id = prop_start_id
        self._prop_stop_id = prop_stop_id

    def is_ent_start(self) -> bool:
        return (
            self._token_ids[-1] == self._ent_start_id
            and self._state is None
        )

    def is_ent_stop(self) -> bool:
        return self._token_ids[-1] == self._ent_stop_id and self.is_ent()

    def is_prop_start(self) -> bool:
        return (
            self._token_ids[-1] == self._prop_start_id
            and self._state is None
        )

    def is_prop_stop(self) -> bool:
        return self._token_ids[-1] == self._prop_stop_id and self.is_prop()

    def is_ent(self) -> bool:
        return self._state == "ent"

    def is_prop(self) -> bool:
        return self._state == "prop"

    def is_obj(self) -> bool:
        return self._state is not None

    def get_obj_token_ids(self) -> List[int]:
        return self._token_ids[self._start_idx:]

    def add(self, token_id: int):
        self._token_ids.append(token_id)
        if (
            (self.is_ent() and token_id == self._ent_stop_id)
            or (self.is_prop() and token_id == self._prop_stop_id)
        ):
            self._state = None
        elif self._state is None and token_id == self._ent_start_id:
            self._state = "ent"
            self._start_idx = len(self._token_ids)
        elif self._state is None and token_id == self._prop_start_id:
            self._state = "prop"
            self._start_idx = len(self._token_ids)


class SPARQLGenerator(corrector.TextCorrector):
    task = "spelling correction"

    @classmethod
    def available_models(cls) -> List[ModelInfo]:
        return [
            ModelInfo(
                name="dummy",
                description="a dummy model",
                tags=["default", "dummy"]
            ),
        ]

    @classmethod
    def supported_input_formats(cls) -> List[str]:
        return ["text", "text_language", "text_detections_language"]

    @classmethod
    def supported_output_formats(cls) -> List[str]:
        return ["text", "text_language"]

    @classmethod
    def _model_url(cls, model: str) -> str:
        return f"{_BASE_URL}/{_NAME_TO_ZIP[model]}"

    @property
    def name(self) -> str:
        return self.cfg["experiment"]["name"]

    @classmethod
    def _model_from_config(cls, cfg: Dict[str, Any]) -> nn.Module:
        input_tokenizer = tokenization.Tokenizer.from_config(
            cfg["input_tokenizer"]
        )
        if "output_tokenizer" in cfg:
            output_tokenizer = tokenization.Tokenizer.from_config(
                cfg["output_tokenizer"]
            )
        else:
            output_tokenizer = None
        return model_from_config(
            cfg["model"],
            input_tokenizer,
            output_tokenizer
        )

    @property
    def max_length(self) -> int:
        return max(512, self.cfg["train"]["data"]["max_length"])

    @property
    def context_length(self) -> int:
        raise NotImplementedError

    def supported_languages(self) -> Optional[List[str]]:
        lang_cfg = self.cfg["input_tokenizer"].get("language")
        if lang_cfg is None:
            return None
        else:
            return lang_cfg["languages"]

    def __init__(
        self,
        model_dir: str,
        device: Union[str, int]
    ) -> None:
        super().__init__(model_dir, device)
        precision = self.cfg["train"].get("mixed_precision_dtype", "fp32")
        self.set_precision(precision)
        self.logger.debug(f"loaded model config:\n{self.cfg['model']}")
        self.logger.info(
            f"running {self.name} SPARQL generator "
            f"on device {device_info(self.device)}"
        )
        self.input_tokenizer = tokenization.Tokenizer.from_config(
            self.cfg["input_tokenizer"]
        )
        assert "output_tokenizer" in self.cfg
        self.output_tokenizer = tokenization.Tokenizer.from_config(
            self.cfg["output_tokenizer"]
        )
        self._initial_token_ids = self.output_tokenizer.tokenize("")
        out_pfx = self.output_tokenizer.num_prefix_tokens()

        # some options for inference
        self._initial_token_ids = self._initial_token_ids.token_ids[:out_pfx]
        self._eos_token = "</s>"
        self._eos_token_id = self.output_tokenizer.special_token_to_id(
            self._eos_token
        )
        self._boe_token_id = self.output_tokenizer.special_token_to_id(
            "<boe>"
        )
        self._eoe_token_id = self.output_tokenizer.special_token_to_id(
            "<eoe>"
        )
        self._bop_token_id = self.output_tokenizer.special_token_to_id(
            "<bop>"
        )
        self._eop_token_id = self.output_tokenizer.special_token_to_id(
            "<eop>"
        )
        self._strategy = "greedy"
        self._beam_width = 5
        self._sample_top_k = 5
        assert self._eos_token_id is not None

        self._entity_index = None
        self._property_index = None

        self._output_conts = [
            self.output_tokenizer.de_tokenize(
                [self._eos_token_id, i, self._eos_token_id],
                False
            )[len(self._eos_token):-len(self._eos_token)].encode("utf8")
            for i in range(self.output_tokenizer.vocab_size())
        ]
        self._initial_ent_conts = [True] * len(self._output_conts)
        self._initial_prop_conts = [True] * len(self._output_conts)

    def _build_inference_loader_config(self) -> Dict[str, Any]:
        return {
            "tokenizer_config": self.cfg["input_tokenizer"],
            "window_config": {"type": "full"}
        }

    def _prepare_batch(self, batch: data.InferenceBatch) -> Dict[str, Any]:
        token_ids_np, pad_mask_np, *_ = batch.tensors()
        inputs = {
            "token_ids": torch.from_numpy(token_ids_np).to(
                non_blocking=True,
                device=self.device
            ),
            "padding_mask": torch.from_numpy(pad_mask_np).to(
                non_blocking=True,
                device=self.device
            ),
        }
        return inputs

    def _initial_decoding_state(
        self,
        batch_size
    ) -> List[DecodingState]:
        # keep track of decoding state
        # None --> nothing
        # (ent, idx) --> entity starting at idx
        # (prop, idx) --> property starting at idx
        return [
            DecodingState(
                list(self._initial_token_ids),
                self._boe_token_id,
                self._eoe_token_id,
                self._bop_token_id,
                self._eop_token_id
            )
            for _ in range(batch_size)
        ]

    def _index_select_fn(
        self,
        decoding_states: List[DecodingState]
    ) -> IdxSelectFn:
        # helper fn to get valid continuations
        # from a prefix index
        def _get_indices_and_conts(
            index: prefix.Vec,
            decoding_states: List[DecodingState],
            state_indices: List[int],
            filter_fn: Callable[[DecodingState], bool],
            initial_conts: List[bool],
            end_token_id: int
        ) -> Tuple[List[int], List[List[bool]]]:
            indices = []
            conts = []
            values = []
            pfx_indices = []
            prefixes = []
            for i, idx in enumerate(state_indices):
                if not filter_fn(decoding_states[idx]):
                    continue
                token_ids = decoding_states[idx].get_obj_token_ids()
                if len(token_ids) == 0:
                    conts.append(initial_conts)
                    values.append(None)
                    indices.append(i)
                    continue
                decoded = self.output_tokenizer.de_tokenize(
                    decoding_states[idx].get_obj_token_ids(),
                    False
                ).encode("utf8")
                prefixes.append(decoded)
                pfx_indices.append(i)

            pfx_conts = index.batch_contains_continuations(
                prefixes
            )
            pfx_values = index.batch_get(prefixes)
            for cont, value in zip(pfx_conts, pfx_values):
                if value is None:
                    continue
                cont[end_token_id] = True

            indices.extend(pfx_indices)
            conts.extend(pfx_conts)
            return indices, conts

        def _fn(
            scores: torch.Tensor,
            indices: List[int]
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            start = time.perf_counter()
            conts = torch.ones(
                *scores.shape,
                dtype=torch.bool
            )
            # print(conts.shape, scores.device, conts.device, indices)

            # first entities
            ent_indices, ent_conts = _get_indices_and_conts(
                self._entity_index,
                decoding_states,
                indices,
                lambda state: state.is_ent(),
                self._initial_ent_conts,
                self._eoe_token_id
            )
            if len(ent_indices) > 0:
                ent_indices = torch.tensor(ent_indices, dtype=torch.long)
                ent_conts = torch.tensor(ent_conts, dtype=torch.bool)
                # print(
                #     f"ent_conts: {ent_conts.shape}, "
                #     f"ent_indices: {ent_indices}"
                # )
                conts[ent_indices, :ent_conts.shape[-1]] = ent_conts

            # then properties
            prop_indices, prop_conts = _get_indices_and_conts(
                self._property_index,
                decoding_states,
                indices,
                lambda state: state.is_prop(),
                self._initial_prop_conts,
                self._eop_token_id
            )
            if len(prop_indices) > 0:
                prop_indices = torch.tensor(prop_indices, dtype=torch.long)
                prop_conts = torch.tensor(prop_conts, dtype=torch.bool)
                # print(
                #     f"prop_conts: {prop_conts.shape}, "
                #     f"prop_indices: {prop_indices}"
                # )
                conts[prop_indices, :prop_conts.shape[-1]] = prop_conts

            scores[torch.logical_not(conts)] = float("-inf")
            token_ids = torch.argmax(scores, -1)
            scores = torch.gather(scores, -1, token_ids[:, None]).squeeze(-1)

            # update decoding states
            for idx, token_id in zip(indices, token_ids.tolist()):
                decoding_states[idx].add(token_id)

            end = time.perf_counter()
            # print(f"select took {1000 * (end - start):.2f}ms")

            return token_ids, scores

        return _fn

    def _inference(self, inputs: Dict[str, Any]) -> Any:
        assert isinstance(self.model, PretrainedEncoderDecoder)
        enc = self.model.encode(**inputs)

        # decode fn gets in token ids and additional kwargs,
        # and return logits over next tokens
        def _decode_fn(
            token_ids: torch.Tensor,
            **kwargs: Any
        ) -> torch.Tensor:
            assert isinstance(self.model, PretrainedEncoderDecoder)
            dec = self.model.decode(
                token_ids,
                kwargs.pop("memory"),
                kwargs.pop("memory_padding_mask"),
            )
            return dec

        def _kwargs_select_fn(
            kwargs: Dict[str, Any],
            mask: torch.Tensor
        ) -> Dict[str, Any]:
            return {
                "memory": kwargs["memory"][mask],
                "memory_padding_mask": kwargs["memory_padding_mask"][mask]
            }

        is_beam = self._strategy == "beam" and self._beam_width > 1
        is_sample = self._strategy == "sample" and self._sample_top_k > 1

        batch_size = len(inputs["token_ids"])
        initial_token_ids = [self._initial_token_ids] * batch_size
        stop_fn = eos_stop_fn(self._eos_token_id)
        if is_beam:
            # TODO: implement indices select fn with beams
            indices_select_fn = beam_select_fn(self._beam_width)

            outputs = beam_search(
                decode_fn=_decode_fn,
                initial_token_ids=initial_token_ids,
                vocab_size=self.output_tokenizer.vocab_size(),
                pad_token_id=self.output_tokenizer.pad_token_id(),
                max_length=self.max_length,
                stop_fn=stop_fn,
                device=self.device,
                normalize_by_length=True,
                alpha=1.0,
                beam_width=self._beam_width,
                select_fn=indices_select_fn,
                kwargs_select_fn=_kwargs_select_fn,
                memory=enc,
                memory_padding_mask=inputs["padding_mask"],
            )
            return [output[0].token_ids for output in outputs]
        else:
            if self.has_indices:
                decoding_state = self._initial_decoding_state(batch_size)
                idx_select_fn = self._index_select_fn(decoding_state)
            else:
                idx_select_fn: IdxSelectFn = sample_select_fn(
                    self._sample_top_k
                ) if is_sample else greedy_select_fn()

            return search(
                decode_fn=_decode_fn,
                initial_token_ids=initial_token_ids,
                pad_token_id=self.output_tokenizer.pad_token_id(),
                max_length=self.max_length,
                select_fn=idx_select_fn,
                stop_fn=stop_fn,
                device=self.device,
                kwargs_select_fn=_kwargs_select_fn,
                memory=enc,
                memory_padding_mask=inputs["padding_mask"],
            )

    def _process_results(
        self,
        items: List[data.InferenceItem],
        outputs: List[Any],
    ) -> data.InferenceData:
        num_pfx = self.output_tokenizer.num_prefix_tokens()
        num_sfx = self.output_tokenizer.num_suffix_tokens()
        merged = "".join(
            self.output_tokenizer.de_tokenize(output[num_pfx:-num_sfx], False)
            for output in outputs
        )
        processed = postprocess_output(merged)
        return data.InferenceData(processed, language=items[0].data.language)

    def set_inference_options(
        self,
        strategy: str = "greedy",
        beam_width: int = 5,
        sample_top_k: int = 5
    ) -> None:
        assert strategy in ["greedy", "beam", "sample"]
        self._strategy = strategy
        self._beam_width = beam_width
        self._sample_top_k = sample_top_k

    def set_indices(
        self,
        entity_index: Optional[Union[str, prefix.Vec]] = None,
        property_index: Optional[Union[str, prefix.Vec]] = None,
    ) -> None:
        if entity_index is not None:
            if isinstance(entity_index, str):
                entity_index = prefix.Vec.load(entity_index)
            self._entity_index = entity_index
            self._entity_index.set_continuations(
                self._output_conts
            )
        if property_index is not None:
            if isinstance(property_index, str):
                property_index = prefix.Vec.load(property_index)
            self._property_index = property_index
            self._property_index.set_continuations(
                self._output_conts
            )
        if self.has_indices:
            self._initial_ent_conts = [
                len(cont.lstrip()) > 0
                and (self._entity_index.contains(cont)
                     or self._entity_index.contains(cont.strip()))
                for cont in self._output_conts
            ]
            self._initial_prop_conts = [
                len(cont.lstrip()) > 0
                and (self._property_index.contains(cont)
                     or self._property_index.contains(cont.strip()))
                for cont in self._output_conts
            ]
        else:
            self._initial_ent_conts = [True] * len(self._output_conts)
            self._initial_prop_conts = [True] * len(self._output_conts)

    @property
    def has_indices(self) -> bool:
        return self._entity_index is not None \
            and self._property_index is not None

    def get_indices(self) -> Optional[Tuple[prefix.Vec, prefix.Vec]]:
        if self.has_indices:
            return self._entity_index, self._property_index
        return None

    def generate(
        self,
        inputs: Union[str, List[str]],
        languages: Optional[List[str]] = None,
        batch_size: int = 16,
        batch_max_tokens: Optional[int] = None,
        sort: bool = True,
        num_threads: Optional[int] = None,
        raw: bool = False,
        with_labels: bool = False,
        show_progress: bool = False,
    ) -> Union[str, List[str]]:
        input_is_string = isinstance(inputs, str)
        assert (
            input_is_string
            or (
                isinstance(inputs, list)
                and all(isinstance(ipt, str) for ipt in inputs)
            )
        ), "input needs to be a string or a list of strings"

        if input_is_string:
            inputs = [inputs]

        if languages is not None:
            if input_is_string:
                assert isinstance(languages, str), \
                    "language must be a string if specified and " \
                    "input is a string"
                langs = [languages]
            else:
                assert (
                    isinstance(languages, list)
                    and all(isinstance(lang, str) for lang in languages)
                    and len(languages) == len(inputs)
                ), "expected same number of languages as inputs"
                langs = languages
        else:
            langs = [None] * len(inputs)

        loader = self._get_loader(
            (data.InferenceData(s, language=l) for s, l in zip(inputs, langs)),
            batch_size,
            batch_max_tokens,
            sort,
            num_threads,
        )

        progress_desc = f"Generating SPARQL from " \
            f"{len(inputs)} sequences"
        progress_total = len(inputs)
        progress_unit = "seq"

        if sort:
            outputs = self._correct_sorted(
                loader,
                progress_desc,
                progress_total,
                progress_unit,
                show_progress
            )
        else:
            outputs = self._correct_unsorted(
                loader,
                progress_desc,
                progress_total,
                progress_unit,
                show_progress
            )

        if input_is_string:
            return self._prepare_sparql_query(
                next(iter(outputs)).text,
                raw,
                with_labels
            )
        else:
            return [
                self._prepare_sparql_query(output.text, raw, with_labels)
                for output in outputs
            ]

    def _prepare_sparql_query(
        self,
        query: str,
        raw: bool = False,
        with_labels: bool = False
    ) -> str:
        if raw or not self.has_indices:
            return query
        return prepare_sparql_query(
            query,
            self._entity_index,
            self._property_index,
            with_labels
        )

    def generate_iter(
        self,
        iter: Iterator[Tuple[str, Optional[str]]],
        batch_size: int = 16,
        batch_max_tokens: Optional[int] = None,
        sort: bool = True,
        num_threads: Optional[int] = None,
        raw: bool = False,
        with_labels: bool = False,
        show_progress: bool = False,
    ) -> Union[Iterator[str], Iterator[data.InferenceData]]:
        loader = self._get_loader(
            (data.InferenceData(s, language=l) for s, l in iter),
            batch_size,
            batch_max_tokens,
            sort,
            num_threads,
        )

        progress_desc = "Generating SPARQL from iterator"
        progress_total = sys.maxsize
        progress_unit = "byte"

        if sort:
            output = self._correct_sorted(
                loader,
                progress_desc,
                progress_total,
                progress_unit,
                show_progress
            )
        else:
            output = self._correct_unsorted(
                loader,
                progress_desc,
                progress_total,
                progress_unit,
                show_progress
            )

        if raw:
            yield from output
        else:
            yield from (
                self._prepare_sparql_query(
                    data.text,
                    with_labels=with_labels
                )
                for data in output
            )

    def generate_file(
            self,
            input_file: str,
            input_file_format: str = "text",
            output_file: Optional[Union[TextIOWrapper, str]] = None,
            output_file_format: str = "text",
            language: Optional[str] = None,
            batch_size: int = 16,
            batch_max_tokens: Optional[int] = None,
            sort: bool = True,
            num_threads: Optional[int] = None,
            raw: bool = False,
            with_labels: bool = False,
            show_progress: bool = False
    ) -> Optional[Iterator[str]]:
        assert input_file_format in self.supported_input_formats(), \
            f"unsupported input file format {input_file_format}, \
        must be one of {self.supported_input_formats()}"
        assert output_file_format in self.supported_output_formats(), \
            f"unsupported output file format {output_file_format}, \
        must be one of 'text' or 'text_language'"
        loader = self._get_loader(
            ([input_file], [language] if language is not None else None),
            batch_size,
            batch_max_tokens,
            sort,
            num_threads,
            file_format=input_file_format,
        )

        file_name = input_file \
            if len(input_file) < 32 else f"...{input_file[-29:]}"
        progress_desc = f"Generating SPARQL from {file_name}"
        progress_total = os.path.getsize(input_file)
        progress_unit = "byte"

        if sort:
            outputs = iter(self._correct_sorted(
                loader,
                progress_desc,
                progress_total,
                progress_unit,
                show_progress
            ))
        else:
            outputs = self._correct_unsorted(
                loader,
                progress_desc,
                progress_total,
                progress_unit,
                show_progress
            )

        if output_file is not None:
            output_file_is_str = isinstance(output_file, str)
            if output_file_is_str:
                output_dir = os.path.dirname(output_file)
                if output_dir != "":
                    os.makedirs(output_dir, exist_ok=True)
                output_file = open(output_file, "w", encoding="utf8")

            for output in outputs:
                output.text = self._prepare_sparql_query(
                    output.text,
                    raw,
                    with_labels
                )
                output_file.write(f"{output.to_str(output_file_format)}\n")

            if output_file_is_str:
                output_file.close()

        else:
            return (
                self._prepare_sparql_query(output.text, raw, with_labels)
                for output in outputs
            )

    def set_precision(self, precision: str) -> None:
        training_precision = self.cfg["train"].get(
            "mixed_precision_dtype", "fp32")
        if precision != "fp32" and precision != training_precision:
            self.logger.warning(
                f"this model was trained with {training_precision} precision, "
                "inference with {precision} might give unexpected results"
            )
        return super().set_precision(precision)
