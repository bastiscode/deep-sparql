from io import TextIOWrapper
import os
import copy
import sys
from typing import Any, Dict, List, Tuple, Optional, Union, Iterator, Callable

import torch
from torch import nn
from peft import get_peft_model

from text_utils import data, tokenization, continuations
from text_utils.api.processor import ModelInfo, TextProcessor
from text_utils.api.utils import (
    Device,
    device_info,
    get_devices,
    get_peft_config
)
from text_utils.inference import (
    Beam,
    BeamSelectFn,
    IdxSelectFn,
    beam_select_fn as default_beam_select_fn,
    greedy_select_fn,
    sample_select_fn,
    search,
    beam_search
)

from deep_sparql import vector
from deep_sparql.model import (
    Model,
    PretrainedDecoder,
    PretrainedEncoderDecoder,
    model_from_config
)
from deep_sparql.utils import (
    KNOWLEDGE_GRAPHS,
    REP,
    format_input,
    clean_sparql,
    format_sparql,
    get_completions,
    prepare_sparql_query,
    special_token_or_token_ids,
    longest_overlap
)

_BASE_URL = "https://ad-publications.informatik.uni-freiburg.de/" \
    "ACL_whitespace_procession_transformer_BHW_2023.materials"
_NAME_TO_ZIP = {
}


class DecodingState:
    def __init__(
        self,
        initial_token_ids: List[int],
        ent_start_ids: List[int],
        ent_stop_ids: List[int],
        prop_start_ids: List[int],
        prop_stop_ids: List[int],
        entity_index: continuations.Continuations,
        property_index: continuations.Continuations
    ):
        self._token_ids = initial_token_ids
        self._initial_length = len(self._token_ids)

        # state tracking
        self._state: str | None = None
        self._start_idx = 0
        self._ent_start = ent_start_ids
        self._ent_stop = ent_stop_ids
        self._prop_start = prop_start_ids
        self._prop_stop = prop_stop_ids
        self._overlap_token_id: int | None = None

        # value tracking
        self._value: str | None = None
        self._decoded: list[tuple[str, str]] = []

        # indices
        self._ent_index = entity_index
        self._prop_index = property_index
        self._sub_index: continuations.Continuations | None = None

    def is_ent_start(self) -> bool:
        return (
            self._token_ids[-len(self._ent_start):] == self._ent_start
            and self._state is None
        )

    def is_ent_stop(self) -> bool:
        return (
            self.is_ent() and
            self._token_ids[-len(self._ent_stop):] == self._ent_stop
        )

    def is_prop_start(self) -> bool:
        return (
            self._token_ids[-len(self._prop_start):] == self._prop_start
            and self._state is None
        )

    def is_prop_stop(self) -> bool:
        return (
            self.is_prop() and
            self._token_ids[-len(self._prop_stop):] == self._prop_stop
        )

    def is_ent(self) -> bool:
        return self._state == "ent"

    def is_prop(self) -> bool:
        return self._state == "prop"

    def has_value(self) -> bool:
        return self._value is not None

    def set_overlap(self, token_id: int | None):
        self._overlap_token_id = token_id

    def is_obj(self) -> bool:
        return self._state is not None

    def get_obj_token_ids(self) -> list[int]:
        return self._token_ids[self._start_idx:]

    def get_index(self) -> continuations.Continuations | None:
        if self.is_ent():
            return self._sub_index or self._ent_index
        elif self.is_prop():
            return self._sub_index or self._prop_index
        else:
            return None

    def calc_overlap(self) -> tuple[int, int]:
        if self.is_ent():
            overlap = len(longest_overlap(self._token_ids, self._ent_stop))
            self._overlap_token_id = self._ent_stop[overlap]
            return overlap, self._overlap_token_id
        elif self.is_prop():
            overlap = len(longest_overlap(self._token_ids, self._prop_stop))
            self._overlap_token_id = self._prop_stop[overlap]
            return overlap, self._overlap_token_id
        else:
            raise RuntimeError(
                "calc overlap should only be called when "
                "decoding an entity or property"
            )

    def calc_sub_index(
        self,
        sparql_fn: Callable[[list[int]], str],
        kg: str = "wikidata",
        qlever_endpoint: str | None = None,
        lang: str = "en",
        max_size: int = 65536
    ):
        if (
            self._sub_index is not None
            or self._state is None
            or len(self._decoded) == 0
            or len(self._token_ids) != self._start_idx
        ):
            return

        num_start_ids = len(
            self._ent_start if self._state == "ent" else self._prop_start
        )
        # get sparql up to current entity / property
        sparql = sparql_fn(
            self._token_ids[self._initial_length:-num_start_ids]
        )
        if self._state == "ent":
            index = self._ent_index
            # we need to differentiate between subject and object
            # for entities, we do that by looking whether the
            # entity is immediately preceded by brackets, braces
            # or a dot.
            sparql = sparql.rstrip()
            current_state = "subject" if len(sparql) == 0 or sparql[-1] in [
                ")" "{", "}", "."
            ] else "object"
        else:
            index = self._prop_index
            current_state = "predicate"

        # get valid completions for this partial sparql query
        values = get_completions(
            sparql,
            current_state,
            self._ent_index,
            self._prop_index,
            kg,
            qlever_endpoint,
            lang,
            max_size
        )
        if values is None or len(values) == 0:
            return
        self._sub_index = index.get_sub_index_by_values(values)
        self._sub_index.compute_memo(max_depth=3)

    def add(
        self,
        token_id: int,
        value: str | None,
    ):
        self._token_ids.append(token_id)
        if self._overlap_token_id == token_id:
            # set or keep the current value only if we are in overlap
            self._value = value or self._value
        else:
            self._value = None
        self._overlap_token_id = None
        if self.is_ent_stop() or self.is_prop_stop():
            self._decoded.append((self._state, self._value))  # type: ignore
            self._state = None
            self._value = None
            self._sub_index = None
        elif self.is_ent_start():
            self._state = "ent"
            self._start_idx = len(self._token_ids)
        elif self.is_prop_start():
            self._state = "prop"
            self._start_idx = len(self._token_ids)

    def __deepcopy__(self, memo) -> "DecodingState":
        state = {
            name: copy.deepcopy(val, memo)
            for name, val in self.__dict__.items()
            if name not in {"_ent_index", "_prop_index", "_sub_index"}
        }
        copied = DecodingState(
            [], [], [], [], [], None, None
        )
        copied.__dict__ = {
            **state,
            "_ent_index": self._ent_index,
            "_prop_index": self._prop_index,
            "_sub_index": self._sub_index
        }
        return copied


class SPARQLGenerator(TextProcessor):
    task = "SPARQL generation"

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
    def _model_from_config(
        cls,
        cfg: Dict[str, Any],
        _: Device
    ) -> nn.Module:
        model = model_from_config(cfg["model"])
        peft = cfg["train"].get("peft", None)
        if peft is not None:
            peft_cfg = get_peft_config(peft)
            model.model = get_peft_model(
                model.model,  # type: ignore
                peft_cfg
            )
        return model

    @property
    def max_length(self) -> int:
        return (
            self._max_length
            or self.cfg["train"]["data"].get("max_length", 512)
        )

    @property
    def context_length(self) -> int:
        raise NotImplementedError

    def supported_languages(self) -> Optional[List[str]]:
        lang_cfg = self.cfg["input_tokenizer"].get("language")
        if lang_cfg is None:
            return None
        return lang_cfg["languages"]

    def __init__(
        self,
        model: Model,
        cfg: Dict[str, Any],
        device: Device
    ) -> None:
        super().__init__(model, cfg, device)
        assert isinstance(model, (PretrainedDecoder, PretrainedEncoderDecoder))
        self.logger.debug(f"got model config:\n{self.cfg['model']}")
        self.logger.info(
            f"running {self.name} SPARQL generator "
            f"on devices {[device_info(d) for d in self.devices]}"
        )
        self.input_tokenizer = tokenization.Tokenizer.from_config(
            self.cfg["input_tokenizer"]
        )
        assert "output_tokenizer" in self.cfg
        self.output_tokenizer = tokenization.Tokenizer.from_config(
            self.cfg["output_tokenizer"]
        )

        # some options for inference
        self._is_encoder_decoder = isinstance(
            self.model,
            PretrainedEncoderDecoder
        )
        eos_token = self.cfg["output_tokenizer"]["eos_token"]
        self._eos_token_id = self.output_tokenizer.special_token_to_id(
            eos_token
        )
        tokenizer_type = self.cfg["output_tokenizer"]["type"]
        boe_token, self._boe_ids = special_token_or_token_ids(
            " <boe>",
            self.output_tokenizer,
            tokenizer_type
        )
        eoe_token, self._eoe_ids = special_token_or_token_ids(
            "<eoe>",
            self.output_tokenizer,
            tokenizer_type
        )
        bop_token, self._bop_ids = special_token_or_token_ids(
            " <bop>",
            self.output_tokenizer,
            tokenizer_type
        )
        eop_token, self._eop_ids = special_token_or_token_ids(
            "<eop>",
            self.output_tokenizer,
            tokenizer_type
        )
        bob_token, _ = special_token_or_token_ids(
            " <bob>",
            self.output_tokenizer,
            tokenizer_type
        )
        eob_token, _ = special_token_or_token_ids(
            "<eob>",
            self.output_tokenizer,
            tokenizer_type
        )
        bov_token, _ = special_token_or_token_ids(
            " <bov>",
            self.output_tokenizer,
            tokenizer_type
        )
        eov_token, _ = special_token_or_token_ids(
            "<eov>",
            self.output_tokenizer,
            tokenizer_type
        )
        self._bracket_special_tokens = (
            (bob_token, "{"),
            (eob_token, "}"),
        )
        self._var_special_tokens = ((bov_token, eov_token), ("<bov>", "<eov>"))
        self._ent_special_tokens = ((boe_token, eoe_token), ("<boe>", "<eoe>"))
        self._prop_special_tokens = (
            (bop_token, eop_token), ("<bop>", "<eop>")
        )
        self._strategy = "greedy"
        self._beam_width = 5
        self._sample_top_k = 5
        self._use_cache = True
        self._subgraph_constraining = False
        self._qlever_endpoint: str | None = None
        self._kg = "wikidata"
        self._lang = "en"
        self._max_length = None
        assert self._eos_token_id is not None

        self._entity_index = None
        self._property_index = None
        self._example_index = None

        self._continuations = [
            self.output_tokenizer.de_tokenize(
                [self._eos_token_id, i, self._eos_token_id],
                False
            )[len(eos_token):-len(eos_token)].encode("utf8")
            for i in range(self.output_tokenizer.vocab_size())
        ]

        def _sparql_from_token_ids(
            token_ids: list[int]
        ) -> str:
            raw = self.output_tokenizer.de_tokenize(token_ids, False)
            return clean_sparql(
                raw,
                self._bracket_special_tokens,
                (
                    self._var_special_tokens,
                    self._ent_special_tokens,
                    self._prop_special_tokens
                )
            )
        self._sparql_from_token_ids = _sparql_from_token_ids

    def to(self, device: Device) -> "SPARQLGenerator":
        self.devices = get_devices(device)
        assert isinstance(self.model, Model)
        self.model = self.model.distribute(self.devices)
        return self

    def _build_inference_loader_config(self) -> Dict[str, Any]:
        return {
            "tokenizer_config": self.cfg["input_tokenizer"],
            "window_config": {"type": "full"}
        }

    def _prepare_batch(self, batch: data.InferenceBatch) -> Dict[str, Any]:
        token_ids_np, pad_mask_np, lengths, *_ = batch.tensors()
        if self._is_encoder_decoder:
            return {
                "token_ids": torch.from_numpy(token_ids_np).to(
                    non_blocking=True,
                    device=self.devices[0]
                ),
                "padding_mask": torch.from_numpy(pad_mask_np).to(
                    non_blocking=True,
                    device=self.devices[0]
                )
            }
        else:
            return {
                "token_ids": token_ids_np,
                "lengths": lengths
            }

    def _initial_decoding_state(
        self,
        initial_token_ids: List[int]
    ) -> DecodingState:
        # keep track of decoding state
        # None --> nothing
        # (ent, idx) --> entity starting at idx
        # (prop, idx) --> property starting at idx
        assert self.has_kg_indices
        return DecodingState(
            initial_token_ids,
            self._boe_ids,
            self._eoe_ids,
            self._bop_ids,
            self._eop_ids,
            self._entity_index,
            self._property_index
        )

    def _update_cont_mask_and_values(
        self,
        cont_mask: torch.Tensor,
        values: list[str | None],
        decoding_states: list[DecodingState],
    ) -> tuple[torch.Tensor, list[str | None]]:
        assert cont_mask.shape[0] == len(values) == len(decoding_states)
        for i, state in enumerate(decoding_states):
            index = state.get_index()
            if index is None:
                continue
            token_ids = state.get_obj_token_ids()
            prefix = self.output_tokenizer.de_tokenize(
                token_ids,
                False
            ).lstrip().encode("utf8")
            mask, value = index.continuation_indices(prefix)
            overlap, overlap_token_id = state.calc_overlap()
            valid_cont = (
                (overlap == 0 and value is not None)
                or
                (overlap > 0 and state.has_value())
            )
            if valid_cont:
                mask.append(overlap_token_id)
            cont_mask[i, :] = False
            cont_mask[i, torch.tensor(mask, dtype=torch.long)] = True
            values[i] = value
        return cont_mask, values

    def _index_select_fn(
        self,
        decoding_states: List[DecodingState],
    ) -> IdxSelectFn:
        def _fn(
            scores: torch.Tensor,
            indices: List[int]
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            conts = torch.ones(
                *scores.shape,
                dtype=torch.bool
            )
            conts[..., self.output_tokenizer.vocab_size():] = False
            values: list[str | None] = [None for _ in range(len(conts))]

            conts, values = self._update_cont_mask_and_values(
                conts,
                values,
                decoding_states
            )

            scores[torch.logical_not(conts)] = float("-inf")
            token_ids = torch.argmax(scores, -1)
            scores = torch.gather(scores, -1, token_ids[:, None]).squeeze(-1)

            # update decoding states
            for idx, token_id, value in zip(
                indices,
                token_ids.tolist(),
                values
            ):
                decoding_states[idx].add(token_id, value)
                if self._subgraph_constraining:
                    decoding_states[idx].calc_sub_index(
                        self._sparql_from_token_ids,
                        self._kg,
                        self._qlever_endpoint,
                        self._lang
                    )

            return token_ids, scores

        return _fn

    def _beam_select_fn(self) -> BeamSelectFn:
        def _fn(
            scores: torch.Tensor,
            batch_beams: List[List[Beam]],
            _: List[int]
        ) -> List[List[Beam]]:
            conts = torch.ones(
                *scores.shape,
                dtype=torch.bool
            )
            conts[..., self.output_tokenizer.vocab_size():] = False
            values: list[str | None] = [None for _ in range(len(conts))]

            decoding_states = []
            for beams in batch_beams:
                for beam in beams:
                    if "state" not in beam.info:
                        beam.info["state"] = DecodingState(
                            beam.token_ids,
                            self._boe_ids,
                            self._eoe_ids,
                            self._bop_ids,
                            self._eop_ids,
                            self._entity_index,
                            self._property_index
                        )
                    decoding_states.append(beam.info["state"])

            conts, values = self._update_cont_mask_and_values(
                conts,
                values,
                decoding_states
            )

            scores[torch.logical_not(conts)] = float("-inf")

            num_beams = [len(b) for b in batch_beams]
            assert scores.ndim == 2 and scores.shape[0] == sum(num_beams)
            k = min(self._beam_width, scores.shape[1])
            top_k = torch.topk(scores, k, dim=1)
            batch_start = 0
            batch_candidates = []
            for beams, num in zip(  # type: ignore
                batch_beams,
                num_beams
            ):
                top_k_indices = top_k.indices[batch_start:batch_start + num]
                top_k_log_probs = top_k.values[batch_start:batch_start + num]
                top_k_values = values[batch_start:batch_start + num]
                assert len(top_k_log_probs) == len(top_k_values)
                batch_start += num
                # create candidates
                candidates = []
                for idx, (token_ids, log_probs) in enumerate(zip(
                    top_k_indices.tolist(),  # type: ignore
                    top_k_log_probs.tolist()
                )):
                    for token_id, log_p in zip(token_ids, log_probs):
                        candidates.append((
                            idx,
                            token_id,
                            log_p
                        ))
                # sort candidates by score
                candidates = sorted(
                    candidates,
                    key=lambda item: -(beams[item[0]].log_prob + item[2]),
                )[:2 * self._beam_width]
                # convert candidates to beams
                candidate_beams = []
                for idx, token_id, log_p in candidates:
                    beam = Beam.from_beam(beams[idx], log_p, token_id)
                    state: DecodingState = beam.info["state"]
                    state.add(token_id, top_k_values[idx])
                    if self._subgraph_constraining:
                        state.calc_sub_index(
                            self._sparql_from_token_ids,
                            self._kg,
                            self._qlever_endpoint,
                            self._lang
                        )
                    candidate_beams.append(beam)
                batch_candidates.append(candidate_beams)

            return batch_candidates

        return _fn

    def _inference(
        self,
        inputs: Dict[str, Any],
    ) -> list[Any]:
        batch_size = len(inputs["token_ids"])
        inference_kwargs = {}
        if self._is_encoder_decoder:
            enc = self.model.encode(**inputs)
            inference_kwargs["memory"] = enc
            inference_kwargs["memory_padding_mask"] = inputs["padding_mask"]
            token_ids = self.output_tokenizer.tokenize("").token_ids
            num_pfx = self.output_tokenizer.num_prefix_tokens()
            initial_token_ids = [
                list(token_ids[:num_pfx])
                for _ in range(batch_size)
            ]
        else:
            initial_token_ids = [
                list(token_ids[:length])
                for token_ids, length in zip(
                    inputs["token_ids"],
                    inputs["lengths"]
                )
            ]

        # decode fn gets in token ids and additional kwargs,
        # and return logits over next tokens and additional info
        def _decode_fn(
            token_ids: torch.Tensor,
            **kwargs: Any
        ) -> Tuple[torch.Tensor, Dict[str, Any]]:
            if self._is_encoder_decoder:
                assert isinstance(self.model, PretrainedEncoderDecoder)
                dec, cache = self.model.decode(
                    token_ids,
                    kwargs["lengths"],
                    kwargs["memory"],
                    kwargs["memory_padding_mask"],
                    kwargs.get("kv_cache", None),
                    self._use_cache
                )
            else:
                assert isinstance(self.model, PretrainedDecoder)
                dec, cache = self.model.decode(
                    token_ids,
                    kwargs["lengths"],
                    kwargs.get("kv_cache", None),
                    self._use_cache
                )
            return dec, {"kv_cache": cache}

        def _kwargs_select_fn(
            kwargs: Dict[str, Any],
            mask: torch.Tensor
        ) -> Dict[str, Any]:
            if self._is_encoder_decoder:
                return {
                    "memory": kwargs["memory"][mask],
                    "memory_padding_mask": kwargs["memory_padding_mask"][mask],
                }
            return {}

        def _kwargs_update_fn(
            kwargs: Dict[str, Any],
            info: Dict[str, Any],
            mask: torch.Tensor
        ) -> None:
            kv_cache = info.get("kv_cache", None)
            if kv_cache is None:
                return
            kwargs["kv_cache"] = tuple(
                tuple(c[mask.to(c.device)] for c in cache)
                for cache in info["kv_cache"]
            )

        is_beam = self._strategy == "beam" and self._beam_width > 1
        is_sample = self._strategy == "sample" and self._sample_top_k > 1

        if is_beam:
            if self.has_kg_indices:
                beam_select_fn = self._beam_select_fn()
            else:
                beam_select_fn = default_beam_select_fn(self._beam_width)

            def beam_stop_fn(beam: Beam, _: int) -> bool:
                return beam.token_ids[-1] == self._eos_token_id

            outputs = beam_search(
                decode_fn=_decode_fn,
                initial_token_ids=initial_token_ids,
                pad_token_id=self.output_tokenizer.pad_token_id(),
                max_length=self.max_length,
                stop_fn=beam_stop_fn,
                device=self.devices[0],
                normalize_by_length=True,
                alpha=1.0,
                beam_width=self._beam_width,
                select_fn=beam_select_fn,
                kwargs_select_fn=_kwargs_select_fn,
                kwargs_update_fn=_kwargs_update_fn,
                **inference_kwargs
            )
            return [output[0].token_ids for output in outputs]

        else:
            if self.has_kg_indices:
                decoding_states = [
                    self._initial_decoding_state(token_ids)
                    for token_ids in initial_token_ids
                ]
                select_fn = self._index_select_fn(decoding_states)
            else:
                select_fn: IdxSelectFn = sample_select_fn(
                    self._sample_top_k
                ) if is_sample else greedy_select_fn()

            def stop_fn(token_ids: torch.Tensor, _: List[int]) -> torch.Tensor:
                return token_ids == self._eos_token_id

            return search(
                decode_fn=_decode_fn,
                initial_token_ids=initial_token_ids,
                pad_token_id=self.output_tokenizer.pad_token_id(),
                max_length=self.max_length,
                select_fn=select_fn,
                stop_fn=stop_fn,
                device=self.devices[0],
                kwargs_select_fn=_kwargs_select_fn,
                kwargs_update_fn=_kwargs_update_fn,
                **inference_kwargs
            )

    def _process_results(
        self,
        items: List[data.InferenceItem],
        outputs: List[Any],
    ) -> data.InferenceData:
        assert len(outputs) == 1
        return data.InferenceData(
            self._sparql_from_token_ids(outputs[0][:-1]),
            language=items[0].data.language
        )

    def set_inference_options(
        self,
        strategy: str = "greedy",
        beam_width: int = 5,
        sample_top_k: int = 5,
        subgraph_constraining: bool = False,
        kg: str = "wikidata",
        lang: str = "en",
        max_length: int | None = None,
        use_cache: bool = True,
        qlever_endpoint: str | None = None
    ) -> None:
        assert strategy in ["greedy", "beam", "sample"]
        self._strategy = strategy
        self._beam_width = beam_width
        self._sample_top_k = sample_top_k
        self._subgraph_constraining = subgraph_constraining
        assert kg in KNOWLEDGE_GRAPHS
        self._kg = kg
        self._qlever_endpoint = qlever_endpoint
        self._lang = lang
        self._max_length = max_length
        self._use_cache = use_cache

    def set_indices(
        self,
        entity_index: Optional[Union[str, continuations.Continuations]] = None,
        property_index: Optional[Union[str,
                                       continuations.Continuations]] = None,
        example_index: Optional[Union[str, vector.Index]] = None,
    ) -> None:
        if entity_index is not None:
            if isinstance(entity_index, str):
                entity_index = continuations.Continuations.load_with_continuations(
                    entity_index,
                    self._continuations
                )
            self._entity_index = entity_index

        if property_index is not None:
            if isinstance(property_index, str):
                property_index = continuations.Continuations.load_with_continuations(
                    property_index,
                    self._continuations
                )
            self._property_index = property_index

        if example_index is not None:
            if isinstance(example_index, str):
                example_index = vector.Index.load(example_index)
            self._example_index = example_index

    @ property
    def has_kg_indices(self) -> bool:
        return self._entity_index is not None \
            and self._property_index is not None

    def get_kg_indices(self) -> Optional[Tuple[continuations.Continuations, continuations.Continuations]]:
        if self.has_kg_indices:
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
        show_progress: bool = False,
        n_examples: int = 3,
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
            (
                data.InferenceData(
                    s if raw else
                    self.prepare_questions([s], n_examples)[0],
                    language=l
                )
                for s, l in zip(inputs, langs)
            ),
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
            outputs = self._process_sorted(
                loader,
                progress_desc,
                progress_total,
                progress_unit,
                show_progress
            )
        else:
            outputs = self._process_unsorted(
                loader,
                progress_desc,
                progress_total,
                progress_unit,
                show_progress
            )

        if input_is_string:
            output = next(iter(outputs)).text
            return output if raw else self.prepare_sparql_query(output)
        else:
            return [
                output.text if raw
                else self.prepare_sparql_query(output.text)
                for output in outputs
            ]

    def prepare_sparql_query(
        self,
        output: str,
        pretty: bool = False,
        post_fn: Callable[[str, REP, REP, REP], str] | None = None
    ) -> str:
        if not self.has_kg_indices:
            return format_sparql(output, pretty=pretty)
        return prepare_sparql_query(
            output,
            self._entity_index,
            self._property_index,
            kg=self._kg,
            pretty=pretty,
            post_fn=post_fn
        )

    def prepare_questions(
        self,
        questions: List[str],
        n_examples: int = 3,
        batch_size: int = 16,
    ) -> List[str]:
        if self._example_index is not None and n_examples > 0:
            examples = vector.get_nearest_neighbors(
                questions,
                self._example_index,
                n_examples,
                batch_size,
            )
        else:
            examples = [[]] * len(questions)

        return [
            format_input(
                q,
                [ex_str for ex_str, _ in ex],
                self._kg,
            ) + ":" * (not self._is_encoder_decoder)
            for q, ex in zip(questions, examples)
        ]

    def generate_iter(
        self,
        iter: Iterator[Tuple[str, Optional[str]]],
        batch_size: int = 16,
        batch_max_tokens: Optional[int] = None,
        sort: bool = True,
        num_threads: Optional[int] = None,
        raw: bool = False,
        show_progress: bool = False,
        n_examples: int = 3,
    ) -> Union[Iterator[str], Iterator[data.InferenceData]]:
        loader = self._get_loader(
            (
                data.InferenceData(
                    s if raw else
                    self.prepare_questions([s], n_examples)[0],
                    language=l
                )
                for s, l in iter
            ),
            batch_size,
            batch_max_tokens,
            sort,
            num_threads,
        )

        progress_desc = "Generating SPARQL from iterator"
        progress_total = sys.maxsize
        progress_unit = "byte"

        if sort:
            output = self._process_sorted(
                loader,
                progress_desc,
                progress_total,
                progress_unit,
                show_progress
            )
        else:
            output = self._process_unsorted(
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
                self.prepare_sparql_query(data.text)
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
        show_progress: bool = False,
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
            outputs = iter(self._process_sorted(
                loader,
                progress_desc,
                progress_total,
                progress_unit,
                show_progress
            ))
        else:
            outputs = self._process_unsorted(
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
                if not raw:
                    output.text = self.prepare_sparql_query(
                        output.text,
                    )
                output_file.write(f"{output.to_str(output_file_format)}\n")

            if output_file_is_str:
                output_file.close()

        else:
            return (
                output.text if raw else
                self.prepare_sparql_query(output.text)
                for output in outputs
            )
