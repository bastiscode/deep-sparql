from io import TextIOWrapper
import os
# import time
import sys
from typing import Any, Dict, List, Tuple, Optional, Union, Iterator, Callable

import torch
from torch import nn
from peft import get_peft_model

from text_correction_utils import data, tokenization, prefix
from text_correction_utils.api.corrector import ModelInfo
from text_correction_utils.api import corrector
from text_correction_utils.api.utils import device_info, get_peft_config
from text_correction_utils.inference import (
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
    format_input,
    postprocess_output,
    prepare_sparql_query,
    special_token_or_token_ids,
    longest_overlap
)

_BASE_URL = "https://ad-publications.informatik.uni-freiburg.de/" \
    "ACL_whitespace_correction_transformer_BHW_2023.materials"
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
    ):
        assert len(initial_token_ids) > 0
        self._token_ids = initial_token_ids
        self._state: Optional[str] = None
        self._start_idx = 0
        self._ent_start = ent_start_ids
        self._ent_stop = ent_stop_ids
        self._prop_start = prop_start_ids
        self._prop_stop = prop_stop_ids
        self._overlap_token_id: Optional[int] = None
        self._has_value = False

    def is_ent_start(self) -> bool:
        return (
            self._token_ids[-len(self._ent_start):] == self._ent_start
            and self._state is None
        )

    def is_ent_stop(self) -> bool:
        return (
            self._token_ids[-len(self._ent_stop)] == self._ent_stop
            and self.is_ent()
        )

    def is_prop_start(self) -> bool:
        return (
            self._token_ids[-len(self._prop_start):] == self._prop_start
            and self._state is None
        )

    def is_prop_stop(self) -> bool:
        return (
            self._token_ids[-len(self._prop_stop)] == self._prop_stop
            and self.is_prop()
        )

    def is_ent(self) -> bool:
        return self._state == "ent"

    def is_prop(self) -> bool:
        return self._state == "prop"

    def has_value(self) -> bool:
        return self._has_value

    def set_overlap(self, token_id: Optional[int]):
        self._overlap_token_id = token_id

    def is_obj(self) -> bool:
        return self._state is not None

    def get_obj_token_ids(self) -> List[int]:
        return self._token_ids[self._start_idx:]

    def add(
        self,
        token_id: int,
    ):
        self._token_ids.append(token_id)
        self._has_value = self._overlap_token_id == token_id
        self._overlap_token_id = None
        if ((
                self.is_ent()
                and self._token_ids[-len(self._ent_stop):] == self._ent_stop
            ) or (
                self.is_prop()
                and self._token_ids[-len(self._prop_stop):] == self._prop_stop
        )):
            self._state = None
            self._has_value = False
        elif (
            self._state is None
            and self._token_ids[-len(self._ent_start):] == self._ent_start
        ):
            self._state = "ent"
            self._start_idx = len(self._token_ids)
        elif (
            self._state is None
            and self._token_ids[-len(self._prop_start):] == self._prop_start
        ):
            self._state = "prop"
            self._start_idx = len(self._token_ids)


class SPARQLGenerator(corrector.TextCorrector):
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
        model = model_from_config(
            cfg["model"],
            input_tokenizer,
            output_tokenizer
        )
        peft = cfg["train"].get("peft", None)
        if peft is not None:
            model.model = get_peft_model(
                model.model,
                get_peft_config(peft)
            )
        return model

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
        model: Model,
        cfg: Dict[str, Any],
        device: torch.device,
    ) -> None:
        super().__init__(model, cfg, device)
        assert isinstance(model, (PretrainedDecoder, PretrainedEncoderDecoder))
        self.logger.debug(f"got model config:\n{self.cfg['model']}")
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

        # some options for inference
        self._initial_token_ids = self.output_tokenizer.tokenize("")
        out_pfx = self.output_tokenizer.num_prefix_tokens()
        self._initial_token_ids = self._initial_token_ids.token_ids[:out_pfx]
        self._is_encoder_decoder = isinstance(
            self.model,
            PretrainedEncoderDecoder
        )
        self._eos_token = self.model.eos_token
        assert self._eos_token is not None
        self._eos_token_id = self.output_tokenizer.special_token_to_id(
            self._eos_token
        )
        boe_token, self._boe_ids = special_token_or_token_ids(
            "<boe>",
            self.output_tokenizer
        )
        eoe_token, self._eoe_ids = special_token_or_token_ids(
            "<eoe>",
            self.output_tokenizer
        )
        bop_token, self._bop_ids = special_token_or_token_ids(
            "<bop>",
            self.output_tokenizer
        )
        eop_token, self._eop_ids = special_token_or_token_ids(
            "<eop>",
            self.output_tokenizer
        )
        bob_token, _ = special_token_or_token_ids(
            "<bob>",
            self.output_tokenizer
        )
        eob_token, _ = special_token_or_token_ids(
            "<eob>",
            self.output_tokenizer
        )
        bov_token, _ = special_token_or_token_ids(
            "<bov>",
            self.output_tokenizer
        )
        eov_token, _ = special_token_or_token_ids(
            "<eov>",
            self.output_tokenizer
        )
        self._bracket_special_tokens = (bob_token, eob_token)
        self._var_special_tokens = (bov_token, eov_token)
        self._ent_special_tokens = (boe_token, eoe_token)
        self._prop_special_tokens = (bop_token, eop_token)
        self._strategy = "greedy"
        self._beam_width = 5
        self._sample_top_k = 5
        self._use_cache = True
        assert self._eos_token_id is not None

        self._entity_index = None
        self._property_index = None
        self._example_index = None

        self._continuations = [
            self.output_tokenizer.de_tokenize(
                [self._eos_token_id, i, self._eos_token_id],
                False
            )[len(self._eos_token):-len(self._eos_token)].encode("utf8")
            for i in range(self.output_tokenizer.vocab_size())
        ]
        self._initial_ent_mask = [True] * self.output_tokenizer.vocab_size()
        self._initial_prop_mask = [True] * self.output_tokenizer.vocab_size()

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
                    device=self.device
                ),
                "padding_mask": torch.from_numpy(pad_mask_np).to(
                    non_blocking=True,
                    device=self.device
                )
            }
        else:
            return {
                "token_ids": token_ids_np,
                "lengths": lengths
            }

    def _initial_decoding_state(
        self,
        initial_token_ids: Optional[List[int]]
    ) -> DecodingState:
        # keep track of decoding state
        # None --> nothing
        # (ent, idx) --> entity starting at idx
        # (prop, idx) --> property starting at idx
        return DecodingState(
            initial_token_ids or list(self._initial_token_ids),
            self._boe_ids,
            self._eoe_ids,
            self._bop_ids,
            self._eop_ids
        )

    def _get_indices_and_conts(
        self,
        index: prefix.Vec,
        decoding_states: List[DecodingState],
        state_indices: List[int],
        filter_fn: Callable[[DecodingState], bool],
        initial_cont_mask: List[bool],
        end_token_ids: List[int]
    ) -> Tuple[List[int], List[List[bool]]]:
        # helper fn to get valid continuations
        # from a prefix index
        indices = []
        masks = []
        prefix_indices = []
        prefixes = []
        for i, idx in enumerate(state_indices):
            if not filter_fn(decoding_states[idx]):
                continue
            token_ids = decoding_states[idx].get_obj_token_ids()
            if len(token_ids) == 0:
                masks.append(initial_cont_mask)
                indices.append(i)
                continue
            decoded = self.output_tokenizer.de_tokenize(
                token_ids,
                False
            ).lstrip().encode("utf8")
            prefixes.append(decoded)
            prefix_indices.append(i)

        cont_masks, values = index.batch_continuation_mask(prefixes)

        for cont_mask, has_value, idx in zip(
            cont_masks,
            values,
            prefix_indices
        ):
            state: DecodingState = decoding_states[state_indices[idx]]
            token_ids = state.get_obj_token_ids()
            overlap = len(longest_overlap(token_ids, end_token_ids))
            overlap_token_id = end_token_ids[overlap]
            assert overlap < len(end_token_ids)
            valid_cont = (
                (overlap == 0 and has_value)
                or
                (overlap > 0 and state.has_value())
            )
            cont_mask[overlap_token_id] = valid_cont
            state.set_overlap(overlap_token_id if valid_cont else None)

        indices.extend(prefix_indices)
        masks.extend(cont_masks)
        return indices, masks

    def _index_select_fn(
        self,
        decoding_states: List[DecodingState]
    ) -> IdxSelectFn:
        def _fn(
            scores: torch.Tensor,
            indices: List[int]
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            conts = torch.ones(
                *scores.shape,
                dtype=torch.bool
            )

            # first entities
            (
                ent_indices,
                ent_conts
            ) = self._get_indices_and_conts(
                self._entity_index,
                decoding_states,
                indices,
                lambda state: state.is_ent(),
                self._initial_ent_mask,
                self._eoe_ids
            )
            if len(ent_indices) > 0:
                ent_indices = torch.tensor(ent_indices, dtype=torch.long)
                ent_conts = torch.tensor(ent_conts, dtype=torch.bool)
                conts[ent_indices, :ent_conts.shape[-1]] = ent_conts

            # then properties
            (
                prop_indices,
                prop_conts
            ) = self._get_indices_and_conts(
                self._property_index,
                decoding_states,
                indices,
                lambda state: state.is_prop(),
                self._initial_prop_mask,
                self._eop_ids
            )
            if len(prop_indices) > 0:
                prop_indices = torch.tensor(prop_indices, dtype=torch.long)
                prop_conts = torch.tensor(prop_conts, dtype=torch.bool)
                conts[prop_indices, :prop_conts.shape[-1]] = prop_conts

            scores[torch.logical_not(conts)] = float("-inf")
            token_ids = torch.argmax(scores, -1)
            scores = torch.gather(scores, -1, token_ids[:, None]).squeeze(-1)

            # update decoding states
            for idx, token_id in zip(
                indices,
                token_ids.tolist()
            ):
                decoding_states[idx].add(token_id)

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

            decoding_states = []
            for beams in batch_beams:
                for beam in beams:
                    if "state" not in beam.info:
                        beam.info["state"] = DecodingState(
                            beam.token_ids,
                            self._boe_ids,
                            self._eoe_ids,
                            self._bop_ids,
                            self._eop_ids
                        )
                    decoding_states.append(beam.info["state"])
            indices = list(range(len(decoding_states)))

            # first entities
            ent_indices, ent_conts = self._get_indices_and_conts(
                self._entity_index,
                decoding_states,
                indices,
                lambda state: state.is_ent(),
                self._eoe_ids
            )
            if len(ent_indices) > 0:
                ent_indices = torch.tensor(ent_indices, dtype=torch.long)
                ent_conts = torch.tensor(ent_conts, dtype=torch.bool)
                conts[ent_indices, :ent_conts.shape[-1]] = ent_conts

            # then properties
            prop_indices, prop_conts = self._get_indices_and_conts(
                self._property_index,
                decoding_states,
                indices,
                lambda state: state.is_prop(),
                self._eop_ids
            )
            if len(prop_indices) > 0:
                prop_indices = torch.tensor(prop_indices, dtype=torch.long)
                prop_conts = torch.tensor(prop_conts, dtype=torch.bool)
                conts[prop_indices, :prop_conts.shape[-1]] = prop_conts

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
                indices = top_k.indices[batch_start:batch_start + num]
                values = top_k.values[batch_start:batch_start + num]
                batch_start += num
                # create candidates
                candidates = []
                for idx, (token_ids, log_probs) in enumerate(zip(
                    indices.tolist(),  # type: ignore
                    values.tolist()
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
                    state.add(token_id)
                    candidate_beams.append(beam)
                batch_candidates.append(candidate_beams)

            return batch_candidates

        return _fn

    def _inference(self, inputs: Dict[str, Any]) -> Any:
        batch_size = len(inputs["token_ids"])
        inference_kwargs = {}
        if self._is_encoder_decoder:
            self.model: PretrainedEncoderDecoder
            enc = self.model.encode(**inputs)
            inference_kwargs["memory"] = enc
            inference_kwargs["memory_padding_mask"] = inputs["padding_mask"]
            initial_token_ids = [
                list(self._initial_token_ids)
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
                dec, cache = self.model.decode(
                    token_ids,
                    kwargs["lengths"],
                    kwargs["memory"],
                    kwargs["memory_padding_mask"],
                    kwargs.get("kv_cache", None),
                    self._use_cache
                )
            else:
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
                tuple(c[mask] for c in cache)
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
                device=self.device,
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
                device=self.device,
                kwargs_select_fn=_kwargs_select_fn,
                kwargs_update_fn=_kwargs_update_fn,
                **inference_kwargs
            )

    def _process_results(
        self,
        items: List[data.InferenceItem],
        outputs: List[Any],
    ) -> data.InferenceData:
        num_pfx = self.output_tokenizer.num_prefix_tokens()
        num_sfx = self.output_tokenizer.num_suffix_tokens()
        if not self._is_encoder_decoder:
            assert num_sfx == 0, \
                "expected 0 suffix tokens for decoder-only models at inference"
            num_sfx = 1
            num_pfx = 0
        merged = "".join(
            self.output_tokenizer.de_tokenize(
                output[num_pfx:len(output)-num_sfx],
                False
            )
            for output in outputs
        )
        processed = postprocess_output(
            merged,
            self._bracket_special_tokens,
            (
                self._var_special_tokens,
                self._ent_special_tokens,
                self._prop_special_tokens
            )
        )
        return data.InferenceData(processed, language=items[0].data.language)

    def set_inference_options(
        self,
        strategy: str = "greedy",
        beam_width: int = 5,
        sample_top_k: int = 5,
        use_cache: bool = True
    ) -> None:
        assert strategy in ["greedy", "beam", "sample"]
        self._strategy = strategy
        self._beam_width = beam_width
        self._sample_top_k = sample_top_k
        self._use_cache = use_cache

    def set_indices(
        self,
        entity_index: Optional[Union[str, prefix.Vec]] = None,
        property_index: Optional[Union[str, prefix.Vec]] = None,
        example_index: Optional[Union[str, vector.Index]] = None,
    ) -> None:
        def _initial_mask(
            index: prefix.Vec,
            continuations: List[bytes]
        ) -> List[bool]:
            index.set_continuations(continuations, max_depth=0)
            cont_mask, _ = index.continuation_mask(b"")
            stripped_continuations = [c.lstrip() for c in continuations]
            index.set_continuations(stripped_continuations, max_depth=0)
            stripped_cont_mask, _ = index.continuation_mask(b"")
            return [
                len(cont) > 0 and (a or b)
                for cont, a, b, in zip(
                    stripped_continuations,
                    cont_mask,
                    stripped_cont_mask
                )
            ]

        if entity_index is not None:
            if isinstance(entity_index, str):
                entity_index = prefix.Vec.load(entity_index)
            self._entity_index = entity_index
            self._entity_index.compute_memo(max_depth=3)  # type: ignore
            self._initial_ent_mask = _initial_mask(
                self._entity_index,
                self._continuations
            )
            self._entity_index.set_continuations(
                self._continuations,
                max_depth=1
            )

        if property_index is not None:
            if isinstance(property_index, str):
                property_index = prefix.Vec.load(property_index)
            self._property_index = property_index
            self._property_index.compute_memo(max_depth=3)  # type: ignore
            self._initial_prop_mask = _initial_mask(
                self._property_index,
                self._continuations
            )
            self._property_index.set_continuations(
                self._continuations,
                max_depth=1
            )

        if example_index is not None:
            if isinstance(example_index, str):
                example_index = vector.Index.load(example_index)
            self._example_index = example_index

    @ property
    def has_kg_indices(self) -> bool:
        return self._entity_index is not None \
            and self._property_index is not None

    def get_kg_indices(self) -> Optional[Tuple[prefix.Vec, prefix.Vec]]:
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
        kg: Optional[str] = None
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
                    self.prepare_questions([s], n_examples, kg=kg)[0],
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
            output = next(iter(outputs)).text
            return output if raw else self.prepare_sparql_query(output, kg)
        else:
            return [
                output.text if raw
                else self.prepare_sparql_query(output.text, kg)
                for output in outputs
            ]

    def prepare_sparql_query(
        self,
        output: str,
        kg: Optional[str] = None
    ) -> str:
        if not self.has_kg_indices:
            return output
        return prepare_sparql_query(
            output,
            self._entity_index,
            self._property_index,
            var_special_tokens=self._var_special_tokens,
            entity_special_tokens=self._ent_special_tokens,
            property_special_tokens=self._prop_special_tokens,
            bracket_special_tokens=self._bracket_special_tokens,
            kg=kg
        )

    def prepare_questions(
        self,
        questions: List[str],
        n_examples: int = 3,
        batch_size: int = 16,
        kg: Optional[str] = None
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
                kg,
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
        kg: Optional[str] = None
    ) -> Union[Iterator[str], Iterator[data.InferenceData]]:
        loader = self._get_loader(
            (
                data.InferenceData(
                    s if raw else
                    self.prepare_questions([s], n_examples, kg=kg)[0],
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
                self.prepare_sparql_query(data.text, kg)
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
        kg: Optional[str] = None
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
                if not raw:
                    output.text = self.prepare_sparql_query(
                        output.text,
                        kg
                    )
                output_file.write(f"{output.to_str(output_file_format)}\n")

            if output_file_is_str:
                output_file.close()

        else:
            return (
                output.text if raw else
                self.prepare_sparql_query(output.text, kg)
                for output in outputs
            )
