from io import TextIOWrapper
import os
import sys
from typing import Any, Dict, List, Tuple, Optional, Union, Iterator

import torch
from torch import nn

from deep_sparql.model import model_from_config

from text_correction_utils import data, tokenization
from text_correction_utils.api.corrector import ModelInfo
from text_correction_utils.api import corrector
from text_correction_utils.api.utils import device_info, to
from text_correction_utils.inference import (
    eos_stop_fn,
    greedy_select_fn,
    sample_select_fn,
    search,
    beam_search
)

_BASE_URL = "https://ad-publications.informatik.uni-freiburg.de/" \
    "ACL_whitespace_correction_transformer_BHW_2023.materials"
_NAME_TO_ZIP = {
}


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
            f"running {self.name} spelling corrector "
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
        self._eos_token_id = self.output_tokenizer.special_token_to_id(
            tokenization.SpecialTokens.EOS
        )
        self._strategy = "greedy"
        self._beam_width = 5
        self._sample_top_k = 5
        assert self._eos_token_id is not None

    def _build_inference_loader_config(self) -> Dict[str, Any]:
        input_tokenizer = tokenization.Tokenizer.from_config(
            self.cfg["input_tokenizer"]
        )
        pfx = input_tokenizer.num_prefix_tokens()
        sfx = input_tokenizer.num_suffix_tokens()

        # use the training max sequence length here,
        # even though some models work with arbitrary long sequences
        # (e.g. LSTM), for better accuracy
        max_length = self.max_length - pfx - sfx
        if self.cfg["input_tokenizer"]["tokenize"]["type"] in {"byte", "bpe"}:
            window_cfg = {
                "type": "byte",
                "max_bytes": max_length,
                "context_bytes": 0
            }
        else:
            raise ValueError(
                "the input tokenizer must be of type 'byte' or 'bpe' \
                for spelling correction"
            )

        return {
            "tokenizer_config": self.cfg["input_tokenizer"],
            "window_config": window_cfg,
        }

    def _prepare_batch(self, batch: data.InferenceBatch) -> Dict[str, Any]:
        token_ids_np, pad_mask_np, lengths, info = batch.tensors()
        inputs = {
            "token_ids": torch.from_numpy(token_ids_np).to(
                non_blocking=True,
                device=self.device
            ),
            "padding_mask": torch.from_numpy(pad_mask_np).to(
                non_blocking=True,
                device=self.device
            ),
            "lengths": lengths,
            **to(info, self.device)
        }
        return inputs

    def _inference(self, inputs: Dict[str, Any]) -> Any:
        assert isinstance(self.model, EncoderDecoderWithHead)
        enc, kwargs = self.model.encode(**inputs)

        # decode fn gets in token ids and additional kwargs,
        # and return logits over next tokens
        def _decode_fn(
            token_ids: torch.Tensor,
            **kwargs: Any
        ) -> torch.Tensor:
            assert isinstance(self.model, EncoderDecoderWithHead)
            dec = self.model.decode(
                token_ids,
                kwargs.pop("memories"),
                **kwargs
            )
            return dec

        def _kwargs_select_fn(
            kwargs: Dict[str, Any],
            mask: torch.Tensor
        ) -> Dict[str, Any]:
            return {
                "memories": {
                    k: v[mask]
                    for k, v in kwargs["memories"].items()
                },
                "memory_padding_masks": {
                    k: v[mask]
                    for k, v in kwargs["memory_padding_masks"].items()
                }
            }

        max_output_length = self.cfg["model"]["decoder_embedding"]["max_length"]

        initial_token_ids = [
            self._initial_token_ids
        ] * inputs["token_ids"].shape[0]
        stop_fn = eos_stop_fn(self._eos_token_id)
        if self._strategy == "beam" and self._beam_width > 1:
            outputs = beam_search(
                decode_fn=_decode_fn,
                initial_token_ids=initial_token_ids,
                vocab_size=self.output_tokenizer.vocab_size(),
                pad_token_id=self.output_tokenizer.pad_token_id(),
                max_length=max_output_length,
                stop_fn=stop_fn,
                device=self.device,
                normalize_by_length=True,
                alpha=1.0,
                beam_width=self._beam_width,
                kwargs_select_fn=_kwargs_select_fn,
                memories=enc,
                **kwargs
            )
            return [output[0].token_ids for output in outputs]
        elif self._strategy == "sample" and self._sample_top_k > 1:
            return search(
                decode_fn=_decode_fn,
                initial_token_ids=initial_token_ids,
                pad_token_id=self.output_tokenizer.pad_token_id(),
                max_length=max_output_length,
                select_fn=sample_select_fn(self._sample_top_k),
                stop_fn=stop_fn,
                device=self.device,
                kwargs_select_fn=_kwargs_select_fn,
                memories=enc,
                **kwargs
            )
        else:
            return search(
                decode_fn=_decode_fn,
                initial_token_ids=initial_token_ids,
                pad_token_id=self.output_tokenizer.pad_token_id(),
                max_length=max_output_length,
                select_fn=greedy_select_fn(),
                stop_fn=stop_fn,
                device=self.device,
                kwargs_select_fn=_kwargs_select_fn,
                memories=enc,
                **kwargs
            )

    def _process_results(
        self,
        items: List[data.InferenceItem],
        outputs: List[Any],
    ) -> data.InferenceData:
        merged = "".join(
            self.output_tokenizer.de_tokenize(output)
            for output in outputs
        )
        return data.InferenceData(merged, language=items[0].data.language)

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

    def correct_text(
            self,
            inputs: Union[str, List[str]],
            languages: Optional[List[str]] = None,
            batch_size: int = 16,
            batch_max_tokens: Optional[int] = None,
            sort: bool = True,
            num_threads: Optional[int] = None,
            show_progress: bool = False
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

        progress_desc = f"Correcting spelling errors in " \
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

        return next(iter(outputs)).text if input_is_string else [output.text for output in outputs]

    def correct_iter(
        self,
        iter: Iterator[Tuple[str, Optional[str]]],
        batch_size: int = 16,
        batch_max_tokens: Optional[int] = None,
        sort: bool = True,
        num_threads: Optional[int] = None,
        return_raw: bool = False,
        show_progress: bool = False
    ) -> Union[Iterator[str], Iterator[data.InferenceData]]:
        loader = self._get_loader(
            (data.InferenceData(s, language=l) for s, l in iter),
            batch_size,
            batch_max_tokens,
            sort,
            num_threads,
        )

        progress_desc = "Correcting whitespaces in iterator"
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

        if return_raw:
            yield from output
        else:
            yield from (data.text for data in output)

    def correct_file(
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
        progress_desc = f"Correcting spelling in {file_name}"
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
                output_file.write(f"{output.to_str(output_file_format)}\n")

            if output_file_is_str:
                output_file.close()

        else:
            return (output.text for output in outputs)

    def set_precision(self, precision: str) -> None:
        training_precision = self.cfg["train"].get(
            "mixed_precision_dtype", "fp32")
        if precision != "fp32" and precision != training_precision:
            self.logger.warning(
                f"this model was trained with {training_precision} precision, "
                "inference with {precision} might give unexpected results"
            )
        return super().set_precision(precision)
