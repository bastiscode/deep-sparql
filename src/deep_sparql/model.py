import copy
from typing import Dict, Any, Optional, Tuple

import torch
from torch import nn

from text_correction_utils import tokenization


from transformers import (
    MT5ForConditionalGeneration,
    T5ForConditionalGeneration,
    LlamaForCausalLM
)
from transformers.modeling_outputs import Seq2SeqLMOutput


class Model(nn.Module):
    def forward(
        self,
        token_ids: torch.Tensor,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError


class PretrainedEncoderDecoder(Model):
    def __init__(
        self,
        name: str,
    ):
        super().__init__()
        assert name in {
            "t5-small",
            "t5-base",
            "t5-large",
            "t5-3b",
            "t5-11b",
            "mt5-small",
            "mt5-base",
            "mt5-large",
            "mt5-xl",
            "mt5-xxl",
            "t5-v1_1-small",
            "t5-v1_1-base",
            "t5-v1_1-large",
            "t5-v1_1-xl",
            "t5-v1_1-xxl",
            "flan-t5-small",
            "flan-t5-base",
            "flan-t5-large",
            "flan-t5-xl",
            "flan-t5-xxl",
        }
        if name.startswith("mt5"):
            self.model = MT5ForConditionalGeneration.from_pretrained(
                f"google/{name}"
            )
        elif name.startswith("t5"):
            self.model = T5ForConditionalGeneration.from_pretrained(name)
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(
                f"google/{name}"
            )

    def forward(
        self,
        token_ids: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        target_token_ids: Optional[torch.Tensor] = None,
        **_: Any
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert target_token_ids is not None
        assert padding_mask is not None
        output: Seq2SeqLMOutput = self.model(
            input_ids=token_ids,
            attention_mask=torch.logical_not(padding_mask).to(torch.float),
            decoder_input_ids=target_token_ids,
        )
        return output.logits, {}


class PretrainedDecoder(Model):
    def __init__(
        self,
        name: str,
        **kwargs: Any
    ):
        super().__init__()
        assert name in {
            "llama",
        }

        assert "llama_path" in kwargs
        self.model = LlamaForCausalLM.from_pretrained(kwargs["llama_path"])

    def forward(
        self,
        token_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **_: Any
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert labels is not None
        output = self.model(
            input_ids=token_ids,
            labels=labels
        )
        return output.last_hidden_state, {"loss": output.loss}


def model_from_config(
    cfg: Dict[str, Any],
    input_tokenizer: tokenization.Tokenizer,
    output_tokenizer: Optional[tokenization.Tokenizer],
) -> Model:
    cfg = copy.deepcopy(cfg)
    model_type = cfg.pop("type")

    if model_type == "pretrained_encoder_decoder":
        return PretrainedEncoderDecoder(**cfg)
    elif model_type == "pretrained_decoder":
        return PretrainedDecoder(**cfg)
    else:
        raise ValueError(f"unknown model type {model_type}")
