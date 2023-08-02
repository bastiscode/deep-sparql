import copy
from typing import Dict, Any, Optional, Tuple

import torch
from torch import nn

from text_correction_utils import tokenization


from transformers import (
    MT5ForConditionalGeneration,
    PreTrainedModel,
    T5ForConditionalGeneration,
    LlamaForCausalLM,
    BertModel,
    RobertaModel
)
from transformers.modeling_outputs import Seq2SeqLMOutput


class Model(nn.Module):
    def forward(
        self,
        token_ids: torch.Tensor,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError


PRETRAINED_ENCODERS = [
    "t5-small",
    "t5-base",
    "t5-large",
    "bert-base-uncased",
    "bert-large-uncased",
    "roberta-base",
    "roberta-large"
]


class PretrainedEncoder(Model):
    def __init__(
        self,
        name: str,
        vocab_size: int,
    ):
        super().__init__()
        assert name in PRETRAINED_ENCODERS, "unknown model"
        self.name = name
        if name.startswith("t5"):
            model = PretrainedEncoderDecoder(
                name,
                vocab_size=vocab_size
            ).model
            assert isinstance(model, PreTrainedModel)
            self.model = model.encoder
        elif name.startswith("bert"):
            self.model = BertModel.from_pretrained(name)
        else:
            self.model = RobertaModel.from_pretrained(name)
        self.max_length = 512

    def forward(
        self,
        token_ids: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        **_: Any
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert padding_mask is not None
        return self.encode(token_ids, padding_mask), {}

    def encode(
        self,
        token_ids: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        output = self.model(
            input_ids=token_ids[:, :self.max_length],
            attention_mask=torch.logical_not(
                padding_mask[:, :self.max_length]
            ).to(torch.float),
        )  # type: ignore
        return output.last_hidden_state


PRETRAINED_ENCODER_DECODERS = [
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
]


class PretrainedEncoderDecoder(Model):
    def __init__(
        self,
        name: str,
        vocab_size: int,
    ):
        super().__init__()
        assert name in PRETRAINED_ENCODER_DECODERS, "unknown model"
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

        assert isinstance(self.model, PreTrainedModel)
        num_emb, _ = self.model.get_input_embeddings().weight.shape
        if vocab_size > num_emb:
            raise NotImplementedError(
                f"vocab size {vocab_size:,} is larger than number of "
                f"embeddings {num_emb:,}, resizing of embedding not "
                "yet implemented"
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
        )  # type: ignore
        return output.logits, {}

    def encode(
        self,
        token_ids: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        assert isinstance(self.model, PreTrainedModel)
        output = self.model.encoder(
            input_ids=token_ids,
            attention_mask=torch.logical_not(padding_mask).to(torch.float),
        )  # type: ignore
        return output.last_hidden_state

    def decode(
        self,
        token_ids: torch.Tensor,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        assert isinstance(self.model, PreTrainedModel)
        output = self.model(
            decoder_input_ids=token_ids,
            encoder_outputs=(memory, ),
            attention_mask=torch.logical_not(
                memory_padding_mask
            ).to(torch.float),
        )  # type: ignore
        return output.logits


class PretrainedDecoder(Model):
    def __init__(
        self,
        name: str,
        vocab_size: int,
        **kwargs: Any
    ):
        super().__init__()
        assert name in {
            "llama-2-7b",
            "llama-2-30b",
            "llama-2-70b",
        }

        self.model = LlamaForCausalLM.from_pretrained(
            f"meta-llama/{name.capitalize()}-hf"
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        **_: Any
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        output = self.model(input_ids=token_ids)  # type: ignore
        return output.logits, {}

    def decode(
        self,
        token_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor]:
        output = self.model(input_ids=token_ids)  # type: ignore
        return output.logits


def model_from_config(
    cfg: Dict[str, Any],
    input_tokenizer: tokenization.Tokenizer,
    output_tokenizer: Optional[tokenization.Tokenizer],
) -> Model:
    cfg = copy.deepcopy(cfg)
    model_type = cfg.pop("type")

    if model_type == "pretrained_encoder_decoder":
        assert output_tokenizer is not None
        assert input_tokenizer.vocab_size() == output_tokenizer.vocab_size()
        return PretrainedEncoderDecoder(
            **cfg,
            vocab_size=input_tokenizer.vocab_size(),
        )
    elif model_type == "pretrained_decoder":
        return PretrainedDecoder(
            **cfg,
            vocab_size=input_tokenizer.vocab_size(),
        )
    else:
        raise ValueError(f"unknown model type {model_type}")
