import copy
import functools
from typing import Dict, Any, Optional, Tuple

import torch
from torch import nn

from text_correction_utils import tokenization
from text_correction_utils.api.trainer import ShardingPolicy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


from transformers import (
    MT5ForConditionalGeneration,
    PreTrainedModel,
    T5ForConditionalGeneration,
    LlamaForCausalLM,
    BertModel,
    RobertaModel,
    GPT2LMHeadModel
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
)
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.models.llama.modeling_llama import CausalLMOutputWithPast, LlamaDecoderLayer
from transformers.models.mt5.modeling_mt5 import MT5Block
from transformers.models.t5.modeling_t5 import T5Block


class Model(nn.Module):
    def forward(
        self,
        token_ids: torch.Tensor,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError

    @property
    def eos_token(self) -> Optional[str]:
        return None

    def get_sharding_policy(self) -> ShardingPolicy | None:
        return None


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
    ):
        super().__init__()
        assert name in PRETRAINED_ENCODERS, f"unknown model {name}"
        self.name = name
        if name.startswith("t5"):
            model = PretrainedEncoderDecoder(name).model
            self.model = model.encoder  # type: ignore
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
        use_8bit: bool = False,
        gradient_checkpointing: bool = False
    ):
        super().__init__()
        assert name in PRETRAINED_ENCODER_DECODERS, f"unknown model {name}"
        if name.startswith("mt5"):
            self.model = MT5ForConditionalGeneration.from_pretrained(
                f"google/{name}",
                load_in_8bit=use_8bit
            )
            self.layer_cls = MT5Block
        elif name.startswith("t5") and not name.startswith("t5-v1_1"):
            self.model = T5ForConditionalGeneration.from_pretrained(
                name,
                load_in_8bit=use_8bit
            )
            self.layer_cls = T5Block
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(
                f"google/{name}",
                load_in_8bit=use_8bit
            )
            self.layer_cls = T5Block

        assert isinstance(self.model, PreTrainedModel)
        if gradient_checkpointing:
            self.model.config.use_cache = False
            self.model.gradient_checkpointing_enable()

    @property
    def eos_token(self) -> str | None:
        return "</s>"

    def get_sharding_policy(self) -> ShardingPolicy | None:
        return functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                self.layer_cls
            }  # type: ignore
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
        output = self.model.encoder(  # type: ignore
            input_ids=token_ids,
            attention_mask=torch.logical_not(padding_mask).to(torch.float),
        )
        return output.last_hidden_state

    def decode(
        self,
        token_ids: torch.Tensor,
        lengths: torch.Tensor,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor,
        kv_cache: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: bool = True
    ) -> Tuple[torch.Tensor, Tuple[Tuple[torch.Tensor]]]:
        if use_cache and kv_cache is not None:
            b, s = token_ids.shape
            assert token_ids.ndim == 2 and s > 0
            token_ids = token_ids[torch.arange(
                b, device=token_ids.device
            ), lengths - 1, None]
        output = self.model.decoder(  # type: ignore
            input_ids=token_ids,
            encoder_hidden_states=memory,
            encoder_attention_mask=torch.logical_not(
                memory_padding_mask
            ).to(torch.float),
            past_key_values=kv_cache,
            use_cache=use_cache,
        )
        assert isinstance(output, BaseModelOutputWithPastAndCrossAttentions)
        logits = self.model.lm_head(output.last_hidden_state)  # type: ignore
        return logits, output.past_key_values  # type: ignore


PRETRAINED_DECODERS = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "llama-2-7b",
    "llama-2-30b",
    "llama-2-70b",
]


class PretrainedDecoder(Model):
    def __init__(
        self,
        name: str,
        use_8bit: bool = False,
        gradient_checkpointing: bool = False
    ):
        super().__init__()
        assert name in PRETRAINED_DECODERS, f"unknown model {name}"
        self.name = name

        if name.startswith("llama"):
            self.model = LlamaForCausalLM.from_pretrained(
                f"meta-llama/{name.capitalize()}-hf",
                load_in_8bit=use_8bit
            )
            self.layer_cls = LlamaDecoderLayer
        else:
            self.model = GPT2LMHeadModel.from_pretrained(
                name,
                load_in_8bit=use_8bit
            )
            self.layer_cls = GPT2Block

        assert isinstance(self.model, PreTrainedModel)
        if gradient_checkpointing:
            self.model.config.use_cache = False
            self.model.gradient_checkpointing_enable()

    @property
    def eos_token(self) -> str | None:
        if self.name.startswith("gpt2"):
            return "<|endoftext|>"
        else:
            return "</s>"

    def get_sharding_policy(self) -> ShardingPolicy | None:
        return functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                self.layer_cls
            }  # type: ignore
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
        lengths: torch.Tensor,
        kv_cache: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: bool = True
    ) -> Tuple[torch.Tensor, Tuple[Tuple[torch.Tensor]]]:
        if use_cache and kv_cache is not None:
            b, s = token_ids.shape
            assert token_ids.ndim == 2 and s > 0
            token_ids = token_ids[torch.arange(
                b, device=token_ids.device
            ), lengths - 1, None]
        output = self.model(  # type: ignore
            input_ids=token_ids,
            past_key_values=kv_cache,
            use_cache=use_cache
        )
        assert isinstance(
            output,
            (CausalLMOutputWithPast, CausalLMOutputWithCrossAttentions)
        )
        return output.logits, output.past_key_values  # type: ignore


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
        return PretrainedEncoderDecoder(**cfg)
    elif model_type == "pretrained_decoder":
        return PretrainedDecoder(**cfg)
    else:
        raise ValueError(f"unknown model type {model_type}")
