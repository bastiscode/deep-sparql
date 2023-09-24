import copy
import tempfile
import functools
from typing import Dict, Any, Optional, Tuple, List
from text_correction_utils.api.utils import Device

import torch
from torch import nn
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

from text_correction_utils.api.trainer import ShardingPolicy
from text_correction_utils.api import utils
from torch.utils.hooks import RemovableHandle


from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    MT5ForConditionalGeneration,
    PreTrainedModel,
    T5ForConditionalGeneration,
    LlamaForCausalLM,
    BertModel,
    RobertaModel,
    GPT2LMHeadModel
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
)
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.models.llama.modeling_llama import (
    CausalLMOutputWithPast,
    LlamaDecoderLayer
)
from transformers.models.mt5.modeling_mt5 import MT5Block
from transformers.models.t5.modeling_t5 import T5Block


def _register_hook(
    hooks: list[RemovableHandle],
    m: nn.Module,
    device: torch.device,
):
    m = m.to(device)

    def _pre_hook(
        m: nn.Module,
        args: tuple,
        kwargs: dict
    ) -> tuple[tuple, dict]:
        m = m.to(device)
        return utils.to(args, device), utils.to(kwargs, device)

    hook = m.register_forward_pre_hook(_pre_hook, with_kwargs=True)
    hooks.append(hook)


class Model(nn.Module):
    model: nn.Module
    hooks: list[RemovableHandle]

    def __init__(self):
        super().__init__()
        self.hooks = []

    def forward(
        self,
        token_ids: torch.Tensor,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError

    def get_sharding_policy(self) -> ShardingPolicy | None:
        return None

    def quantize(
        self,
        scheme: str,
        output_dir: str,
        **kwargs: Any
    ) -> None:
        raise NotImplementedError("quantization not supported")

    def distribute(
        self,
        devices: list[torch.device]
    ) -> "Model":
        assert len(devices) == 1, "only single device is supported"
        return self.to(devices[0])


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
        name: str | PreTrainedModel,
    ):
        super().__init__()
        self.max_length = 512
        if isinstance(name, PreTrainedModel):
            self.model = name
            return

        assert name in PRETRAINED_ENCODERS, f"unknown model {name}"
        self.name = name
        if name.startswith("t5"):
            model = PretrainedEncoderDecoder(name).model
            self.model = model.encoder  # type: ignore
        elif name.startswith("bert"):
            self.model = BertModel.from_pretrained(name)
        else:
            self.model = RobertaModel.from_pretrained(name)

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
        name: str | PreTrainedModel,
        gradient_checkpointing: bool = False
    ):
        super().__init__()
        if isinstance(name, PreTrainedModel):
            self.model = name
        else:
            assert isinstance(name, str)
            assert name in PRETRAINED_ENCODER_DECODERS, f"unknown model {name}"
            if name.startswith("mt5"):
                self.model = MT5ForConditionalGeneration.from_pretrained(
                    f"google/{name}",
                )  # type: ignore
            elif name.startswith("t5") and not name.startswith("t5-v1_1"):
                self.model = T5ForConditionalGeneration.from_pretrained(
                    name,
                )  # type: ignore
            else:
                self.model = T5ForConditionalGeneration.from_pretrained(
                    f"google/{name}",
                )  # type: ignore

        if isinstance(self.model, MT5ForConditionalGeneration):
            self.layer_cls = MT5Block
        else:
            self.layer_cls = T5Block

        assert isinstance(self.model, PreTrainedModel)
        if gradient_checkpointing:
            self.model.config.use_cache = False
            self.model.gradient_checkpointing_enable()

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

    def distribute(self, devices: list[torch.device]) -> "Model":
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        assert len(devices) > 0
        if len(devices) == 1:
            return self.to(devices[0])

        # distribute the layers
        layers = [
            m
            for m in self.model.modules()
            if isinstance(m, self.layer_cls)
        ]
        assert len(layers) > 0 and len(devices) <= len(layers), \
            f"{len(devices)} devices for {len(layers)} layers not supported"

        # distribute evenly among devices
        layers_per_device = [len(layers) // len(devices)] * len(devices)
        for i in range(len(layers) % len(devices)):
            layers_per_device[i] += 1

        last_encoder_idx = 0
        device_idx = 0
        for i, m in enumerate(layers):
            _register_hook(self.hooks, m, devices[device_idx])
            if i + 1 == len(layers) // 2:
                last_encoder_idx = device_idx
            if i + 1 == sum(layers_per_device[:device_idx + 1]):
                device_idx += 1

        assert device_idx == len(devices)
        _register_hook(
            self.hooks,
            self.model.shared,
            devices[0]
        )
        _register_hook(
            self.hooks,
            self.model.encoder.final_layer_norm,
            devices[last_encoder_idx],
        )
        _register_hook(
            self.hooks,
            self.model.decoder.final_layer_norm,
            devices[-1],
        )
        _register_hook(
            self.hooks,
            self.model.lm_head,
            devices[-1],
        )
        return self


QUANTIZATION_SCHEMES = [
    "w8a16",
    "w4a16"
]
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
        name: str | PreTrainedModel,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        if isinstance(name, PreTrainedModel):
            assert isinstance(name, PreTrainedModel)
            self.model = name
        else:
            assert name in PRETRAINED_DECODERS, f"unknown model {name}"
            if name.startswith("llama"):
                self.model = LlamaForCausalLM.from_pretrained(
                    f"meta-llama/{name.capitalize()}-hf"
                )  # type: ignore
            else:
                self.model = GPT2LMHeadModel.from_pretrained(
                    name
                )  # type: ignore

        if isinstance(self.model, LlamaForCausalLM):
            self.layer_cls = LlamaDecoderLayer
        else:
            self.layer_cls = GPT2Block

        assert isinstance(self.model, PreTrainedModel)
        if gradient_checkpointing:
            self.model.config.use_cache = False
            self.model.gradient_checkpointing_enable()

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
            (BaseModelOutputWithPast, CausalLMOutputWithPast,
             CausalLMOutputWithCrossAttentions)
        ), f"unexpected output type {type(output)}"
        return output.logits, output.past_key_values  # type: ignore

    def distribute(
        self,
        devices: list[torch.device]
    ) -> "Model":
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        assert len(devices) > 0
        if len(devices) == 1:
            return self.to(devices[0])

        # distribute the layers
        layers = [
            m
            for m in self.model.modules()
            if isinstance(m, self.layer_cls)
        ]
        assert len(layers) > 0 and len(devices) <= len(layers), \
            f"{len(devices)} devices for {len(layers)} layers not supported"

        # distribute evenly among devices
        layers_per_device = [len(layers) // len(devices)] * len(devices)
        for i in range(len(layers) % len(devices)):
            layers_per_device[i] += 1

        device_idx = 0
        for i, m in enumerate(layers):
            _register_hook(self.hooks, m, devices[device_idx])
            if i + 1 == sum(layers_per_device[:device_idx + 1]):
                device_idx += 1

        assert device_idx == len(devices)
        # add additional hooks for modules outside the regular
        # transformer layers
        if isinstance(self.model, LlamaForCausalLM):
            _register_hook(
                self.hooks,
                self.model.model.embed_tokens,
                devices[0]
            )
            _register_hook(
                self.hooks,
                self.model.model.norm,
                devices[-1]
            )
            _register_hook(
                self.hooks,
                self.model.lm_head,
                devices[-1]
            )
        else:
            assert isinstance(self.model, GPT2LMHeadModel)
            _register_hook(
                self.hooks,
                self.model.transformer.wte,
                devices[0]
            )
            _register_hook(
                self.hooks,
                self.model.transformer.wpe,
                devices[0]
            )
            _register_hook(
                self.hooks,
                self.model.transformer.ln_f,
                devices[-1]
            )
            _register_hook(
                self.hooks,
                self.model.lm_head,
                devices[-1]
            )
        return self

    def quantize(
        self,
        scheme: str,
        output_dir: str,
        examples: Optional[
            List[Dict[str, List[int] | torch.LongTensor]]
        ] = None,
        batch_size: int = 16,
        use_triton: bool = False,
        cache_on_gpu: bool = True,
        **kwargs: Any
    ) -> None:
        assert scheme in QUANTIZATION_SCHEMES, \
            f"unknown quantization scheme {scheme}, must be one of " \
            f"{QUANTIZATION_SCHEMES}"
        assert examples is not None

        if scheme == "w8a16":
            bits = 8
        elif scheme == "w4a16":
            bits = 4
        else:
            raise ValueError(f"unknown quantization scheme {scheme}")

        config = BaseQuantizeConfig(
            bits=bits,
            group_size=128,
            desc_act=False
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            self.model: PreTrainedModel
            self.model.save_pretrained(tmpdir)
            quant_model = AutoGPTQForCausalLM.from_pretrained(
                tmpdir,
                config
            )
        quant_model.quantize(
            examples,
            batch_size,
            use_triton=use_triton,
            cache_examples_on_gpu=cache_on_gpu,
        )
        quant_model.save_quantized(output_dir)


def model_from_config(
    cfg: Dict[str, Any],
    device: Device
) -> Model:
    cfg = copy.deepcopy(cfg)
    model_type = cfg.pop("type")

    if model_type == "pretrained_encoder_decoder":
        return PretrainedEncoderDecoder(**cfg)
    elif model_type == "custom_pretrained_encoder_decoder":
        model = AutoModelForSeq2SeqLM.from_pretrained(cfg["path"])
        return PretrainedEncoderDecoder(model)
    elif model_type == "pretrained_decoder":
        return PretrainedDecoder(**cfg)
    elif model_type == "custom_pretrained_decoder":
        model = AutoModelForCausalLM.from_pretrained(cfg["path"])
        return PretrainedDecoder(model)
    elif model_type == "quantized_decoder":
        assert device != "cpu", "quantized model must be on GPU"
        quant = AutoGPTQForCausalLM.from_quantized(
            cfg["path"]
        )
        assert isinstance(quant.model, PreTrainedModel)
        return PretrainedDecoder(quant.model)
    else:
        raise ValueError(f"unknown model type {model_type}")
