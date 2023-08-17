import os
import logging
from typing import Dict, Any, Tuple

import torch
from torch import nn
from peft import (
    PeftConfig,
    prepare_model_for_kbit_training,
    get_peft_model
)

from text_correction_utils.api.trainer import Trainer
from text_correction_utils import tokenization, data, api

from deep_sparql.api.generator import SPARQLGenerator
from deep_sparql.model import (
    PretrainedDecoder,
    PretrainedEncoderDecoder,
    model_from_config
)
from deep_sparql.utils import calc_f1


class SPARQLGenerationTrainer(Trainer):
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

    @classmethod
    def _prepare_peft(
        cls,
        model: nn.Module,
        peft_cfg: PeftConfig,
        use8_bit: bool = False
    ) -> nn.Module:
        if isinstance(model, PretrainedEncoderDecoder) or \
                isinstance(model, PretrainedDecoder):
            if use8_bit:
                model.model = prepare_model_for_kbit_training(
                    model.model
                )
            model.model = get_peft_model(model.model, peft_cfg)
        else:
            raise RuntimeError(
                "peft is only supported for pretrained models"
            )
        return model

    def _prepare_batch(
        self,
        batch: data.DataBatch
    ) -> Tuple[Dict[str, Any], torch.Tensor]:
        assert len(batch) > 0, "got empty batch"

        (
            token_ids_np,
            pad_mask_np,
            lengths,
            info,
            labels_np,
            label_info
        ) = batch.tensors()

        labels = torch.from_numpy(labels_np).to(
            non_blocking=True,
            dtype=torch.long,
            device=self.info.device
        )
        inputs = {
            "token_ids": torch.from_numpy(token_ids_np).to(
                non_blocking=True,
                device=self.info.device
            ),
            "padding_mask": torch.from_numpy(pad_mask_np).to(
                non_blocking=True,
                device=self.info.device
            ),
            "lengths": lengths,
            **api.to(info, self.info.device)
        }

        if self.cfg["model"]["type"] == "pretrained_encoder_decoder":
            # for encoder decoder models we need to provide additional
            # information for the targets
            inputs["target_token_ids"] = torch.from_numpy(
                label_info["token_ids"]
            ).to(
                non_blocking=True,
                device=self.info.device
            )

        elif self.cfg["model"]["type"] == "pretrained_decoder":
            # shift inputs for decoder only models
            labels = inputs["token_ids"][..., 1:]
            for i in range(len(inputs["lengths"])):
                inputs["lengths"][i] -= 1
                labels[i, inputs["lengths"][i]:] = -1
            inputs["token_ids"] = inputs["token_ids"][..., :-1]
            inputs["padding_mask"] = inputs["padding_mask"][..., :-1]

        else:
            raise RuntimeError(
                f"unknown model type: {self.cfg['model']['type']}"
            )

        return inputs, labels

    def _benchmark_and_checkpoint(self):
        assert self.info.is_main_process, \
            "benchmark should only be run on main process"
        self.model = self.model.eval()

        batch_size = self.cfg["train"]["batch_limit"]
        if self.cfg["train"]["batch_limit_type"] == "batch_size":
            batch_max_tokens = None
        else:
            batch_max_tokens = batch_size

        cfg = self.cfg["val"]["benchmark"]
        gen = SPARQLGenerator(self.model, self.cfg, self.info.device)
        gen.set_indices(
            cfg.get("entity_index", None),
            cfg.get("property_index", None),
            cfg.get("example_index", None)
        )
        gen.set_precision(
            self.cfg["train"].get("mixed_precision_dtype", "fp32")
            if self.cfg["train"].get("mixed_precision", False)
            else "fp32"
        )
        gen.set_inference_options(
            cfg.get("search", "greedy"),
            cfg.get("beam_width", 5),
            cfg.get("sample_top_k", 5),
            use_cache=False
        )
        scores = []
        p_invs = t_invs = 0
        for batch in self.val_loader:
            sparqls = []
            questions = []
            for item in batch.items:
                sparqls.append(item.target)
                questions.append(item.input)
            outputs = gen.generate(
                inputs=questions,
                batch_size=batch_size,
                batch_max_tokens=batch_max_tokens,
                raw=True
            )
            predictions = [
                gen.prepare_sparql_query(output)
                for output in outputs
            ]
            targets = [
                gen.prepare_sparql_query(sparql)
                for sparql in sparqls
            ]
            if gen.has_indices:
                for p, t in zip(predictions, targets):
                    f1, p_inv, t_inv = calc_f1(p, t)
                    p_invs += int(p_inv)
                    t_invs += int(t_inv)
                    scores.append(f1 or 0.0)
            else:
                scores.extend(
                    float(p == t)
                    for p, t in zip(predictions, targets)
                )

        self.summary_writer.add_scalar(
            f"val_benchmark_{'f1' if gen.has_indices else 'acc'}",
            sum(scores) / len(scores),
            self.total_step
        )
        self.summary_writer.add_scalar(
            "val_benchmark_prediction_invalid",
            p_invs / len(scores),
            self.total_step
        )
        self.summary_writer.add_scalar(
            "val_benchmark_target_invalid",
            t_invs / len(scores),
            self.total_step
        )
        self.model = self.model.train()


def main():
    parser = SPARQLGenerationTrainer.parser(
        "Train SPARQL Generator", "Train a model for generating SPARQL queries"
    )
    args = parser.parse_args()
    work_dir = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        ".."
    )
    if args.platform == "local":
        SPARQLGenerationTrainer.train_local(
            work_dir, args.experiment, args.config, args.profile
        )
    else:
        SPARQLGenerationTrainer.train_slurm(
            work_dir, args.experiment, args.config
        )


if __name__ == "__main__":
    main()
