import os
from typing import Dict, Any, Tuple

import torch
from torch import nn

from text_correction_utils.api.trainer import Trainer
from text_correction_utils import tokenization, data, api

from deep_sparql.model import model_from_config


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
            inputs["token_ids"] = inputs["token_ids"][..., :-1]
            inputs["padding_mask"] = inputs["padding_mask"][..., :-1]
            inputs["lengths"] = [length - 1 for length in inputs["lengths"]]

        else:
            raise RuntimeError(
                f"unknown model type: {self.cfg['model']['type']}"
            )

        return inputs, labels


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
