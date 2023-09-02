import os
from typing import Dict, Any, Tuple

import torch
from torch import nn
from peft import (
    PeftConfig,
    prepare_model_for_kbit_training,
    get_peft_model
)

from text_correction_utils.api.trainer import ShardingPolicy, Trainer
from text_correction_utils import tokenization, data, api, distributed

from deep_sparql.api.generator import SPARQLGenerator
from deep_sparql.model import (
    PretrainedDecoder,
    PretrainedEncoderDecoder,
    model_from_config
)
from deep_sparql.utils import calc_f1


class SPARQLGenerationTrainer(Trainer):
    @classmethod
    def _model_from_config(
        cls,
        cfg: Dict[str, Any]
    ) -> Tuple[nn.Module, ShardingPolicy | None]:
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
        return model, model.get_sharding_policy()

    @classmethod
    def _prepare_peft(
        cls,
        model: nn.Module,
        peft_cfg: PeftConfig,
        use8_bit: bool = False
    ) -> nn.Module:
        if isinstance(model, (PretrainedEncoderDecoder, PretrainedDecoder)):
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
        batch: data.DataBatch,
        train: bool = True
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

        if self.cfg["model"]["type"] == "pretrained_encoder_decoder":
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
            # for encoder decoder models we need to provide additional
            # information for the targets
            inputs["target_token_ids"] = torch.from_numpy(
                label_info["token_ids"]
            ).to(
                non_blocking=True,
                device=self.info.device
            )

        elif self.cfg["model"]["type"] == "pretrained_decoder":
            mask_prefix = self.cfg["train"].get("mask_prefix", False)
            if (not train or mask_prefix) and "prefix_lengths" in label_info:
                # mask out the prefix in the labels with -1 to ignore it
                for i, pfx_l in enumerate(label_info["prefix_lengths"]):
                    if pfx_l <= 0:
                        continue
                    labels_np[i, :pfx_l - 1] = -1
                    print(pfx_l)
                    print(labels_np[i], labels_np[i, pfx_l - 1:])
                    print(self.input_tokenizer.de_tokenize(
                        labels_np[i, pfx_l - 1:label_info["lengths"][i]],
                        False
                    ))

            inputs = {
                "token_ids": torch.from_numpy(
                    label_info["token_ids"]
                ).to(
                    non_blocking=True,
                    device=self.info.device
                ),
                "padding_mask": torch.from_numpy(
                    label_info["padding_mask"]
                ).to(
                    non_blocking=True,
                    device=self.info.device
                ),
                "lengths": label_info["lengths"],
            }

        else:
            raise RuntimeError(
                f"unknown model type: {self.cfg['model']['type']}"
            )

        labels = torch.from_numpy(labels_np).to(
            non_blocking=True,
            dtype=torch.long,
            device=self.info.device
        )

        return inputs, labels

    def _benchmark_and_checkpoint(self):
        cfg = self.cfg["val"].get("benchmark", None)
        if cfg is None:
            return
        assert self.info.is_main_process, \
            "benchmark should only be run on main process"
        self.model = self.model.eval()
        self.logger.info(
            f"[step {self.total_step}] "
            "benchmarking model on validation data"
        )

        gen = SPARQLGenerator(
            distributed.unwrap_model(self.model),
            self.cfg,
            self.info.device
        )
        gen.set_indices(
            cfg.get("entity_index", None),
            cfg.get("property_index", None),
        )
        gen.set_precision(self.cfg["train"].get("precision", "fp32"))
        gen.set_inference_options(
            cfg.get("search", "greedy"),
            cfg.get("beam_width", 5),
            cfg.get("sample_top_k", 5),
        )
        self.logger.info(
            f"[step {self.total_step}] "
            "setup SPARQL generator"
        )
        limit = min(
            cfg.get("limit", self.val_loader.min_items),
            self.val_loader.min_items
        )
        log_n = cfg.get("log_n_samples", 8)
        kg = cfg.get("kg", "wikidata")
        batch_size = cfg.get("batch_size", None)
        scores = []
        tok = gen.output_tokenizer
        num_pfx = tok.num_prefix_tokens()
        num_sfx = tok.num_suffix_tokens()
        p_invs = t_invs = 0
        samples = []
        for batch in self.val_loader:
            if len(scores) >= limit:
                break
            sparqls = []
            questions = []
            for item in batch.items:
                if len(scores) + len(sparqls) >= limit:
                    break
                suffix = ":" * (not gen._is_encoder_decoder)
                question = item.data.input + suffix
                questions.append(question)
                sparqls.append(item.data.target)
            outputs = gen.generate(
                inputs=questions,
                batch_size=batch_size or len(questions),
                raw=True
            )
            if len(samples) < log_n:
                diff = log_n - len(samples)
                for q, o, s in zip(
                    questions[:diff],
                    outputs[:diff],
                    sparqls[:diff]
                ):
                    samples.append(
                        f"Question  : {q}\n\n"
                        f"Prediction: {o}\n\n"
                        f"Target    : {s}"
                    )
            if gen.has_kg_indices:
                predictions = [
                    gen.prepare_sparql_query(output, kg)
                    for output in outputs
                ]
                targets = []
                for sparql in sparqls:
                    token_ids = tok.tokenize(sparql).token_ids
                    target = gen.prepare_sparql_query(
                        tok.de_tokenize(
                            token_ids[num_pfx:len(token_ids) - num_sfx],
                            False
                        ).strip(),
                        kg
                    )
                    targets.append(target)
                for p, t in zip(predictions, targets):
                    f1, p_inv, t_inv = calc_f1(p, t)
                    p_invs += int(p_inv)
                    t_invs += int(t_inv)
                    scores.append(f1 or 0.0)
            else:
                for p, t in zip(outputs, sparqls):
                    token_ids = tok.tokenize(t).token_ids
                    score = p.strip() == tok.de_tokenize(
                        token_ids[num_pfx:len(token_ids) - num_sfx],
                        False
                    ).strip()
                    scores.append(float(score))
            self.logger.info(
                f"[step {self.total_step}] "
                f"benchmark_progress: {len(scores) / (limit or 1):.2%}, "
                f"{len(scores):,} / {limit:,} items"
            )

        assert self.summary_writer is not None
        sample_str = f"\n\n{'-' * 80}\n\n".join(samples)
        self.summary_writer.add_text(
            "val_benchmark_samples",
            sample_str,
            self.total_step
        )
        self.logger.info(
            f"[step {self.total_step}] "
            f"val_benchmark_samples:\n\n{sample_str}"
        )
        postfix = "f1" if gen.has_kg_indices else "acc"
        num_scores = len(scores) or 1
        score = sum(scores) / num_scores
        self.summary_writer.add_scalar(
            f"val_benchmark_{postfix}",
            score,
            self.total_step
        )
        self.logger.info(
            f"[step {self.total_step}] "
            f"val_benchmark_{postfix}: {score:.2%}"
        )
        p_inv = p_invs / num_scores
        self.summary_writer.add_scalar(
            "val_benchmark_invalid_predictions",
            p_inv,
            self.total_step
        )
        self.logger.info(
            f"[step {self.total_step}] "
            f"val_benchmark_invalid_predictions: {p_inv:.2%}"
        )
        t_inv = t_invs / num_scores
        self.summary_writer.add_scalar(
            "val_benchmark_invalid_targets",
            t_inv,
            self.total_step
        )
        self.logger.info(
            f"[step {self.total_step}] "
            f"val_benchmark_invalid_targets: {t_inv:.2%}"
        )
        if self.best_benchmark is None or score > self.best_benchmark:
            self.best_benchmark = score
            self._save_checkpoint(
                os.path.join(
                    self.directories["checkpoints"],
                    "benchmark_best.pt"
                ),
                0.0,
                full=False,
                benchmark_score=self.best_benchmark
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
