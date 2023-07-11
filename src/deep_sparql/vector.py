from typing import Iterable, Union, List, Tuple, Dict, Any
import os
import random

import yaml
import torch
import annoy
from tqdm import tqdm

from text_correction_utils import tokenization, data, io

from deep_sparql.model import PretrainedEncoder


class Index:
    def __init__(
        self,
        index: annoy.AnnoyIndex,
        data: List[str],
        model: PretrainedEncoder,
        tokenizer_cfg: Dict[str, Any],
        device: torch.device,
    ):
        self.index = index
        self.data = data
        self.tokenizer_cfg = tokenizer_cfg
        self.model = model
        self.device = device

    @staticmethod
    def _load_model(
        model_name: str,
        device: Union[str, torch.device] = "cuda",
    ) -> Tuple[Dict[str, Any], PretrainedEncoder, torch.device]:
        assert model_name in ["t5-small", "t5-base", "t5-large"], \
            f"model name must be one of 't5-small', 't5-base', 't5-large', " \
            f"got {model_name}"
        tokenizer_cfg = {
            "tokenize": {
                "type": "huggingface",
                "name": model_name,
            },
            "special": {
                "pad": "<pad>",
                "tokens": ["<pad>"],
            }
        }
        tokenizer = tokenization.Tokenizer.from_config(tokenizer_cfg)
        device = torch.device(device)
        model = PretrainedEncoder(
            model_name,
            tokenizer.vocab_size()
        ).to(device).eval()
        return tokenizer_cfg, model, device

    @staticmethod
    def build_from_iter(
        iter: Union[List[Tuple[str, str]], Iterable[Tuple[str, str]]],
        model: str,
        dir: str,
        batch_size: int = 16,
        n_trees: int = 16,
    ):
        keys = []
        items = []
        for key, item in iter:
            keys.append(key)
            items.append(item)

        if len(keys) == 0:
            raise ValueError("no data to index")

        tokenizer_cfg, encoder, device = Index._load_model(model)

        dim = encoder.encode(
            torch.tensor([[0]], device=device, dtype=torch.long),
            torch.tensor([[False]], device=device, dtype=torch.bool),
        ).shape[-1]

        vector_index = annoy.AnnoyIndex(dim, "angular")
        loader = data.InferenceLoader.from_iterator(
            (data.InferenceData(k) for k in keys),
            tokenizer_config=tokenizer_cfg,
            window_config={"type": "full"},
            batch_limit=batch_size,
            prefetch_factor=16
        )

        total = len(keys) // batch_size  # type: ignore

        idx = 0
        for batch in tqdm(
            loader,
            desc="building vector index",
            total=total,
            leave=False
        ):
            token_ids_np, pad_mask_np, lengths, _ = batch.tensors()
            inputs = {
                "token_ids": torch.from_numpy(token_ids_np).to(
                    non_blocking=True,
                    device=device
                ),
                "padding_mask": torch.from_numpy(pad_mask_np).to(
                    non_blocking=True,
                    device=device
                ),
            }
            with torch.inference_mode():
                encoded = encoder.encode(**inputs).cpu().numpy()

            for vector, length in zip(encoded, lengths):
                vector_index.add_item(idx, vector[:length].mean(axis=0))
                idx += 1

        vector_index.build(n_trees, n_jobs=-1)

        os.makedirs(dir, exist_ok=True)
        vector_index.save(os.path.join(dir, "index.bin"))
        with open(os.path.join(dir, "config.yaml"), "w") as of:
            yml = yaml.dump({
                "model": model,
                "dim": dim,
                "metric": "angular",
            })
            of.write(yml + "\n")

        with open(os.path.join(dir, "data.txt"), "w") as of:
            for item in items:
                of.write(item + "\n")

    @staticmethod
    def load(
        dir: str,
        device: Union[str, torch.device] = "cuda",
    ) -> "Index":
        with open(os.path.join(dir, "config.yaml")) as f:
            cfg = yaml.full_load(f)
        items = io.load_text_file(os.path.join(dir, "data.txt"))

        vector_index = annoy.AnnoyIndex(cfg["dim"], cfg["metric"])
        vector_index.load(os.path.join(dir, "index.bin"))
        tokenizer_cfg, encoder, device = Index._load_model(
            cfg["model"],
            device
        )
        return Index(vector_index, items, encoder, tokenizer_cfg, device)

    @torch.inference_mode()
    def query(
        self,
        keys: List[str],
        n: int = 10,
        batch_size: int = 16,
    ) -> List[List[Tuple[str, float]]]:
        loader = data.InferenceLoader.from_iterator(
            (data.InferenceData(k) for k in keys),
            tokenizer_config=self.tokenizer_cfg,
            window_config={"type": "full"},
            batch_limit=batch_size,
            prefetch_factor=16
        )
        result = []
        for batch in loader:
            token_ids_np, pad_mask_np, lengths, _ = batch.tensors()
            inputs = {
                "token_ids": torch.from_numpy(token_ids_np).to(
                    non_blocking=True,
                    device=self.device
                ),
                "padding_mask": torch.from_numpy(pad_mask_np).to(
                    non_blocking=True,
                    device=self.device
                ),
            }
            encoded = self.model.encode(**inputs).cpu().numpy()
            for vector, length in zip(encoded, lengths):
                top_indices, top_distances = self.index.get_nns_by_vector(
                    vector[:length].mean(axis=0),
                    n,
                    include_distances=True
                )
                result.append([
                    (self.data[i], dist)
                    for i, dist in zip(top_indices, top_distances)
                ])
        return result


def sample_nearest_neighbors(
    questions: List[str],
    index: Index,
    max_neighbors: int,
    batch_size: int = 16,
    progress: bool = True,
) -> List[str]:
    nn_strs = []
    for i in tqdm(
        range(0, len(questions), batch_size),
        desc="getting nearest neighbors",
        total=len(questions) // batch_size,
        leave=False,
        disable=not progress
    ):
        neighbors = index.query(
            questions[i:i + batch_size],
            max_neighbors + 1,
        )
        for nns in neighbors:
            nns = [
                ex
                for ex, dist in nns
                if dist > 0.0
            ]
            nn_str = " ".join(
                nns[:random.randint(0, max_neighbors)]
            )
            nn_strs.append(nn_str)
    return nn_strs
