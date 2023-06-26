import os
import argparse

from deep_sparql.model import PretrainedEncoder

from tqdm import tqdm
import torch
import annoy
from text_correction_utils import prefix, tokenization, data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prefix-index",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=["t5-small", "t5-base", "t5-large"],
        required=True,
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "-t",
        "--n-trees",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    return parser.parse_args()


@torch.inference_mode()
def build(args: argparse.Namespace):
    tokenizer_config = {
        "tokenize": {
            "type": "huggingface",
            "name": args.model,
        },
        "special": {
            "pad": "<pad>",
            "tokens": ["<pad>"],
        }
    }
    tokenizer = tokenization.Tokenizer.from_config(tokenizer_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = PretrainedEncoder(
        args.model,
        tokenizer.vocab_size()
    ).to(device)
    index = prefix.Vec.load(args.prefix_index)

    dim = encoder.encode(
        torch.tensor([[0]], device=device, dtype=torch.long),
        torch.tensor([[False]], device=device, dtype=torch.bool),
    ).shape[-1]

    vector_index = annoy.AnnoyIndex(dim, "angular")
    if not os.path.exists(args.output) or args.overwrite:
        loader = data.InferenceLoader.from_iterator(
            (data.InferenceData(name) for name, _
             in index.get_continuations("".encode("utf8"))),
            tokenizer_config=tokenizer_config,
            window_config={"type": "full"},
            batch_limit=args.batch_size,
            prefetch_factor=16
        )

        idx = 0
        for batch in tqdm(
            loader,
            total=len(index) // args.batch_size,
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
            encoded = encoder.encode(**inputs).cpu().numpy()
            for vector, length in zip(encoded, lengths):
                vector_index.add_item(idx, vector[:length].mean(axis=0))
                idx += 1

        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        vector_index.build(args.n_trees, n_jobs=-1)
        vector_index.save(args.output)

    vector_index.load(args.output)

    while True:
        query = input("Query: ")
        loader = data.InferenceLoader.from_iterator(
            iter([data.InferenceData(query.strip())]),
            tokenizer_config=tokenizer_config,
            window_config={"type": "full"}
        )
        for batch in loader:
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
            encoded = encoder.encode(**inputs).cpu().numpy()
            for vector, length in zip(encoded, lengths):
                result, dists = vector_index.get_nns_by_vector(
                    vector[:length].mean(axis=0),
                    10,
                    include_distances=True
                )
                for i, (r, dist) in enumerate(zip(result, dists)):
                    print(f"{i + 1}. {index.at(r)} ({dist:.4f})")
                print("-" * 80)


if __name__ == "__main__":
    build(parse_args())
