import os
import argparse

import torch

from text_correction_utils import io
from deep_sparql.model import PRETRAINED_ENCODERS


from deep_sparql.vector import Index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    input = parser.add_mutually_exclusive_group(required=True)
    input.add_argument(
        "--prefix-index",
        type=str,
    )
    input.add_argument(
        "--data",
        type=str,
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
        choices=PRETRAINED_ENCODERS,
        required=True,
    )
    parser.add_argument(
        "-t",
        "--tokenizer",
        type=str,
        required=True
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
        default=16,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    parser.add_argument(
        "--interactive",
        action="store_true"
    )
    return parser.parse_args()


@torch.inference_mode()
def build(args: argparse.Namespace):
    if not os.path.exists(args.output) or args.overwrite:
        if args.prefix_index is not None:
            raise NotImplementedError
        else:
            iter = [
                tuple(line.split("\t"))
                for line in io.load_text_file(args.data)
            ]

        Index.build_from_iter(
            iter,
            args.model,
            args.output,
            args.batch_size,
            args.n_trees,
        )

    if not args.interactive:
        return

    vector_index = Index.load(args.output)

    while True:
        query = input("Query: ")
        neighbors = vector_index.query([query], 10)
        for i, (neighbor, dist) in enumerate(neighbors[0]):
            print(f"{i + 1}. [{dist:.4f}] {neighbor}")
        print("-" * 80)


if __name__ == "__main__":
    build(parse_args())
