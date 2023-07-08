import argparse
import os


from text_correction_utils import io

from deep_sparql.utils import format_example


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
    )
    return parser.parse_args()


def prepare(args: argparse.Namespace):
    inputs = io.load_text_file(args.input)
    targets = io.load_text_file(args.target)
    assert len(inputs) == len(targets), \
        "expected same number of inputs and targets"
    dir = os.path.dirname(args.output)
    if dir:
        os.makedirs(dir, exist_ok=True)
    with open(args.output, "w") as of:
        for input, target in zip(inputs, targets):
            example = format_example(input, target)
            of.write(f"{input}\t{example}\n")


if __name__ == "__main__":
    prepare(parse_args())
