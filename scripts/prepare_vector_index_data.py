import argparse
import os
import re


from text_utils import io

from deep_sparql.utils import format_example


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--targets",
        type=str,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--raws",
        type=str,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--kg",
        choices=["wikidata", "dbpedia", "freebase"],
        default="wikidata",
    )
    return parser.parse_args()


def prepare(args: argparse.Namespace):
    assert len(args.inputs) == len(args.targets) == len(args.raws), \
        "expected same number of inputs, targets, and raw files"
    q_pattern = re.compile(r"\w+ question \"(.*)\"")
    if os.path.exists(args.output) and os.path.isfile(args.output):
        os.remove(args.output)
    raw_seen = set()
    for input_path, target_path, raw_path in zip(
        args.inputs,
        args.targets,
        args.raws
    ):
        inputs = io.load_text_file(input_path)
        targets = io.load_text_file(target_path)
        raws = io.load_text_file(raw_path)
        assert len(inputs) == len(targets) == len(raws), \
            "expected same number of inputs, targets, and raws"
        dir = os.path.dirname(args.output)
        if dir:
            os.makedirs(dir, exist_ok=True)
        with open(args.output, "a", encoding="utf8") as of:
            for input, target, raw in zip(inputs, targets, raws):
                if raw in raw_seen:
                    continue
                matches = list(q_pattern.finditer(input))
                assert len(matches) == 1
                match = matches[0]
                question = match.group(1)
                example = format_example(question, target)
                of.write(f"{question}\t{example}\n")
                raw_seen.add(raw)


if __name__ == "__main__":
    prepare(parse_args())
