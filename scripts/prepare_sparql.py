import argparse
import os

from deep_sparql import SPARQLGenerator

from text_correction_utils import prefix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--entity-index", type=str, required=True)
    parser.add_argument("--property-index", type=str, required=True)
    return parser.parse_args()


def prepare(args: argparse.Namespace):
    entity_index = prefix.Vec.load(args.entity_index)
    property_index = prefix.Vec.load(args.property_index)

    cor = SPARQLGenerator.from_experiment(
        args.experiment,
        device="cpu"
    )
    cor.set_indices(entity_index, property_index)

    dir = os.path.dirname(args.output)
    if dir:
        os.makedirs(dir, exist_ok=True)

    with open(args.input, "r", encoding="utf-8") as inf, \
            open(args.output, "w", encoding="utf-8") as of:
        for line in inf:
            line = line.strip()
            out = cor.prepare_sparql_query(line)
            of.write(out + "\n")


if __name__ == "__main__":
    prepare(parse_args())
