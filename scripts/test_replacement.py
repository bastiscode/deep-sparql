import argparse


from text_correction_utils import prefix


from deep_sparql.utils import (
    prepare_sparql_query,
    query_qlever,
    format_qlever_result
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--entity-index", type=str, required=True)
    parser.add_argument("--property-index", type=str, required=True)
    return parser.parse_args()


def replace(args: argparse.Namespace):
    entity_index = prefix.Vec.load(args.entity_index)
    property_index = prefix.Vec.load(args.property_index)

    with open(args.input, "r", encoding="utf8") as inf:
        for line in inf:
            print("Question:")
            print("Model output:")
            print(line.strip())
            print()
            sparql = prepare_sparql_query(
                line.strip(),
                entity_index,
                property_index,
                with_labels=True,
                lang="en"
            )
            print("Processed SPARQL:")
            print(sparql)
            print()
            result = query_qlever(sparql)
            print("First 10 results:")
            print(format_qlever_result(result[:10]))
            print("-" * 80)
            break


if __name__ == "__main__":
    replace(parse_args())
