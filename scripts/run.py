import argparse
import time


from text_correction_utils import prefix


from deep_sparql.utils import (
    prepare_sparql_query,
    query_qlever,
    format_qlever_result
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", type=str, required=True)
    parser.add_argument("-q", "--query", type=str, required=True)
    parser.add_argument("-i", "--interactive", type=str, required=True)
    parser.add_argument("--entity-index", type=str, required=True)
    parser.add_argument("--property-index", type=str, required=True)
    return parser.parse_args()


def run(args: argparse.Namespace):
    entity_index = prefix.Vec.load(args.entity_index)
    property_index = prefix.Vec.load(args.property_index)

    if args.interactive:
        while True:
            question = input("Ask something: ")
            answer_question(
                question.strip(),
                entity_index,
                property_index
            )
    else:
        answer_question(
            args.query.strip(),
            entity_index,
            property_index
        )


def answer_question(
    question: str,
    entity_index: prefix.Vec,
    property_index: prefix.Vec
):
    print(f"Question:\n{question}\n")
    start = time.perf_counter()
    output = "test"
    end = time.perf_counter()
    print(f"Output ({1000 * (end - start):.2f}ms):\n{output}\n")
    try:
        start = time.perf_counter()
        sparql = prepare_sparql_query(
            output,
            entity_index,
            property_index,
            with_labels=True,
            lang="en"
        )
        end = time.perf_counter()
        print(f"SPARQL ({1000 * (end - start):.2f}ms):\n{sparql}\n")
    except Exception as e:
        print(f"Error preparing SPARQL:\n{e}\n")
        return
    try:
        start = time.perf_counter()
        result = query_qlever(sparql)
        formatted = format_qlever_result(result[:10])
        end = time.perf_counter()
        print(f"Results ({1000 * (end - start):.2f}ms):\n{formatted}\n")
    except Exception as e:
        print(f"Error querying QLever:\n{e}\n")
        return


if __name__ == "__main__":
    run(parse_args())
