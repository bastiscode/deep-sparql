import argparse
from typing import Set

from tqdm import tqdm

from text_correction_utils.io import load_text_file


from deep_sparql.utils import (
    query_qlever
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--predictions", type=str, required=True)
    return parser.parse_args()


def get_entities(q: str) -> Set[str]:
    for _ in range(5):
        try:
            result = query_qlever(q)
            if len(result) == 0:
                return set()
            vars = list(result[0].keys())
            assert len(vars) == 1, "expected only one variable"
            return set(r[vars[0]] for r in result)
        except Exception as e:
            print(f"failed to query QLever: {e}")
            print("retrying...")
            continue

    raise RuntimeError("failed to query QLever in 5 attempts")


def calc_f1(a: Set[str], b: Set[str]) -> float:
    if len(a) == 0 and len(b) == 0:
        return 1.0
    return 2 * len(a.intersection(b)) / (len(a) + len(b))


def evaluate(args: argparse.Namespace):
    targets = load_text_file(args.target)
    predictions = load_text_file(args.predictions)
    assert len(targets) == len(predictions), \
        "expected the same number of targets and predictions"

    f1s = []
    for tgt, pred in tqdm(
        zip(targets, predictions),
        desc="evaluating simple questions",
        total=len(targets),
        leave=False
    ):
        # if the queries already match, then the F1 is 1.0
        if tgt == pred:
            f1s.append(1.0)
            continue

        tgt_res = get_entities(tgt)
        pred_res = get_entities(pred)
        f1 = calc_f1(tgt_res, pred_res)
        f1s.append(f1)


if __name__ == "__main__":
    evaluate(parse_args())
