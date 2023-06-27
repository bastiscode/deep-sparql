import argparse
import os
from typing import Set, Optional, Tuple
from multiprocessing import Pool

from tqdm import tqdm

from text_correction_utils.io import load_text_file


from deep_sparql.utils import (
    query_qlever
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--prediction", type=str, required=True)
    parser.add_argument("--save-invalid", type=str, default=None)
    parser.add_argument("--save-incorrect", type=str, default=None)
    parser.add_argument("-n", "--num-processes", type=int, default=4)
    return parser.parse_args()


def get_entities(q: str) -> Optional[Set[str]]:
    try:
        result = query_qlever(q)
        if len(result) == 0:
            return set()
        vars = list(result[0].keys())
        assert len(vars) == 1, "expected only one variable"
        return set(r[vars[0]]["value"] for r in result)
    except Exception:
        return None


def calc_f1(pred_and_target: Tuple[str, str]) -> Optional[float]:
    pred, target = pred_and_target
    if pred == target:
        return 1.0
    pred_set = get_entities(pred)
    target_set = get_entities(target)
    assert target_set is not None, "target query must be valid"
    if pred_set is None:
        return None
    if len(pred_set) == 0 and len(target_set) == 0:
        return 1.0
    tp = len(pred_set.intersection(target_set))
    fp = len(pred_set.difference(target_set))
    fn = len(target_set.difference(pred_set))
    # calculate precision, recall and f1
    if tp > 0:
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1 = 2 * p * r / (p + r)
    else:
        f1 = 0.0
    return f1


def delete_file_or_create_dir(path: str):
    if os.path.exists(path):
        os.remove(path)
    else:
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)


def evaluate(args: argparse.Namespace):
    inputs = load_text_file(args.input)
    targets = load_text_file(args.target)
    predictions = load_text_file(args.prediction)
    assert len(inputs) == len(targets) == len(predictions), \
        "expected the same number of inputs, targets and predictions"

    if args.save_invalid:
        delete_file_or_create_dir(args.save_invalid)

    if args.save_incorrect:
        delete_file_or_create_dir(args.save_incorrect)

    f1s = []
    invalid = 0
    with Pool(args.num_processes) as pool:
        for i, f1 in tqdm(
            enumerate(pool.imap(
                calc_f1,
                zip(predictions, targets),
                chunksize=16
            )),
            desc="evaluating simple questions",
            total=len(targets),
            leave=False
        ):
            if args.save_invalid:
                with open(args.save_invalid, "a", encoding="utf8") as f:
                    f.write(f"{inputs[i]}\n{predictions[i]}\n{targets[i]}\n\n")
            if args.save_incorrect and f1 is not None and f1 < 1.0:
                with open(args.save_incorrect, "a", encoding="utf8") as f:
                    f.write(f"{inputs[i]}\n{predictions[i]}\n{targets[i]}\n\n")

            if f1 is None:
                invalid += 1
                f1s.append(0.0)
            else:
                f1s.append(f1)
    print(
        f"Query-averaged F1: {100 * sum(f1s) / len(f1s):.2f} "
        f"({invalid:,} invalid, {100 * invalid / len(f1s):.2f}%)"
    )


if __name__ == "__main__":
    evaluate(parse_args())
