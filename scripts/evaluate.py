import argparse
import os
from typing import Optional, Tuple
from multiprocessing import Pool

from tqdm import tqdm

from text_correction_utils.io import load_text_file


from deep_sparql.utils import (
    calc_f1
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


def calc_f1_map(
    pred_and_target: Tuple[str, str]
) -> Tuple[Optional[float], bool, bool]:
    return calc_f1(*pred_and_target)


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
    pred_invalid = 0
    tgt_invalid = 0
    with Pool(args.num_processes) as pool:
        for i, (f1, pred_inv, tgt_inv) in tqdm(
            enumerate(pool.imap(
                calc_f1_map,
                zip(predictions, targets),
                chunksize=16
            )),
            desc="evaluating simple questions",
            total=len(targets),
            leave=False
        ):
            if args.save_invalid and f1 is None:
                with open(args.save_invalid, "a", encoding="utf8") as f:
                    f.write(f"{inputs[i]}\n{predictions[i]}\n{targets[i]}\n\n")
            if args.save_incorrect and f1 is not None and f1 < 1.0:
                with open(args.save_incorrect, "a", encoding="utf8") as f:
                    f.write(f"{inputs[i]}\n{predictions[i]}\n{targets[i]}\n\n")

            if pred_inv:
                pred_invalid += 1
                f1 = 0.0
            if tgt_inv:
                tgt_invalid += 1
                f1 = 0.0
            f1s.append(f1)
    print(
        f"Query-averaged F1: {sum(f1s) / len(f1s):.2%} "
        f"({pred_invalid:,} invalid predictions, "
        f"{pred_invalid / len(f1s):.2%} | "
        f"{tgt_invalid:,} invalid targets, "
        f"{tgt_invalid / len(f1s):.2%})"
    )


if __name__ == "__main__":
    evaluate(parse_args())
