import random
import argparse
import os


from text_utils import io


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "-n",
        "--num-lines",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-p",
        "--postfix",
        type=str,
        default=None
    )
    return parser.parse_args()


def subset(args: argparse.Namespace):
    files = [
        io.load_text_file(path)
        for path in args.input
    ]
    assert all(len(files[0]) == len(file) for file in files), \
        "all files must have the same number of lines"
    if len(files) == 0:
        return
    num_lines = len(files[0])
    sample_size = min(num_lines, args.num_lines)
    indices = random.sample(
        list(range(num_lines)),
        sample_size
    )
    if args.postfix is None:
        args.postfix = f"_{sample_size}"
    for i, file in enumerate(files):
        ipt = args.input[i]
        path, ext = os.path.splitext(ipt)
        with open(path + args.postfix + ext, "w") as f:
            for index in indices:
                f.write(file[index] + "\n")


if __name__ == "__main__":
    subset(parse_args())
