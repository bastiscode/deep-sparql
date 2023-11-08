import argparse
import re
import os
import shutil
from urllib.parse import unquote_plus

from tqdm import tqdm

from text_utils import text
from deep_sparql.utils import (
    get_prefixes,
    load_kg_index
)

from prepare_data import prepare_sparqls


def prepare_file(
    file: str,
    entity_index: dict[str, list[str]],
    entity_redir: dict[str, str],
    property_index: dict[str, list[str]],
    args: argparse.Namespace
) -> tuple[int, int]:
    num_total = 0
    num_invalid = 0
    wd_prefixes = get_prefixes("wikidata")

    general_obj_pattern = re.compile(r"<http://.+?>")

    patterns = {}
    for prefix in wd_prefixes:
        short_name, pfx = prefix.split(":", 1)
        pfx = pfx.strip()
        assert pfx[0] == "<" and pfx[-1] == ">"
        patterns[prefix] = (
            short_name[len("PREFIX "):],
            re.compile(re.escape(pfx[:-1]) + r"(.+?)>")
        )

    clean_pattern = re.compile(r"\s+", flags=re.MULTILINE)

    num_lines, _ = text.file_size(file)
    with open(file, "r", encoding="utf8") as f:
        _ = next(f)  # forward headers
        for line in tqdm(
            f,
            desc=f"processing {os.path.basename(file)}",
            total=num_lines - 1,
            disable=not args.progress,
            leave=False
        ):
            sparql, _, source, _ = line.strip().split("\t")
            sparql = clean_pattern.sub(" ", unquote_plus(sparql)).strip()
            prefixes = []
            for prefix, (short, pattern) in patterns.items():
                changed = []

                def _sub_and_change(match: re.Match) -> str:
                    changed.append(match)
                    return f"{short}:{match.group(1)}"

                sparql = pattern.sub(
                    _sub_and_change,
                    sparql
                )
                if len(changed) == 0:
                    continue
                prefixes.append(prefix)

            num_total += 1

            match = general_obj_pattern.search(sparql)
            if match is not None:
                num_invalid += 1
                continue

            sparql = " ".join(prefixes) + " " + sparql
            sparqls = prepare_sparqls(
                "wikidata",
                sparql,
                entity_index,
                entity_redir,
                property_index,
                args.bracket_begin,
                args.bracket_end,
                args.var_begin,
                args.var_end,
                args.entity_begin,
                args.entity_end,
                args.property_begin,
                args.property_end
            )
            if sparqls is None:
                num_invalid += 1
                continue

            with open(
                os.path.join(args.output_dir, f"{source}.txt"),
                "a",
                encoding="utf8"
            ) as of:
                of.write(sparql + "\n")

    return num_total, num_invalid


def prepare(args: argparse.Namespace):
    entity_index, entity_redir = load_kg_index(
        args.entity_index,
        args.entity_redirects,
        args.progress
    )
    property_index, _ = load_kg_index(
        args.property_index,
        progress=args.progress
    )

    num_total = 0
    num_invalid = 0
    for file in args.files:
        total, invalid = prepare_file(
            file,
            entity_index,
            entity_redir,
            property_index,
            args
        )
        num_total += total
        num_invalid += invalid

    print(f"{num_invalid / num_total:.1%}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--files",
        nargs="+",
        type=str,
        required=True
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=True
    )
    parser.add_argument(
        "--progress",
        action="store_true"
    )
    parser.add_argument("--entity-index", type=str, required=True)
    parser.add_argument("--entity-redirects", type=str, default=None)
    parser.add_argument("--property-index", type=str, required=True)
    parser.add_argument("--bracket-begin", type=str, default="<bob>")
    parser.add_argument("--bracket-end", type=str, default="<eob>")
    parser.add_argument("--entity-begin", type=str, default="<boe>")
    parser.add_argument("--entity-end", type=str, default="<eoe>")
    parser.add_argument("--property-begin", type=str, default="<bop>")
    parser.add_argument("--property-end", type=str, default="<eop>")
    parser.add_argument("--var-begin", type=str, default="<bov>")
    parser.add_argument("--var-end", type=str, default="<eov>")
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
