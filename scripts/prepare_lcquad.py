import argparse
import os
import json
import re
import random
from typing import List

import numpy as np
from tqdm import tqdm

from deep_sparql.utils import (
    load_wikidata_index,
    wikidata_prefixes,
    SPARQL_PREFIX
)
from deep_sparql.vector import Index, sample_nearest_neighbors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--bracket-begin", type=str, default="<bob>")
    parser.add_argument("--bracket-end", type=str, default="<eob>")
    parser.add_argument("--entity-begin", type=str, default="<boe>")
    parser.add_argument("--entity-end", type=str, default="<eoe>")
    parser.add_argument("--property-begin", type=str, default="<bop>")
    parser.add_argument("--property-end", type=str, default="<eop>")
    parser.add_argument("--var-begin", type=str, default="<bov>")
    parser.add_argument("--var-end", type=str, default="<eov>")
    parser.add_argument("--entity-index", type=str, default=None)
    parser.add_argument("--property-index", type=str, default=None)
    parser.add_argument("--example-index", type=str, default=None)
    parser.add_argument("--max-num-examples", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-indices", action="store_true")
    return parser.parse_args()


def surround(s: str, op: str, cl: str) -> str:
    return f"{op}{s}{cl}"


def prepare(args: argparse.Namespace):
    if not args.no_indices:
        assert (
            args.entity_index is not None
            and args.property_index is not None
        ), "all indices must be provided if --no-indices is not set"
        entity_index, entity_redir = load_wikidata_index(args.entity_index)
        property_index, _ = load_wikidata_index(args.property_index)
    else:
        entity_index = entity_redir = property_index = {}

    if args.example_index is not None:
        example_index = Index.load(args.example_index)
    else:
        example_index = None

    os.makedirs(os.path.dirname(args.input), exist_ok=True)
    os.makedirs(os.path.dirname(args.target), exist_ok=True)

    with open(args.data, "r", encoding="utf8") as inf:
        data = json.load(inf)

    def clean(s: str) -> str:
        return re.sub(r"\s+", " ", s, flags=re.DOTALL).strip()

    num_invalid = 0
    num_too_long = 0
    lengths = []
    with open(args.target, "w", encoding="utf8") as tf, \
            open(args.input, "w", encoding="utf8") as inf:
        for sample in tqdm(data, "preparing lc quad", leave=False):
            if "sparql_wikidata" not in sample:
                num_invalid += 1
                continue

            sparql = clean(sample["sparql_wikidata"])

            question = sample["question"]
            if question is None:
                num_invalid += 1
                continue
            question = clean(question)

            raw_questions = [question]
            if "paraphrased_question" in sample:
                par = sample["paraphrased_question"]
                if isinstance(par, list):
                    raw_questions.extend(clean(p) for p in par)
                else:
                    raw_questions.append(clean(par))

            for q in raw_questions:
                lengths.append(len(q))

            # filter questions based on length
            questions = [
                q for q in raw_questions
                if len(q) <= 256
            ]
            num_too_long += len(raw_questions) - len(questions)
            if len(questions) == 0:
                continue

            if args.no_indices:
                for question in questions:
                    inf.write(f"{question}\n")
                    prefix = " ".join(wikidata_prefixes())
                    tf.write(f"{prefix} {sparql}\n")
                    continue

            # replace variables
            for match in re.finditer(
                r"\s+(\?([\w\d]+))",
                sparql,
                flags=re.IGNORECASE
            ):
                var_rep = surround(
                    match.group(2),
                    args.var_begin,
                    args.var_end
                )
                sparql = sparql.replace(match.group(1), var_rep)

            # replace brackets
            sparql = sparql.replace("{", args.bracket_begin)
            sparql = sparql.replace("}", args.bracket_end)

            # get entities
            try:
                ents = []
                for match in re.finditer(
                    r"(wd:Q\d+)",
                    sparql,
                    flags=re.IGNORECASE
                ):
                    ent = match.group(1)
                    if ent not in entity_index:
                        ent = entity_redir[ent]
                    ents.append((
                        ent,
                        [
                            surround(e, args.entity_begin, args.entity_end)
                            for e in
                            reversed(entity_index[ent])
                        ]
                    ))
            except KeyError as e:
                print(f"could not find {e} in entity index:\n{question}")
                num_invalid += 1
                continue

            try:
                # get properties
                props = [
                    (
                        match.group(),
                        [
                            surround(p, args.property_begin, args.property_end)
                            for p in
                            reversed(property_index[match.group(1)])
                        ]
                    )
                    for match in re.finditer(
                        r"((?:wdt|p|pq|ps):P\d+)",
                        sparql,
                        flags=re.IGNORECASE
                    )
                ]
            except KeyError as e:
                # print(f"could not find {e} in property index")
                num_invalid += 1
                continue

            if len(ents) == 0 and len(props) == 0:
                num_invalid += 1
                continue

            def replace_ents_and_props(sparql: str) -> List[str]:
                sparqls = []
                while True:
                    rep_sparql = sparql

                    for ent_match, entities in ents:
                        if len(entities) == 0:
                            return sparqls
                        ent = entities.pop()
                        rep_sparql = rep_sparql.replace(ent_match, ent, 1)

                    for prop_match, properties in props:
                        if len(properties) == 0:
                            return sparqls
                        prop = properties.pop()
                        rep_sparql = rep_sparql.replace(prop_match, prop, 1)

                    sparqls.append(rep_sparql)

            if example_index is not None:
                example_strs = sample_nearest_neighbors(
                    questions,
                    example_index,
                    args.max_num_examples,
                    args.batch_size
                )
            else:
                example_strs = [""] * len(questions)

            sparqls = replace_ents_and_props(sparql)
            for sparql in sparqls:
                idx = random.randint(0, len(questions) - 1)
                question = questions[idx]
                if example_strs[idx]:
                    inf.write(f"{example_strs[idx]} ")
                inf.write(f"{SPARQL_PREFIX}{question}\n")
                tf.write(f"{sparql}\n")

        print(
            f"invalid samples: {num_invalid} / {len(data)} "
            f"({num_invalid / len(data):.2%})"
        )
        print(
            f"too long questions (>256 chars): {num_too_long} / {len(data)} "
            f"({num_too_long / len(data):.2%})"
        )
        print(
            f"length percentiles: 90% = {np.percentile(lengths, 90):.2f}, "
            f"95% = {np.percentile(lengths, 95):.2f}, "
            f"99% = {np.percentile(lengths, 99):.2f}, "
            f"99.9% = {np.percentile(lengths, 99.9):.2f}"
        )


if __name__ == "__main__":
    prepare(parse_args())
