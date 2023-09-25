import argparse
import random
import os
import re
import json
import collections
from typing import List, Tuple, Optional, Dict

from tqdm import tqdm
from datasets import load_dataset


from deep_sparql.utils import (
    load_kg_index,
    load_inverse_index,
    wikidata_prefixes,
    format_input,
    uppercase_sparql_keywords
)
from deep_sparql.vector import Index, get_nearest_neighbors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    data = parser.add_mutually_exclusive_group(required=True)

    # wikidata
    data.add_argument("--wikidata-simple-questions", type=str)
    data.add_argument("--qald-10", type=str)
    data.add_argument("--time-questions", type=str)
    data.add_argument("--cron-questions", type=str)
    data.add_argument("--mkqa", type=str)
    data.add_argument("--mintaka", type=str)
    data.add_argument("--lc-quad2-wikidata", type=str)
    data.add_argument("--mcwq", type=str)
    data.add_argument("--qa-wiki", type=str)
    data.add_argument("--kqa-pro", type=str)

    # freebase
    data.add_argument("--graph-questions", type=str)
    data.add_argument("--wqsp", type=str)
    data.add_argument("--complex-web-questions", type=str)
    data.add_argument("--freebase-simple-questions", type=str)
    data.add_argument("--30mqa", type=str)
    data.add_argument("--cfq", type=str)
    data.add_argument("--grail-qa", type=str)
    data.add_argument("--freebase-qa", type=str)

    # dbpedia
    data.add_argument("--lc-quad2-dbpedia", type=str)
    data.add_argument("--qald-9-plus", type=str)
    data.add_argument("--simple-dbpedia-qa", type=str)
    data.add_argument("--mlpq", type=str)
    data.add_argument("--monument", type=str)

    parser.add_argument("--output", type=str, required=True)
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
    parser.add_argument("--inverse-index", type=str, default=None)
    parser.add_argument("--example-index", type=str, default=None)
    parser.add_argument("--max-num-examples", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


Sample = collections.namedtuple("Sample", ["question", "sparql", "result"])


SPLIT_RENAME = {
    "train": "train",
    "test": "test",
    "dev": "val",
    "valid": "val",
    "validation": "val",
}


def load_data(args: argparse.Namespace) -> Tuple[
    str,
    Dict[str, List[Sample]]
]:
    output = {}
    if args.wikidata_simple_questions is not None:
        kg = "wikidata"
        data = load_dataset(args.wikidata_simple_questions)
        for split, items in data.items():
            split = SPLIT_RENAME.get(split, split)
            assert split in {"train", "val", "test"}
            samples = []
            for item in items:
                question = item["question"]
                subj = item["answer"]["subject"]
                obj = item["answer"]["object"]
                prop = item["answer"]["predicate"]

                if prop.startswith("R"):
                    subj, obj = obj, subj
                    subj = "x"
                    prop = "P" + prop[1:]
                else:
                    obj = "x"
                prop = "wdt:" + prop

                if subj == "x":
                    subj = "?" + subj
                    obj = "wd:" + obj
                else:
                    obj = "?" + obj
                    subj = "wd:" + subj

                sparql = f"SELECT ?x WHERE {{ {subj} {prop} {obj} . }}"
                samples.append(Sample(question, sparql, None))
            output[split] = samples
    elif args.qald_10 is not None:
        kg = "wikidata"
        data = load_dataset(args.qald_10)
        for split, items in data.items():
            split = SPLIT_RENAME.get(split, split)
            assert split in {"train", "val", "test"}
            samples = []
            for item in items:
                sparql = item["query"]["sparql"]
                questions = [
                    q["string"]
                    for q in json.loads(item["question"])
                    if q["language"] == "en"
                ]
                # replace entities and properties
                sparql = re.sub(
                    r"<http://www.wikidata.org/entity/(Q\d+?)>",
                    lambda match: "wd:" + match.group(1),
                    sparql
                )

                def _rep_prop(m: re.Match) -> str:
                    pfx = m.group(1)
                    if pfx == "direct":
                        pfx = "wdt"
                    else:
                        raise RuntimeError(f"unknown prefix {pfx}")
                    return f"{pfx}:{m.group(2)}"

                sparql = re.sub(
                    r"<http://www.wikidata.org/prop/(?:(\S+?)/)?(P\d+?)>",
                    _rep_prop,
                    sparql
                )
                # remove prefixes
                sparql = PREFIX_REGEX.sub("", sparql).strip()
                for q in questions:
                    samples.append(Sample(q, sparql, None))
            output[split] = samples
    elif args.lc_quad2_wikidata is not None:
        kg = "wikidata"
        data = load_dataset(args.lc_quad2_wikidata, "lcquad2-wikidata")
        for split, items in data.items():
            split = SPLIT_RENAME.get(split, split)
            assert split in {"train", "val", "test"}
            samples = []
            for item in items:
                questions = [item["question"]]
                sparql = item["sparql"]
                for pq in item["paraphrased_question"]:
                    questions.append(pq)
                for q in questions:
                    if q is None or q.strip() == "":
                        continue
                    samples.append(Sample(q, sparql, None))
            output[split] = samples
    elif args.mcwq is not None:
        kg = "wikidata"
        with open(os.path.join(args.mcwq, "dataset.json"), "r") as inf:
            train_data = json.load(inf)
        with open(os.path.join(args.mcwq, "gold_test.json"), "r") as inf:
            test_data = json.load(inf)
        for data, split in [(train_data, "train"), (test_data, "test")]:
            samples = []
            for item in data:
                question = item["questionWithBrackets"]
                # sub out brackets
                question = re.sub(
                    r"\[(.+?)\]",
                    lambda m: m.group(1),
                    question
                )
                # repair some whitespace issues
                # words followed by 's
                question = re.sub(
                    r"(\w+)\s+('s)(?:\s+|$)",
                    lambda m: m.group(1) + m.group(2) + " ",
                    question
                )
                # punctuation with surrounding spaces
                question = re.sub(
                    r"\s+([,.?!;])(?:\s+|$)",
                    lambda m: m.group(1) + " ",
                    question
                )
                sparql = item["sparql"]
                samples.append(Sample(question, sparql, None))
            output[split] = samples
    elif args.qa_wiki is not None:
        kg = "wikidata"
        samples = []
        with open(args.qa_wiki, "r") as inf:
            for line in inf:
                line = line.strip()
                sparql, question = line.split("\t")
                samples.append(Sample(question, sparql, None))
        output["train"] = samples
    else:
        raise RuntimeError("unknown dataset")
    return kg, output


PREFIX_REGEX = re.compile(
    r"(prefix\s+\S+:\s*<.+?>)",
    flags=re.IGNORECASE | re.DOTALL
)
VAR_REGEX = re.compile(r"\s+(\?(\w+))")
WIKIDATA_ENT_REGEX = re.compile(r"(wd:Q\d+)")
WIKIDATA_PROP_REGEX = re.compile(r"((?:wdt|p|pq|pqn|ps|psn):P\d+)")
FREEBASE_ENT_REGEX = re.compile(r"(fb:m\.\w+)")
FREEBASE_PROP_REGEX = re.compile(r"(fb:\w+\.\w+\.\w+)")
DBPEDIA_ENT_REGEX = re.compile(r"(dbr:\w+)")
DBPEDIA_PROP_REGEX = re.compile(r"((?:dbo|dbp):\w+)")


def surround(s: str, op: str, cl: str) -> str:
    return f"{op}{s}{cl}"


def prepare_sparqls(
    kg: str,
    sparql: str,
    entity_index: Dict[str, List[str]],
    entity_redir: Dict[str, str],
    property_index: Dict[str, List[str]],
    args: argparse.Namespace
) -> Optional[List[str]]:
    if kg == "wikidata":
        ent_regex = WIKIDATA_ENT_REGEX
        prop_regex = WIKIDATA_PROP_REGEX
    else:
        raise RuntimeError("unknown knowledge graph")

    # replace variables
    for match in re.finditer(VAR_REGEX, sparql):
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
        for match in re.finditer(ent_regex, sparql):
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
    except KeyError:  # as e:
        # print(f"could not find {e} in entity index")
        return None

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
            for match in re.finditer(prop_regex, sparql)
        ]
    except KeyError:  # as e:
        # print(f"could not find {e} in property index")
        return None

    if len(ents) == 0 and len(props) == 0:
        return None

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

    return replace_ents_and_props(sparql)


def prepare(args: argparse.Namespace):
    kg, data = load_data(args)

    if args.entity_index is not None:
        entity_index, entity_redir = load_kg_index(
            args.entity_index,
            args.progress
        )
    else:
        entity_index = entity_redir = None

    if args.property_index is not None:
        property_index, _ = load_kg_index(
            args.property_index,
            args.progress
        )
    else:
        property_index = None

    if args.inverse_index is not None:
        inverse_index = load_inverse_index(args.inverse_index)
    else:
        inverse_index = None
    del inverse_index  # to avoid unused warnings

    if args.example_index is not None and args.example_index != "":
        example_index = Index.load(args.example_index)
    else:
        example_index = None

    has_examples = example_index is not None
    if has_examples:
        assert example_index is not None
        split_examples = {
            split: [
                # filter out exact question matches
                # which can happen for the training split
                [ex for ex, dist in nns if dist > 0.0]
                for nns in get_nearest_neighbors(
                    [
                        sample.question
                        for sample in data.get(split, [])
                    ],
                    example_index,
                    args.max_num_examples * (1 + (split == "train")) + 1,
                    args.batch_size,
                    progress=args.progress
                )]
            for split in ["train", "val", "test"]
        }
    else:
        split_examples = {}

    os.makedirs(args.output, exist_ok=True)

    if kg == "wikidata":
        prefixes = wikidata_prefixes()
    else:
        raise RuntimeError("unknown knowledge graph")
    prefix = " ".join(prefixes)

    for split, samples in data.items():
        examples = split_examples.get(split, [[]] * len(samples))
        input = os.path.join(
            args.output,
            f"{split}_input{'_examples' * has_examples}.txt"
        )
        assert len(samples) > 0, f"no samples for split {split}"
        has_sparql = samples[0].sparql is not None
        target_name = "sparql" if has_sparql else "result"
        target = os.path.join(
            args.output,
            f"{split}_{target_name}{'_examples' * has_examples}.txt"
        )
        raw = os.path.join(
            args.output,
            f"{split}_raw.txt"
        )
        print(f"found {len(samples):,} {split} samples")
        num_invalid = 0
        with open(input, "w") as inf, \
                open(target, "w") as tf, \
                open(raw, "w") as rf:
            for i, sample in enumerate(tqdm(
                samples,
                desc=f"processing and writing {split} samples",
                leave=False,
                disable=not args.progress
            )):
                # clean sample
                sample = Sample(
                    re.sub(
                        r"\s+",
                        " ",
                        sample.question,
                        flags=re.DOTALL
                    ).strip(),
                    uppercase_sparql_keywords(
                        re.sub(
                            r"\s+",
                            " ",
                            sample.sparql,
                            flags=re.DOTALL
                        ).strip(),
                        with_new_line=False
                    ) if has_sparql else None,
                    sample.result if not has_sparql else None
                )
                # skip too long questions
                if len(sample.question) > args.max_length:
                    num_invalid += 1
                    continue
                # test sets always contain full sparql targets
                # (with prefix and original kg ids) for evaluation
                if split == "test" or not has_sparql:
                    rf.write(sample.sparql + "\n")
                    inf.write(format_input(
                        sample.question,
                        examples[i][:args.max_num_examples],
                        kg
                    ) + "\n")
                    if has_sparql:
                        # write sparql query
                        tf.write(f"{prefix} {sample.sparql}\n")
                    else:
                        # write result set as space separated values
                        tf.write(
                            " ".join(r.strip() for r in sample.result)
                            + "\n"
                        )
                    continue

                assert (
                    entity_index is not None
                    and entity_redir is not None
                    and property_index is not None
                ), \
                    "for val and train splits, entity and property indices " \
                    "must be provided"

                sparqls = prepare_sparqls(
                    kg,
                    sample.sparql,
                    entity_index,
                    entity_redir,
                    property_index,
                    args
                )
                if sparqls is None:
                    num_invalid += 1
                    continue
                for sparql in sparqls:
                    if split == "train":
                        indices = list(range(len(examples[i])))
                        random.shuffle(indices)
                        exs = [
                            examples[i][j]
                            for j in sorted(indices[:args.max_num_examples])
                        ]
                    else:
                        exs = examples[i][:args.max_num_examples]
                    inf.write(format_input(
                        sample.question,
                        exs,
                        kg
                    ) + "\n")
                    tf.write(sparql + "\n")
                    rf.write(sample.sparql + "\n")
        print(
            f"{num_invalid:,} of {len(samples):,} "
            f"({num_invalid / len(samples):.1%}) "
            f"{split} samples were invalid"
        )


if __name__ == "__main__":
    prepare(parse_args())
