import argparse
import os
import random

from tqdm import tqdm


from deep_sparql.utils import (
    load_str_index,
    load_id_index,
    wikidata_prefixes,
    SPARQL_PREFIX
)
from deep_sparql.vector import Index, sample_nearest_neighbors


from text_correction_utils.io import load_text_file
from text_correction_utils import edit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--bracket-begin", type=str, default="<bob>")
    parser.add_argument("--bracket-end", type=str, default="<eob>")
    parser.add_argument("--entity-begin", type=str, default="<boe>")
    parser.add_argument("--entity-end", type=str, default="<eoe>")
    parser.add_argument("--property-begin", type=str, default="<bop>")
    parser.add_argument("--property-end", type=str, default="<eop>")
    parser.add_argument("--var-begin", type=str, default="<bov>")
    parser.add_argument("--var-end", type=str, default="<eov>")
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--entity-index", type=str, default=None)
    parser.add_argument("--property-index", type=str, default=None)
    parser.add_argument("--inverse-index", type=str, default=None)
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
            and args.inverse_index is not None
        ), "all indices must be provided if --no-indices is not set"
        entity_index = load_str_index(args.entity_index)
        property_index = load_str_index(args.property_index)
        inverse_index = load_id_index(args.inverse_index)
        # delete inverse relations for child, father, mother relations
        # from index
        child_invs = inverse_index.get(40, [])
        for prop_id in [22, 25]:
            if prop_id in child_invs:
                child_invs.remove(prop_id)
    else:
        entity_index = property_index = inverse_index = {}

    if args.example_index is not None:
        example_index = Index.load(args.example_index)
    else:
        example_index = None

    os.makedirs(os.path.dirname(args.input), exist_ok=True)
    os.makedirs(os.path.dirname(args.target), exist_ok=True)

    lines = load_text_file(args.data)

    if example_index is not None:
        questions = [line.split("\t")[-1].strip() for line in lines]
        example_strs = sample_nearest_neighbors(
            questions,
            example_index,
            args.max_num_examples,
            args.batch_size
        )
    else:
        example_strs = [""] * len(lines)

    with open(args.input, "w") as of, \
            open(args.target, "w") as tf:
        for i, line in enumerate(tqdm(
            lines,
            desc="preparing simple questions",
            leave=False
        )):
            subj, prop, obj, question = line.split("\t")
            if prop.startswith("R"):
                subj, obj = obj, subj
                subj = "x"
                prop = "P" + prop[1:]
            else:
                obj = "x"
                prop = "P" + prop[1:]

            if args.no_indices:
                of.write(f"{SPARQL_PREFIX}{question}\n")
                if subj == "x":
                    subj = "?" + subj
                    obj = "wd:" + obj
                else:
                    obj = "?" + obj
                    subj = "wd:" + subj
                prop = "wdt:" + prop
                prefix = " ".join(wikidata_prefixes())
                tf.write(
                    f"{prefix} SELECT ?x "
                    f"WHERE {{ {subj} {prop} {obj} . }}\n"
                )
                continue

            if subj != "x":
                subj_id = int(subj[1:])
                if subj_id not in entity_index:
                    continue
                subjs = [
                    surround(subj, args.entity_begin, args.entity_end)
                    for subj in reversed(entity_index[subj_id])
                ]
                objs = [surround(obj, args.var_begin, args.var_end)]
            else:
                obj_id = int(obj[1:])
                if obj_id not in entity_index:
                    continue
                objs = [
                    surround(obj, args.entity_begin, args.entity_end)
                    for obj in reversed(entity_index[obj_id])
                ]
                subjs = [surround(subj, args.var_begin, args.var_end)]

            prop_id = int(prop[1:])
            assert prop_id in property_index
            properties = property_index[prop_id]
            props = [
                (
                    surround(prop, args.property_begin, args.property_end),
                    edit.distance(prop, properties[0]),
                    False
                )
                for prop in properties
            ]
            for inv_prop_id in inverse_index.get(prop_id, []):
                properties = property_index[inv_prop_id]
                props += [
                    (
                        surround(prop, args.property_begin, args.property_end),
                        edit.distance(prop, properties[0]),
                        True
                    )
                    for prop in properties
                ]
            props.sort(
                key=lambda item: (-item[1], not item[2])  # type: ignore
            )

            var = surround('x', args.var_begin, args.var_end)

            while True:
                if len(subjs) == 0 or len(props) == 0 or len(objs) == 0:
                    break
                subj = subjs.pop()
                prop, _, is_inv = props.pop()
                obj = objs.pop()
                if is_inv:
                    subj, obj = obj, subj
                tf.write(
                    f"SELECT {var} "
                    f"WHERE {args.bracket_begin} "
                    f"{subj} {prop} {obj} . "
                    f"{args.bracket_end}\n"
                )
                if example_strs[i]:
                    of.write(f"{example_strs[i]} ")
                of.write(f"{SPARQL_PREFIX}{question}\n")


if __name__ == "__main__":
    prepare(parse_args())
