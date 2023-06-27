import re
import requests
from typing import Dict, List, Callable, Any, Tuple


from text_correction_utils import prefix
from text_correction_utils.api.table import generate_table

VAR_REGEX = r"<bov>(.+?)<eov>"
PROP_REGEX = r"<bop>(.+?)<eop>"
ENT_REGEX = r"<boe>(.+?)<eoe>"

SELECT_REGEX = r"SELECT\s+(.*)\s+WHERE"
WHERE_REGEX = r"WHERE\s*{(.*)}"

WD_ENT_URL = "http://www.wikidata.org/entity/"
WD_PROP_URL = "http://www.wikidata.org/prop/direct/"
RDFS_URL = "http://www.w3.org/2000/01/rdf-schema#"
WD_QLEVER_URL = "https://qlever.cs.uni-freiburg.de/api/wikidata"

SPARQL_PREFIX = "Generate SPARQL query >> "


def load_str_index(path: str) -> Dict[int, List[str]]:
    with open(path, "r", encoding="utf8") as f:
        index = {}
        for line in f:
            split = line.strip().split("\t")
            assert len(split) == 2
            obj_name = split[0]
            obj_id = int(split[1])
            if obj_id not in index:
                index[obj_id] = [obj_name]
            else:
                index[obj_id].append(obj_name)
        return index


def load_id_index(path: str) -> Dict[int, List[int]]:
    with open(path, "r", encoding="utf8") as f:
        index = {}
        for line in f:
            split = line.strip().split("\t")
            assert len(split) == 2
            obj_id_1 = int(split[0])
            obj_id_2 = int(split[1])
            if obj_id_1 not in index:
                index[obj_id_1] = [obj_id_2]
            else:
                index[obj_id_1].append(obj_id_2)
        return index


def _replace(
    s: str,
    pattern: str,
    replacement_fn: Callable[[str], str],
) -> Tuple[str, List[str]]:
    org_len = len(s)
    len_diff = 0
    matches = []
    for match in re.finditer(pattern, s):
        replacement = replacement_fn(match.group(1))
        matches.append(match.group(1).strip())
        print(f"Replacing '{match.group(1)}' with '{replacement}'")
        start = match.start() + len_diff
        end = match.end() + len_diff
        s = s[:start] + replacement + s[end:]
        len_diff = len(s) - org_len
    return s, list(set(matches))


def replace_vars(s: str) -> Tuple[str, List[str]]:
    return _replace(s, VAR_REGEX, lambda v: f"?{v.strip()}")


def replace_entities(s: str, index: prefix.Vec, prefix: str = "") -> str:
    return _replace(
        s,
        ENT_REGEX,
        lambda e: f"{prefix}Q{index.get(e.strip().encode('utf8'))}"
    )[0]


def replace_properties(s: str, index: prefix.Vec, prefix: str = "") -> str:
    return _replace(
        s,
        PROP_REGEX,
        lambda p: f"{prefix}P{index.get(p.strip().encode('utf8'))}"
    )[0]


def replace_brackets(s: str) -> str:
    return s.replace("<bob>", "{").replace("<eob>", "}")


def _label_statement(
    var: str,
    lang: str = "en"
) -> str:
    return f"OPTIONAL {{ ?{var} rdfs:label ?{var}Label . " \
        f"FILTER(LANG(?{var}Label) = \"{lang}\") . }}"


def _inject_labels(
    s: str,
    vars: List[str],
    lang: str = "en"
) -> str:
    org_s = s
    # first inject OPTIONAL statements into WHERE clause
    # to get the labels
    where_matches = list(re.finditer(
        WHERE_REGEX,
        s,
        flags=re.DOTALL
    ))
    if len(where_matches) != 1:
        return s
    end = where_matches[0].end(1)
    label_s = " ".join(_label_statement(var, lang) for var in vars)
    s = s[:end].rstrip() + f" {label_s} " + s[end:].lstrip()

    # then add label varaibles to the SELECT clause
    select_matches = list(re.finditer(SELECT_REGEX, s))
    if len(select_matches) != 1:
        return org_s
    end = select_matches[0].end(1)
    label_s = " ".join(f"?{var}Label" for var in vars)
    s = s[:end].rstrip() + f" {label_s} " + s[end:].lstrip()
    return s


def postprocess_output(
    s: str
) -> str:
    s = _replace(
        s,
        r"(<[be]o[vepb]>)",
        lambda p: " " + p.strip() + " "
    )[0]
    s = re.sub(r"\s+", " ", s, flags=re.DOTALL).strip()
    s = re.sub(
        r"(<bo[vep]>)\s*(.*?)\s*(<eo[vep]>)",
        r"\1\2\3",
        s
    )
    return s


def wikidata_prefixes() -> List[str]:
    return [
        f"PREFIX wd: <{WD_ENT_URL}>",
        f"PREFIX wdt: <{WD_PROP_URL}>",
    ]


def prepare_sparql_query(
    s: str,
    entity_index: prefix.Vec,
    property_index: prefix.Vec,
    with_labels: bool = False,
    lang: str = "en"
) -> str:
    s, vars = replace_vars(s)
    s = replace_entities(s, entity_index, prefix="wd:")
    s = replace_properties(s, property_index, prefix="wdt:")
    s = replace_brackets(s)
    prefix = " ".join(wikidata_prefixes())
    if with_labels:
        prefix += f" PREFIX rdfs: <{RDFS_URL}>"
        s = _inject_labels(s, vars, lang)
    query = prefix + " " + s
    return query


def query_qlever(
    sparql_query: str,
) -> List[Dict[str, Any]]:
    response = requests.get(WD_QLEVER_URL, params={"query": sparql_query})
    if response.status_code != 200:
        msg = response.json().get("exception", "unknown exception")
        raise RuntimeError(
            f"query \n{sparql_query}\n failed with "
            f"status code {response.status_code}:\n{msg}"
        )
    return response.json()["results"]["bindings"]


def format_qlever_result(
    result: List[Dict[str, Any]],
    max_column_width: int = 80,
) -> str:
    if len(result) == 0:
        return "no results"

    columns = sorted(result[0].keys())
    if len(columns) == 0:
        return "no bindings"

    return generate_table(
        headers=[columns],
        data=[
            [
                obj[col]["value"] if col in obj else "-"
                for col in columns
            ]
            for obj in result
        ],
        alignments=["left"] * len(columns),
        max_column_width=max_column_width,
    )
