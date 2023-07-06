import re
import requests
from typing import Dict, List, Callable, Any, Tuple

from text_correction_utils import prefix, tokenization
from text_correction_utils.api.table import generate_table

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
        start = match.start() + len_diff
        end = match.end() + len_diff
        s = s[:start] + replacement + s[end:]
        len_diff = len(s) - org_len
    return s, list(set(matches))


def replace_vars(
    s: str,
    open: str = "<bov>",
    close: str = "<eov>"
) -> Tuple[str, List[str]]:
    return _replace(s, f"{open}(.+?){close}", lambda v: f"?{v.strip()}")


def replace_entities(
    s: str,
    index: prefix.Vec,
    prefix: str = "",
    open: str = "<bop>",
    close: str = "<eop>"
) -> str:
    return _replace(
        s,
        f"{open}(.+?){close}",
        lambda e: f"{prefix}Q{index.get(e.strip().encode('utf8'))}"
    )[0]


def replace_properties(
    s: str,
    index: prefix.Vec,
    prefix: str = "",
    open: str = "<bop>",
    close: str = "<eop>"
) -> str:
    return _replace(
        s,
        f"{open}(.+?){close}",
        lambda p: f"{prefix}P{index.get(p.strip().encode('utf8'))}"
    )[0]


def replace_brackets(
    s: str,
    open: str = "<bob>",
    close: str = "<eob>"
) -> str:
    return s.replace(open, "{").replace(close, "}")


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
    s: str,
    special_tokens: Tuple[str, ...] = (
        "<bob>",
        "<eob>",
    ),
    special_token_pairs: Tuple[Tuple[str, str], ...] = (
        ("<bov>", "<eov>"),
        ("<bop>", "<eop>"),
        ("<boe>", "<eoe>"),
    )
) -> str:
    escaped_special_tokens = []
    for tok in special_tokens:
        escaped_special_tokens.append(re.escape(tok))
    escaped_pairs = []
    for first, second in special_token_pairs:
        escaped_pairs.append(re.escape(first))
        escaped_pairs.append(re.escape(second))
    tokens = "|".join(set(escaped_special_tokens + escaped_pairs))
    s = _replace(
        s,
        f"({tokens})",
        lambda p: f" {p.strip()} "
    )[0]
    s = re.sub(r"\s+", " ", s, flags=re.DOTALL).strip()
    for i in range(0, len(escaped_pairs), 2):
        first = escaped_pairs[i]
        second = escaped_pairs[i + 1]
        s = re.sub(
            f"({first})\\s*(.*?)\\s*({second})",
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
    lang: str = "en",
    var_special_tokens: Tuple[str, str] = ("<bov>", "<eov>"),
    entity_special_tokens: Tuple[str, str] = ("<bop>", "<eop>"),
    property_special_tokens: Tuple[str, str] = ("<bop>", "<eop>"),
    bracket_special_tokens: Tuple[str, str] = ("<bob>", "<eob>")
) -> str:
    s, vars = replace_vars(s, *var_special_tokens)
    s = replace_entities(s, entity_index, "wd:", *entity_special_tokens)
    s = replace_properties(s, property_index, "wdt:", *property_special_tokens)
    s = replace_brackets(s, *bracket_special_tokens)
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
            f"query {sparql_query} failed with "
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


def special_token_or_token_ids(
    s: str,
    tok: tokenization.Tokenizer
) -> Tuple[str, List[int]]:
    token_id = tok.special_token_to_id(s)
    if token_id is not None:
        return s, [token_id]
    num_pfx = tok.num_prefix_tokens()
    num_sfx = tok.num_suffix_tokens()
    # adding an a here is necessary because some tokenizers implicitly add
    # a leading space, causing the token ids to potentially contain
    # the token id for a space in the beginning
    dummy_token_ids = tok.tokenize("a").token_ids[num_pfx:-num_sfx]
    token_ids = tok.tokenize("a" + s).token_ids[num_pfx:-num_sfx]
    token_ids = token_ids[len(dummy_token_ids):]
    return tok.de_tokenize(token_ids, False).strip(), token_ids


def longest_overlap(
    list1: List[int],
    list2: List[int]
) -> List[int]:
    min_len = min(len(list1), len(list2))
    overlap = 0

    for i in range(1, min_len + 1):
        if list1[-i:] == list2[:i]:
            overlap = i

    return list1[-overlap:] if overlap else []
