import time
from typing import Dict, Any

from flask import Response, jsonify, request, abort

from text_correction_utils.api.server import TextCorrectionServer, Error
from text_correction_utils.api.utils import ProgressIterator

from deep_sparql.api.generator import SPARQLGenerator
from deep_sparql.utils import SPARQL_PREFIX, prepare_sparql_query


class SPARQLServer(TextCorrectionServer):
    text_corrector_cls = SPARQLGenerator

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        for cfg in config["models"]:
            if "entity_index" not in cfg or "property_index" not in cfg:
                continue
            if "path" in cfg:
                name = cfg["path"]
            else:
                name = cfg["name"]
            cor_name = self.name_to_text_corrector[name]
            (cor, _) = self.text_correctors[cor_name]
            assert isinstance(cor, SPARQLGenerator)
            cor.set_indices(
                cfg["entity_index"],
                cfg["property_index"]
            )
            self.logger.info(
                f"loaded indices from {cfg['entity_index']} "
                f"and {cfg['property_index']} for {cor_name}"
            )

        @self.server.route(f"{self.base_url}/answer", methods=["POST"])
        def _answer() -> Response:
            json = request.get_json()
            if json is None:
                return abort(Response("request body must be json", status=400))
            elif "model" not in json:
                return abort(Response("missing model in json", status=400))
            elif "questions" not in json:
                return abort(Response("missing questions in json", status=400))

            with_labels = json.get("labels", True)
            search_strategy = json.get("search_strategy", "greedy")
            beam_width = json.get("beam_width", 5)

            try:
                with self.text_corrector(json["model"]) as cor:
                    if isinstance(cor, Error):
                        return abort(cor.to_response())
                    assert isinstance(cor, SPARQLGenerator)
                    cor.set_inference_options(
                        search_strategy,
                        beam_width
                    )
                    start = time.perf_counter()
                    iter = ProgressIterator(
                        (
                            (SPARQL_PREFIX + q.strip(), None)
                            for q in json["questions"]
                        ),
                        size_fn=lambda e: len(e[0].encode("utf8"))
                    )
                    generated = []
                    sparql = []
                    for item in cor.generate_iter(
                        iter,
                        raw=True
                    ):
                        generated.append(item.text)
                        if not cor.has_indices:
                            continue
                        query = prepare_sparql_query(
                            item.text,
                            *cor.get_indices(),
                            with_labels=with_labels,
                            lang=item.language or "en",
                            var_special_tokens=cor._var_special_tokens,
                            entity_special_tokens=cor._ent_special_tokens,
                            property_special_tokens=cor._prop_special_tokens,
                            bracket_special_tokens=cor._bracket_special_tokens
                        )
                        sparql.append(query)

                    end = time.perf_counter()
                    b = iter.total_size
                    s = end - start

                    output = {
                        "raw": generated,
                        "runtime": {"b": b, "s": s}
                    }
                    if cor.has_indices:
                        output["sparql"] = sparql
                    return jsonify(output)

            except Exception as error:
                return abort(
                    Response(
                        f"request failed with unexpected error: {error}",
                        status=500
                    )
                )
