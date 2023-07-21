import os
from io import TextIOWrapper
from typing import Iterator, Optional, Union, Iterable

from text_correction_utils.api.cli import TextCorrectionCli
from text_correction_utils.api.corrector import TextCorrector
from text_correction_utils import data

from deep_sparql import version
from deep_sparql.api.generator import SPARQLGenerator
from deep_sparql.api.server import SPARQLServer
from deep_sparql.utils import (
    query_qlever,
    format_qlever_result,
    generate_label_queries
)


class SPARQLCli(TextCorrectionCli):
    text_corrector_cls = SPARQLGenerator
    text_correction_server_cls = SPARQLServer

    def version(self) -> str:
        return version.__version__

    def format_output(self, item: data.InferenceData) -> Iterable[str]:
        self.cor: SPARQLGenerator
        if not self.cor.has_indices:
            yield item.text
            return
        query = self.cor.prepare_sparql_query(item.text)
        if not self.args.execute and not self.args.execute_with_labels:
            yield query
            return

        yield f"Output:\n{item.text}\n"
        yield f"Query:\n{query}\n"
        try:
            result = query_qlever(query)
            if self.args.execute_with_labels:
                lang = item.language or "en"
                label_queries = generate_label_queries(
                    result,
                    lang
                )
                label_results = {
                    var: query_qlever(q)
                    for var, q in label_queries.items()
                }
                assert all(
                    len(val) == len(result)
                    for val in label_results.values()
                )
                for var, val in label_results.items():
                    for i, v in enumerate(val):
                        if v is None:
                            continue
                        result[i][var] = v[var]
            formatted = format_qlever_result(result)
            nl = "\n" if self.args.interactive else ""
            yield f"Result:\n{formatted}" + nl
        except RuntimeError as e:
            yield f"query execution failed: {e}"

    def setup_corrector(self) -> TextCorrector:
        cor = super().setup_corrector()
        # perform some additional setup
        assert isinstance(cor, SPARQLGenerator)
        cor.set_inference_options(
            strategy=self.args.search_strategy,
            beam_width=self.args.beam_width,
            sample_top_k=self.args.sample_top_k
        )
        index_dir = os.environ.get("SPARQL_PREFIX_INDEX", None)
        if (
            self.args.entity_index is None
            and index_dir is not None
            and os.path.exists(os.path.join(index_dir, "entities.bin"))
        ):
            self.args.entity_index = os.path.join(index_dir, "entities.bin")
        if (
            self.args.property_index is None
            and index_dir is not None
            and os.path.exists(os.path.join(index_dir, "properties.bin"))
        ):
            self.args.property_index = os.path.join(
                index_dir,
                "properties.bin"
            )
        example_dir = os.environ.get("SPARQL_EXAMPLE_INDEX", None)
        if (
            self.args.example_index is None
            and example_dir is not None
            and os.path.exists(example_dir)
        ):
            self.args.example_index = example_dir
        cor.set_indices(
            self.args.entity_index,
            self.args.property_index,
            self.args.example_index
        )
        self.indices = cor.get_indices()
        return cor

    def correct_iter(
        self,
        corrector: SPARQLGenerator,
        iter: Iterator[data.InferenceData]
    ) -> Iterator[data.InferenceData]:
        yield from corrector.generate_iter(
            (
                (
                    corrector.prepare_questions(
                        [data.text],
                        self.args.n_examples
                    )[0],
                    data.language
                )
                for data in iter
            ),
            self.args.batch_size,
            self.args.batch_max_tokens,
            not self.args.unsorted,
            self.args.num_threads,
            raw=True,
            show_progress=self.args.progress
        )

    def correct_file(
        self,
        corrector: SPARQLGenerator,
        path: str,
        lang: Optional[str],
        out_file: Union[str, TextIOWrapper]
    ):
        # disable execution for file input
        self.args.execute_with_labels = False
        self.args.execute = False
        corrector.generate_file(
            path,
            self.args.input_format,
            out_file,
            self.args.output_format,
            lang,
            self.args.batch_size,
            self.args.batch_max_tokens,
            not self.args.unsorted,
            self.args.num_threads,
            show_progress=self.args.progress
        )


def main():
    parser = SPARQLCli.parser(
        "SPARQL Generator",
        "Generate SPARQL queries from natural language questions."
    )
    parser.add_argument(
        "--search-strategy",
        choices=["greedy", "beam", "sample"],
        type=str,
        default="greedy",
        help="Search strategy to use during decoding"
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=5,
        help="Beam width to use for beam search decoding"
    )
    parser.add_argument(
        "--sample-top-k",
        type=int,
        default=5,
        help="Sample from top k tokens during sampling decoding"
    )
    parser.add_argument(
        "--entity-index",
        type=str,
        default=None,
        help="Path to entity index file"
    )
    parser.add_argument(
        "--property-index",
        type=str,
        default=None,
        help="Path to property index file"
    )
    parser.add_argument(
        "--example-index",
        type=str,
        default=None,
        help="Path to example index directory"
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=3,
        help="Number of examples to use for each question"
    )
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument(
        "--execute",
        action="store_true",
        help="Execute the generated query and show the results"
    )
    execution.add_argument(
        "--execute-with-labels",
        action="store_true",
        help="Execute the generated query with labels and show the results"
    )
    SPARQLCli(parser.parse_args()).run()
