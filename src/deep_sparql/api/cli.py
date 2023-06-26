from io import TextIOWrapper
from typing import Iterator, Optional, Union

from text_correction_utils.api.cli import TextCorrectionCli
from text_correction_utils.api.corrector import TextCorrector
from text_correction_utils import data

from deep_sparql import version
from deep_sparql.api.generator import SPARQLGenerator
from deep_sparql.api.server import SPARQLServer
from deep_sparql.utils import (
    SPARQL_PREFIX,
    prepare_sparql_query,
    query_qlever,
    format_qlever_result
)


class SPARQLCli(TextCorrectionCli):
    text_corrector_cls = SPARQLGenerator
    text_correction_server_cls = SPARQLServer

    def version(self) -> str:
        return version.__version__

    def format_output(self, item: data.InferenceData) -> str:
        if self.indices is None:
            return item.text
        query = prepare_sparql_query(
            item.text,
            *self.indices,
            with_labels=self.args.execute_with_labels,
        )
        if self.args.execute or self.args.execute_with_labels:
            try:
                result = query_qlever(query)
                formatted = format_qlever_result(result)
                nl = "\n" if self.args.interactive else ""
                s = nl + f"Output:\n{item.text}\n\n" \
                    f"Query:\n{query}\n\n" \
                    f"Result:\n{formatted}" + nl
                return s
            except RuntimeError as e:
                return f"query execution failed: {e}"
        return query

    def setup_corrector(self) -> TextCorrector:
        cor = super().setup_corrector()
        # perform some additional setup
        assert isinstance(cor, SPARQLGenerator)
        cor.set_inference_options(
            strategy=self.args.search_strategy,
            beam_width=self.args.beam_width,
            sample_top_k=self.args.sample_top_k
        )
        cor.set_indices(
            self.args.entity_index,
            self.args.property_index
        )
        self.indices = cor.get_indices()
        return cor

    def correct_iter(
        self,
        corrector: SPARQLGenerator,
        iter: Iterator[data.InferenceData]
    ) -> Iterator[data.InferenceData]:
        yield from corrector.correct_iter(
            ((SPARQL_PREFIX + data.text, data.language) for data in iter),
            self.args.batch_size,
            self.args.batch_max_tokens,
            not self.args.unsorted,
            self.args.num_threads,
            return_raw=True,
            show_progress=self.args.progress
        )

    def correct_file(
        self,
        corrector: SPARQLGenerator,
        path: str,
        lang: Optional[str],
        out_file: Union[str, TextIOWrapper]
    ):
        corrector.correct_file(
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
