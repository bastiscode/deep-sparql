import os
from io import TextIOWrapper
from typing import Iterator, Optional, Union, Iterable

from text_utils.api.cli import TextProcessingCli
from text_utils.api.processor import TextProcessor
from text_utils import data

from deep_sparql import version
from deep_sparql.api.generator import SPARQLGenerator
from deep_sparql.api.server import SPARQLServer
from deep_sparql.utils import (
    KNOWLEDGE_GRAPHS,
    query_qlever,
    format_sparql,
    format_qlever_result,
    add_labels,
    _qlever_ask_to_select_post_fn
)


class SPARQLCli(TextProcessingCli):
    text_processor_cls = SPARQLGenerator
    text_processing_server_cls = SPARQLServer

    def version(self) -> str:
        return version.__version__

    def format_output(self, item: data.InferenceData) -> Iterable[str]:
        self.cor: SPARQLGenerator
        execute = self.args.execute or self.args.execute_with_labels
        execute = execute and self.cor.has_kg_indices
        query = self.cor.prepare_sparql_query(
            item.text,
            execute or self.args.process is not None,
            _qlever_ask_to_select_post_fn if execute else None
        )
        if not execute:
            yield query
            return

        yield f"Output:\n{format_sparql(item.text, pretty=True)}\n"
        yield f"Query:\n{query}\n"
        try:
            result = query_qlever(
                query,
                self.args.kg,
                self.args.qlever_endpoint
            )
            if self.args.execute_with_labels:
                add_labels(
                    result,
                    query,
                    item.language or "en",
                    self.args.kg,
                    self.args.qlever_endpoint
                )
            formatted = format_qlever_result(result)
            nl = "\n" if self.args.interactive else ""
            yield f"Result:\n{formatted}" + nl
        except Exception as e:
            yield f"execution failed with {type(e).__name__}: {e}"

    def setup(self) -> TextProcessor:
        gen = super().setup()
        # perform some additional setup
        assert isinstance(gen, SPARQLGenerator)
        gen.set_inference_options(
            strategy=self.args.search_strategy,
            beam_width=self.args.beam_width,
            sample_top_k=self.args.sample_top_k,
            subgraph_constraining=self.args.subgraph_constraining,
            kg=self.args.kg,
            lang=self.args.lang or "en",
            max_length=self.args.max_length,
            use_cache=not self.args.no_kv_cache
        )

        index_dir = os.environ.get("SPARQL_PREFIX_INDEX", None)
        if index_dir is not None:
            entity_index = os.path.join(
                index_dir,
                f"{self.args.kg}-entities.bin"
            )
            property_index = os.path.join(
                index_dir,
                f"{self.args.kg}-properties.bin"
            )
        else:
            entity_index = None
            property_index = None
        if (
            self.args.entity_index is None
            and entity_index is not None
            and os.path.exists(entity_index)
        ):
            self.args.entity_index = entity_index
        if (
            self.args.property_index is None
            and property_index is not None
            and os.path.exists(property_index)
        ):
            self.args.property_index = property_index

        example_dir = os.environ.get("SPARQL_EXAMPLE_INDEX", None)
        if example_dir is not None:
            example_index = os.path.join(
                example_dir,
                self.args.kg,
            )
        else:
            example_index = None
        if (
            self.args.example_index is None
            and example_index is not None
            and os.path.exists(example_index)
        ):
            self.args.example_index = example_index

        gen.set_indices(
            self.args.entity_index,
            self.args.property_index,
            self.args.example_index
        )
        return gen

    def process_iter(
        self,
        processor: SPARQLGenerator,
        iter: Iterator[data.InferenceData]
    ) -> Iterator[data.InferenceData]:
        yield from processor.generate_iter(
            (
                (
                    processor.prepare_questions(
                        [data.text],
                        self.args.n_examples,
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

    def process_file(
        self,
        processor: SPARQLGenerator,
        path: str,
        lang: Optional[str],
        out_file: Union[str, TextIOWrapper]
    ):
        # disable execution for file input
        self.args.execute_with_labels = False
        self.args.execute = False
        processor.generate_file(
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
        "--no-kv-cache",
        action="store_true",
        help="Whether to use key and value caches during decoding"
    )
    parser.add_argument(
        "--subgraph-constraining",
        action="store_true",
        help="Whether to constrain entities and properties to already decoded "
        "subgraph, only works with entity and property indices"
    )
    parser.add_argument(
        "--kg",
        choices=list(KNOWLEDGE_GRAPHS),
        default="wikidata",
        help="knowledge graph to use"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Maximum supported input/output length in tokens"
    )
    parser.add_argument(
        "-E", "--entity-index",
        type=str,
        default=None,
        help="Path to entity index file"
    )
    parser.add_argument(
        "-P", "--property-index",
        type=str,
        default=None,
        help="Path to property index file"
    )
    parser.add_argument(
        "-X", "--example-index",
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
    parser.add_argument(
        "--qlever-endpoint",
        type=str,
        default=None,
        help="URL to QLever endpoint to use for query execution"
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
    args = parser.parse_args()
    # set default device to auto if not set
    # (different from underlying library which sets a single gpu as default)
    args.device = args.device or "auto"
    SPARQLCli(args).run()
