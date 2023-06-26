.PHONY: wd-simple
wd-simple:
	@echo "Preparing Wikidata Simple Questions"
	@python scripts/prepare_simple_questions.py \
	--data ${SPARQL}/wikidata-simplequestions/annotated_wd_data_train_answerable.txt \
	--input ${SPARQL}/wikidata-simplequestions/answerable/train_questions.txt \
	--target ${SPARQL}/wikidata-simplequestions/answerable/train_sparql.txt \
	--entity-index ${SPARQL}/wikidata-natural-language-index/wikidata-entities-popular-index.tsv \
	--property-index ${SPARQL}/wikidata-natural-language-index/wikidata-properties-index.tsv \
	--inverse-index ${SPARQL}/wikidata-natural-language-index/wikidata-properties-inverse-index.tsv

.PHONY: wd-data
wd-data: wd-simple

.PHONY: wd-indices
wd-indices: wd-nl-index wd-prefix-index wd-vec-index

.PHONY: wd-nl-index
wd-nl-index:
	@echo "Creating Wikidata natural language indices"
	@make -C third_party/wikidata-natural-language-index index \
	OUT_DIR=${SPARQL}/wikidata-natural-language-index

.PHONY: wd-prefix-index
wd-prefix-index:
	@echo "Creating Wikidata prefix indices"
	@cd third_party/text-correction-utils && \
	python scripts/create_prefix_vec.py \
	--file ${SPARQL}/wikidata-natural-language-index/wikidata-properties-index.tsv \
	--out ${SPARQL}/wikidata-prefix-index/properties.bin
	@cd third_party/text-correction-utils && \
	python scripts/create_prefix_vec.py \
	--file ${SPARQL}/wikidata-natural-language-index/wikidata-entities-popular-index.tsv \
	--out ${SPARQL}/wikidata-prefix-index/entities.bin

MODEL=t5-base
BATCH_SIZE=32

.PHONY: wd-vec-index
wd-vec-index:
	@echo "Creating Wikidata vector indices"
	@python scripts/build_vector_index.py \
	--prefix-index ${SPARQL}/wikidata-prefix-index/properties.bin \
	--output ${SPARQL}/wikidata-vector-index/$(MODEL)-properties.bin \
	--model $(MODEL) --batch-size $(BATCH_SIZE)
	@python scripts/build_vector_index.py \
	--prefix-index ${SPARQL}/wikidata-prefix-index/entities.bin \
	--output ${SPARQL}/wikidata-vector-index/$(MODEL)-entities.bin \
	--model $(MODEL) --batch-size $(BATCH_SIZE)
