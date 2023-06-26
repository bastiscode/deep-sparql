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

.PHONY: wd
wd: wd-simple
