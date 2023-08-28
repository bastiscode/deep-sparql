WD_PROP=data/kg-index/wikidata-properties-index.tsv
WD_ENT=data/kg-index/wikidata-entities-index.tsv
WD_EX=""
WD_SIMPLE_EX=""

.PHONY: simple_data
simple_data:
	@echo "Preparing simple questions"
	@python scripts/prepare_data.py \
	--wikidata-simple-questions third_party/KGQA-datasets/simple_wikidata_qa \
	--output data/wikidata-simplequestions \
	--entity-index $(WD_ENT) \
	--property-index $(WD_PROP) \
	--example-index $(WD_SIMPLE_EX) \
	--progress

.PHONY: data
data:
	@echo "Preparing lc quad wikidata"
	@python scripts/prepare_data.py \
	--lc-quad2-wikidata third_party/KGQA-datasets/lcquad_v2 \
	--output data/wikidata-lcquad2 \
	--entity-index $(WD_ENT) \
	--property-index $(WD_PROP) \
	--example-index $(WD_EX) \
	--progress
	@echo "Preparing qald 10"
	@python scripts/prepare_data.py \
	--qald-10 third_party/KGQA-datasets/qald/qald-10.py \
	--output data/wikidata-qald10 \
	--entity-index $(WD_ENT) \
	--property-index $(WD_PROP) \
	--example-index $(WD_EX) \
	--progress
	@echo "Preparing mcwq"
	@python scripts/prepare_data.py \
	--mcwq data/raw/mcwq \
	--output data/wikidata-mcwq \
	--entity-index $(WD_ENT) \
	--property-index $(WD_PROP) \
	--example-index $(WD_EX) \
	--progress
	@echo "Preparing qa wiki"
	@python scripts/prepare_data.py \
	--qa-wiki data/raw/qa_wiki/qa_wiki.tsv \
	--output data/wikidata-qa-wiki \
	--entity-index $(WD_ENT) \
	--property-index $(WD_PROP) \
	--example-index $(WD_EX) \
	--progress

MODEL=roberta-base
BATCH_SIZE=32

.PHONY: example-indices
example-indices:
	@echo "Preparing wikidata example index"
	@python scripts/prepare_vector_index_data.py \
	--inputs data/wikidata-lcquad2/train_input.txt \
	data/wikidata-qald10/train_input.txt \
	data/wikidata-mcwq/train_input.txt \
	data/wikidata-qa-wiki/train_input.txt \
	--targets data/wikidata-lcquad2/train_sparql.txt \
	data/wikidata-qald10/train_sparql.txt \
	data/wikidata-mcwq/train_sparql.txt \
	data/wikidata-qa-wiki/train_sparql.txt \
	--output data/example-index/wikidata.txt
	@echo "Preparing wikidata simple questions example index"
	@python scripts/prepare_vector_index_data.py \
	--inputs data/wikidata-simplequestions/train_input.txt \
	--targets data/wikidata-simplequestions/train_sparql.txt \
	--output data/example-index/wikidata_simplequestions.txt
	@echo "Building wikidata example index"
	@python scripts/build_vector_index.py \
	--data data/example-index/wikidata.txt \
	--output data/example-index/wikidata-$(MODEL) \
	--model $(MODEL) --batch-size $(BATCH_SIZE) --overwrite
	@echo "Building wikidata simplequestions example index"
	@python scripts/build_vector_index.py \
	--data data/example-index/wikidata_simplequestions.txt \
	--output data/example-index/wikidata-simplequestions-$(MODEL) \
	--model $(MODEL) --batch-size $(BATCH_SIZE) --overwrite

TOKENIZER = "t5"

.PHONY: prefix-indices
prefix-indices:
	@echo "Creating wikidata prefix indices"
	@python third_party/text-correction-utils/scripts/create_prefix_vec.py \
	--file data/kg-index/wikidata-properties-index.tsv \
	--tokenizer-cfg configs/tokenizers/$(TOKENIZER).yaml \
	--out data/prefix-index/wikidata-$(TOKENIZER)-properties.bin
	@python third_party/text-correction-utils/scripts/create_prefix_vec.py \
	--file data/kg-index/wikidata-entities-index.tsv \
	--tokenizer-cfg configs/tokenizers/$(TOKENIZER).yaml \
	--out data/prefix-index/wikidata-$(TOKENIZER)-entities.bin
	@python third_party/text-correction-utils/scripts/create_prefix_vec.py \
	--file data/kg-index/wikidata-entities-small-index.tsv \
	--tokenizer-cfg configs/tokenizers/$(TOKENIZER).yaml \
	--out data/prefix-index/wikidata-$(TOKENIZER)-entities-small.bin

.PHONY: indices
indices: prefix-indices example-indices

.PHONY: all
all:
	make data
	make simple_data
	make indices
	make simple_data WD_SIMPLE_EX=data/example-index/wikidata-simplequestions-$(MODEL)
	make data WD_EX=data/example-index/wikidata-$(MODEL)
