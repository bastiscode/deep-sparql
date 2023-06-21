FROM nvcr.io/nvidia/pytorch:23.01-py3

WORKDIR /deep-sparql

COPY . .

RUN pip install .

ENV DEEP_SPARQL_DOWNLOAD_DIR=/deep-sparql/download
ENV DEEP_SPARQL_CACHE_DIR=/deep-sparql/cache
ENV PYTHONWARNINGS="ignore"

ENTRYPOINT ["/opt/conda/bin/deep-sparql"]
