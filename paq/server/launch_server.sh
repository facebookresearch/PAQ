#!/usr/bin/env bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
python -m paq.download -v -n models.retrievers.retriever_multi_base_256
python -m paq.download -v -n paq.TQA_TRAIN_NQ_TRAIN_PAQ
python -m paq.download -v -n indices.multi_base_256_hnsw_sq8

python -m paq.server.server \
    --model_name_or_path data/models/retrievers/retriever_multi_base_256 \
    --qas_to_retrieve_from data/paq/TQA_TRAIN_NQ_TRAIN_PAQ/tqa-train-nq-train-PAQ.jsonl \
    --top_k 10 \
    --faiss_index_path data/indices/multi_base_256_hnsw_sq8.faiss \
    --fp16 \
    --memory_friendly_parsing \
    --verbose