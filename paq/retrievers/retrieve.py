#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import torch
import logging
import time
import faiss
import numpy as np
from paq.retrievers.retriever_utils import load_retriever
from paq.paq_utils import load_jsonl, dump_jsonl, parse_vectors_from_directory
from paq.retrievers.embed import embed
from copy import deepcopy

logger = logging.getLogger(__name__)

CUDA = torch.cuda.is_available()


def get_output_format(qas_to_answer, qas_to_retrieve_from, top_indices, top_scores):
    results = []
    for qa_ind, qa in enumerate(qas_to_answer):
        res = []
        for score_ind, ind in enumerate(top_indices[qa_ind]):
            score = top_scores[qa_ind][score_ind]
            ret_qa = deepcopy(qas_to_retrieve_from[ind])
            ret_qa['score'] = float(score)
            res.append(ret_qa)
        results.append(res)

    return [{'input_qa': in_qa, 'retrieved_qas': ret_qas} for in_qa, ret_qas in zip(qas_to_answer, results)]


def _torch_mips(index, query_batch, top_k):
    sims = torch.matmul(query_batch, index.t())
    return sims.topk(top_k)


def _flat_index_mips(index, query_batch, top_k):
    return index.search(query_batch.numpy(), top_k)


def _aux_dim_index_mips(index, query_batch, top_k):
    # querying faiss indexes for MIPS using a euclidean distance index, used with hnsw
    aux_dim = query_batch.new(query_batch.shape[0]).fill_(0)
    aux_query_batch = torch.cat([query_batch, aux_dim.unsqueeze(-1)], -1)
    return index.search(aux_query_batch.numpy(), top_k)


def _get_mips_function(index):
    if type(index) == torch.Tensor:
        return _torch_mips
    elif 'hnsw' in str(type(index)).lower():
        return _aux_dim_index_mips
    else:
        return _flat_index_mips


def mips(index, queries, top_k, n_queries_to_parallelize=256):
    t = time.time()
    all_top_indices = None
    all_top_scores = None

    _mips = _get_mips_function(index)

    for mb in range(0, len(queries), n_queries_to_parallelize):
        query_batch = queries[mb:mb + n_queries_to_parallelize].float()
        scores, top_indices = _mips(index, query_batch, top_k)

        all_top_indices = top_indices if all_top_indices is None else np.concatenate([all_top_indices, top_indices])
        all_top_scores = scores if all_top_scores is None else np.concatenate([all_top_scores, scores])

        delta = time.time() - t
        logger.info(
            f'{len(all_top_indices)}/ {len(queries)} queries searched in {delta:04f} '
            f'seconds ({len(all_top_indices) / delta} per second)')

    assert len(all_top_indices) == len(queries)

    delta = time.time() - t
    logger.info(f'Index searched in {delta:04f} seconds ({len(queries) / delta} per second)')
    return all_top_indices, all_top_scores


def run_queries(model, tokenizer, qas_to_retrieve_from, qas_to_answer, top_k, index=None,
                batch_size=128, fp16=False, n_queries_to_parallelize=2048):

    if index is None:
        index = embed(model, tokenizer, qas_to_retrieve_from, bsz=batch_size, fp16=fp16).float()

    logger.info('Embedding QAs to answer:')
    embedded_qas_to_answer = embed(model, tokenizer, qas_to_answer, bsz=batch_size, fp16=fp16)
    logger.info('Running MIPS search:')
    top_indices, top_scores = mips(index, embedded_qas_to_answer, top_k, n_queries_to_parallelize=n_queries_to_parallelize)

    return get_output_format(qas_to_answer, qas_to_retrieve_from, top_indices, top_scores)


def _load_index_if_exists(faiss_index_path, precomputed_embeddings_dir, n_vectors_to_load=None, memory_friendly=False, efsearch=128):
    index = None
    if faiss_index_path is not None:
        assert precomputed_embeddings_dir is None, "Do not specify both a --faiss_index_path and --precomputed_embeddings_dir"
        logger.info('Loading Faiss index:')
        index = faiss.read_index(faiss_index_path)
        if hasattr(index, 'hnsw'):
             index.hnsw.efSearch = efsearch

    elif precomputed_embeddings_dir is not None:
        logger.info('Loading vectors index from file:')
        index = parse_vectors_from_directory(
            precomputed_embeddings_dir,
            memory_friendly=memory_friendly,
            size=n_vectors_to_load
        ).float()

    logger.info('Index loaded') if index is not None else None
    return index


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "Perform REPAQ QA-Pair Retrieval. This program will embed a file of questions which need"
        " answering passed as `--qas_to_answer`. These will be answered by retrieving QA-pairs from a "
        " set of QA pairs to retrieve answers from, passed in as `--qas_to_retrieve_from`. "
        " The program can retrieve either from a prebuilt faiss index for `qas_to_retrieve_from`, "
        "or a directory of precomputed vectors, or, if neither are passed in, "
        "will embed the `qas_to_retrieve_from` before performing retrieval"
    )
    parser.add_argument('--model_name_or_path', type=str, required=True, help='path to HF model dir')
    parser.add_argument('--qas_to_answer', type=str, required=True, help="path to questions to answer in jsonl format")
    parser.add_argument('--qas_to_retrieve_from', type=str, required=True,
                        help="path to QA-pairs to retrieve answers from in jsonl format")
    parser.add_argument('--top_k', type=int, default=50, help="top K QA-pairs to retrieve for each input question")
    parser.add_argument('--output_file', type=str, required=True, help='Path to write jsonl results to')
    parser.add_argument('--faiss_index_path', default=None, type=str, help="Path to faiss index, if retrieving from a faiss index")
    parser.add_argument('--precomputed_embeddings_dir', default=None, type=str, help="path to a directory of vector embeddings if retrieving from raw embeddign vectors")
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for embedding questions for querying')
    parser.add_argument('--n_queries_to_parallelize', type=int, default=256, help="query batch size")
    parser.add_argument('-v', '--verbose', action="store_true")
    parser.add_argument('--memory_friendly_parsing', action='store_true', help='Pass this to load files more slowly, but save memory')
    parser.add_argument('--faiss_efsearch', type=int, default=128, help='EFSearch searchtime parameter for hnsw , higher is more accuate but slower')

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    qas_to_answer = load_jsonl(args.qas_to_answer, memory_friendly=args.memory_friendly_parsing)
    qas_to_retrieve_from = load_jsonl(args.qas_to_retrieve_from, memory_friendly=args.memory_friendly_parsing)

    index = _load_index_if_exists(
        args.faiss_index_path,
        args.precomputed_embeddings_dir,
        n_vectors_to_load=len(qas_to_retrieve_from),
        memory_friendly=args.memory_friendly_parsing,
        efsearch=args.faiss_efsearch
    )
    model, tokenizer = load_retriever(args.model_name_or_path)

    retrieved_answers = run_queries(
        model,
        tokenizer,
        qas_to_retrieve_from,
        qas_to_answer,
        args.top_k,
        index,
        args.batch_size,
        args.fp16,
        args.n_queries_to_parallelize,
    )
    dump_jsonl(retrieved_answers, args.output_file)
