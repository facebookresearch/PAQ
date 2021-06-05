#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import torch
import logging
import faiss
import os
import random
from paq.paq_utils import parse_vectors_from_directory

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def get_vector_sample(cached_embeddings_path, sample_fraction):
    samples = []
    max_phi = -1
    N = 0
    vectors = parse_vectors_from_directory(cached_embeddings_path, as_chunks=True)
    for chunk in vectors:
        phis = (chunk ** 2).sum(1)
        max_phi = max(max_phi, phis.max())
        N += chunk.shape[0]
        if sample_fraction == 1.0:
            chunk_sample = chunk
        else:
            chunk_sample = chunk[random.sample(range(0, len(chunk)), int(len(chunk) * sample_fraction))]
        samples.append(chunk_sample)

    del vectors
    vector_sample = torch.cat(samples)
    return vector_sample, max_phi, N


def get_vectors_dim(cached_embeddings_path):
    vectors = parse_vectors_from_directory(cached_embeddings_path, as_chunks=True)
    vector_size = next(vectors).shape[1]
    del(vectors)
    return vector_size


def augment_vectors(vectors, max_phi):
    phis = (vectors ** 2).sum(1)
    aux_dim = torch.sqrt(max_phi - phis)
    vectors = torch.cat([vectors, aux_dim.unsqueeze(-1)], -1)
    return vectors


def build_index_streaming(cached_embeddings_path,
                          output_path,
                          hnsw=False,
                          sq8_quantization=False,
                          fp16_quantization=False,
                          store_n=256,
                          ef_search=32,
                          ef_construction=80,
                          sample_fraction=0.1,
                          indexing_batch_size=5000000,
                          ):

    vector_size = get_vectors_dim(cached_embeddings_path)

    if hnsw:
        if sq8_quantization:
            index = faiss.IndexHNSWSQ(vector_size + 1, faiss.ScalarQuantizer.QT_8bit, store_n)
        elif fp16_quantization:
            index = faiss.IndexHNSWSQ(vector_size + 1, faiss.ScalarQuantizer.QT_fp16, store_n)
        else:
            index = faiss.IndexHNSWFlat(vector_size + 1, store_n)

        index.hnsw.efSearch = ef_search
        index.hnsw.efConstruction = ef_construction
    else:
        if sq8_quantization:
            index = faiss.IndexScalarQuantizer(vector_size, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_L2)
        elif fp16_quantization:
            index = faiss.IndexScalarQuantizer(vector_size, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_L2)
        else:
            index = faiss.IndexIP(vector_size + 1, store_n)

    vector_sample, max_phi, N = get_vector_sample(cached_embeddings_path, sample_fraction)
    if hnsw:
        vector_sample = augment_vectors(vector_sample, max_phi)

    if sq8_quantization or fp16_quantization: # index requires training
        vs = vector_sample.numpy()
        logging.info(f'Training Quantizer with matrix of shape {vs.shape}')
        index.train(vs)
        del vs
    del vector_sample

    chunks_to_add = []
    added = 0
    for vector_chunk in parse_vectors_from_directory(cached_embeddings_path, as_chunks=True):
        if hnsw:
            vector_chunk = augment_vectors(vector_chunk, max_phi)

        chunks_to_add.append(vector_chunk)

        if sum(c.shape[0] for c in chunks_to_add) > indexing_batch_size:
            logging.info(f'Adding Vectors {added} -> {added + to_add.shape[0]} of {N}')
            to_add = torch.cat(chunks_to_add)
            chunks_to_add = []
            index.add(to_add)
            added += 1

    if len(chunks_to_add) > 0:
        to_add = torch.cat(chunks_to_add).numpy()
        index.add(to_add)
        logging.info(f'Adding Vectors {added} -> {added + to_add.shape[0]} of {N}')

    logger.info(f'Index Built, writing index to {output_path}')
    faiss.write_index(index, output_path)
    logger.info(f'Index dumped')
    return index


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Build a FAISS index from precomputed vector files from embed.py. "
                                     "Provides functionality to build either flat indexes (slow but exact)"
                                     " or HNSW indexes (much faster, but approximate). "
                                     "Optional application of 8bit or 16bit quantization is also available."
                                     " Many more indexes are possible with Faiss, consult the Faiss repository here"
                                     " if you want to build more advanced indexes.")
    parser.add_argument('--embeddings_dir', type=str, help='path to directory containing vectors to build index from')
    parser.add_argument('--output_path', type=str, help='path to write results to')
    parser.add_argument('--hnsw', action='store_true', help='Build an HNSW index rather than Flat')
    parser.add_argument('--SQ8', action='store_true', help='use SQ8 quantization on index to save memory')
    parser.add_argument('--fp16', action='store_true', help='use fp16 quantization on index to save memory')
    parser.add_argument('--store_n', type=int, default=32, help='hnsw store_n parameter')
    parser.add_argument('--ef_construction', type=int, default=128, help='hnsw ef_construction parameter')
    parser.add_argument('--ef_search', type=int, default=128, help='hnsw ef_search parameter')
    parser.add_argument('--sample_fraction', type=float, default=1.0,
                        help='If memory is limited, specify a fraction (0.0->1.0) of the '
                             'data to sample for training the quantizer')
    parser.add_argument('--indexing_batch_size', type=int, default=None,
                        help='If memory is limited, specify the approximate number '
                             'of vectors to add to the index at once')
    parser.add_argument('-v', '--verbose', action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    assert not (args.SQ8 and args.fp16), 'cant use both sq8 and fp16 Quantization'
    assert not os.path.exists(args.output_path), "Faiss index with name specificed in --output_path already exists"

    args.indexing_batch_size = 10000000000000 if args.indexing_batch_size is None else args.indexing_batch_size
    assert 0 < args.sample_fraction <= 1.0

    if args.sample_fraction:
        build_index_streaming(
            args.embeddings_dir,
            args.output_path,
            args.hnsw,
            sq8_quantization=args.SQ8,
            fp16_quantization=args.fp16,
            store_n=args.store_n,
            ef_construction=args.ef_construction,
            ef_search=args.ef_search,
            sample_fraction=args.sample_fraction,
            indexing_batch_size=args.indexing_batch_size,
        )
