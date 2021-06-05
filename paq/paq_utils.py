#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
import logging
import torch
import glob
import os
import csv
try:
    import submitit
    _has_submitit = True
except ImportError:
    _has_submitit = False
try:
    import apex
    from apex import amp
    apex.amp.register_half_function(torch, "einsum")
    _has_apex = True
except ImportError:
    _has_apex = False
try:
    import spacy
    from spacy.util import minibatch, compounding
    spacy.prefer_gpu()

    _has_spacy = True
except (ImportError, AttributeError):
    _has_spacy = False


logger = logging.getLogger(__name__)


def is_spacy_available():
    return _has_spacy


def is_submitit_available():
    return _has_submitit


def is_apex_available():
    return _has_apex


def to_fp16(model):
    if is_apex_available():
        model = amp.initialize(model, opt_level="O1")
    else:
        model = model.half()
    return model


def load_jsonl_memory_friendly(fi):
    logging.info(f'Loading {fi}')

    results = []
    for ln, line in enumerate(open(fi)):
        results.append(json.loads(line))
        logging.info(f'Loaded {ln + 1} Items from {fi}') if ln % 1000000 == 0 else None

    logging.info(f'Loaded {ln + 1} Items from {fi}')
    return results


def load_jsonl_fast(fi):
    logging.info(f'Loading {fi}')

    results = []
    with open(fi) as f:
        txt = f.read()
        logging.info(f'{fi} Loaded, splitting into lines...')
        lines = [t for t in txt.split('\n') if t.strip()!='']
        logging.info(f'Parsing {len(lines)} items from jsonl:')

    for ln, line in enumerate(lines):
        results.append(json.loads(line))
        logging.info(f'Loaded {ln + 1} Items from {fi}') if ln % 1000000 == 0 else None

    logging.info(f'Loaded {ln + 1} Items from {fi}')
    return results


def load_jsonl(fi, memory_friendly=False):
    if memory_friendly:
        return load_jsonl_memory_friendly(fi)
    else:
        return load_jsonl_fast(fi)


def dump_jsonl(items, fi):
    logging.info(f'Dumping {len(items)} items into {fi}')
    k = 0

    with open(fi, 'w') as f:
        for k, item in enumerate(items):
            f.write(json.dumps(item) + '\n')
            logging.info(f'Written {k + 1} / {len(items)} items') if k % 10000 == 0 else None

    logging.info(f'Written {k + 1} / {len(items)} items')


def load_dpr_tsv(fi):
    items = []
    with open(fi) as ifile:
        reader = csv.reader(ifile, delimiter='\t')
        for spl in reader:
            idd, text, title = spl
            items.append({'passage_id': idd, "passage": text, "metadata": {'title': title}})
    return items


def get_vectors_file_paths_in_vector_directory(embeddings_dir):
    paths = glob.glob(os.path.abspath(embeddings_dir) + '/*')
    np = len(paths)
    template = '.'.join(paths[0].split('.')[:-1])
    return [template + f'.{j}' for j in range(np)]


def parse_vectors_from_directory_chunks(embeddings_dir, half):
    paths = get_vectors_file_paths_in_vector_directory(embeddings_dir)
    for j, p in enumerate(paths):
        logger.info(f'Loading vectors from {p} ({j+1} / {len(paths)})')
        m = torch.load(p)
        assert int(p.split('.')[-1]) == j, (p, j)

        if half:
            m = m if m.dtype == torch.float16 else m.half()
        else:
            m = m if m.dtype == torch.float32 else m.float()
        yield m


def parse_vectors_from_directory_fast(embeddings_dir):
    ms = []
    for m in parse_vectors_from_directory_chunks(embeddings_dir):
        ms.append(m)

    out = torch.cat(ms)
    logger.info(f'loaded index of shape {out.shape}')
    return out


def parse_vectors_from_directory_memory_friendly(embeddings_dir, size=None):
    paths = get_vectors_file_paths_in_vector_directory(embeddings_dir)
    if size is None:
        size = 0
        for j, p in enumerate(paths):
            logger.info(f'Loading vectors from {p} ({j+1} / {len(paths)}) to find total num vectors')
            m = torch.load(p)
            size += m.shape[0]

    out = None
    offset = 0
    for j, p in enumerate(paths):
        logger.info(f'Loading vectors from {p} ({j+1} / {len(paths)})')
        m = torch.load(p)

        assert int(p.split('.')[-1]) == j, (p, j)
        if out is None:
            out = torch.zeros(size, m.shape[1])
        out[offset: offset + m.shape[0]] = m
        offset += m.shape[0]
    assert offset == size
    logger.info(f'loaded index of shape {out.shape}')

    return out


def parse_vectors_from_directory(fi, memory_friendly=False, size=None, as_chunks=False, half=False):
    assert os.path.isdir(fi), f"Vectors directory {fi} doesnt exist, or is not a directory of pytorch vectors"
    if as_chunks:
        return parse_vectors_from_directory_chunks(fi, half)

    if memory_friendly:
        out = parse_vectors_from_directory_memory_friendly(fi, size=size)
    else:
        out = parse_vectors_from_directory_fast(fi)

    if half:
        out = out if out.dtype == torch.float16 else out.half()
    else:
        out = out if out.dtype == torch.float32 else out.float()

    return out


def get_submitit_executor(n_jobs=10, comment="", partition='learnfair'):
    if not is_submitit_available():
        raise Exception('Submitit Not installed')
    executor = submitit.AutoExecutor(folder='PAQ_embedding_jobs')
    executor.update_parameters(timeout_min=120,
                               slurm_partition=partition,
                               slurm_nodes=1,
                               slurm_ntasks_per_node=1,
                               slurm_cpus_per_task=10,
                               slurm_constraint='volta32gb',
                               slurm_gpus_per_node='volta:1',
                               slurm_array_parallelism=n_jobs,
                               slurm_comment=comment,
                               slurm_mem='64G')
    return executor

