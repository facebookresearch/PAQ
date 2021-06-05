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
import math
import os
from paq.paq_utils import is_apex_available, load_jsonl, get_submitit_executor, to_fp16
from paq.retrievers.retriever_utils import load_retriever


logger = logging.getLogger(__name__)
CUDA = torch.cuda.is_available()


def embed(model, tokenizer, qas, bsz=256, cuda=CUDA, fp16=False):

    def normalize_q(question: str) -> str:
        return question.strip().strip('?').lower().strip()

    def tokenize(batch_qas):
        input_qs = [normalize_q(q['question']) for q in batch_qas]
        inputs = tokenizer.batch_encode_plus(
            input_qs, return_tensors='pt', padding=True, add_special_tokens=True
        )
        return {k: v.cuda() for k, v in inputs.items()} if cuda else inputs

    if cuda:
        model = model.cuda()
        model = to_fp16(model) if fp16 else model

    t = time.time()

    def log_progress(j, outputs):
        t2 = time.time()
        logger.info(
            f'Embedded {j + 1} / {len(list(range(0, len(qas), bsz)))} batches in {t2 - t:0.2f} seconds '
            f'({sum([len(o) for o in outputs]) / (t2 - t): 0.4f} QAs per second)')

    outputs = []
    with torch.no_grad():
        for j, batch_start in enumerate(range(0, len(qas), bsz)):
            batch_qas = qas[batch_start: batch_start + bsz]
            inputs = tokenize(batch_qas)
            batch_outputs = model(**inputs)
            outputs.append(batch_outputs.cpu())
            if j % 10 == 0:
                log_progress(j, outputs)

    log_progress(j, outputs)

    return torch.cat(outputs, dim=0).cpu()


def embed_job(qas_to_embed_path, model_name_or_path, output_file_name, n_jobs, job_num, batch_size, fp16, memory_friendly_parsing):
    os.makedirs(os.path.dirname(output_file_name), exist_ok=True)

    qas_to_embed = load_jsonl(qas_to_embed_path, memory_friendly=memory_friendly_parsing)
    chunk_size = math.ceil(len(qas_to_embed) / n_jobs)

    qas_to_embed_this_job = qas_to_embed[job_num * chunk_size: (job_num + 1) * chunk_size]
    logger.info(f'Embedding Job {job_num}: Embedding {len(qas_to_embed)} inputs in {int(len(qas_to_embed) / batch_size)} batches:')

    model, tokenizer = load_retriever(model_name_or_path)
    mat = embed(model, tokenizer, qas_to_embed_this_job, bsz=batch_size, fp16=fp16)
    torch.save(mat.half(), output_file_name + f'.{job_num}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True, help='path to HF model dir')
    parser.add_argument('--qas_to_embed', type=str,required=True, help='Path to questions to embed in jsonl format')
    parser.add_argument('--n_jobs', type=int, required=True, help='how many jobs to embed with (n_jobs=-1 will run a single job locally)')
    parser.add_argument('--output_dir', type=str, help='path to write vectors to')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--memory_friendly_parsing', action='store_true', help='Pass this to load jsonl files more slowly, but save memory')
    parser.add_argument('--slurm_partition', type=str, default="learnfair", help='If using submitit to run slurm jobs, define cluster partition here')
    parser.add_argument('--slurm_comment', type=str, default="", help='If using submitit to run slurm jobs, define job comment heree')
    parser.add_argument('-v', '--verbose', action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    if args.fp16 and not CUDA:
        raise Exception('Cant use --fp16 without a gpu, CUDA not found')

    output_path = os.path.join(args.output_dir, 'embeddings.pt')

    if args.n_jobs == -1:
        embed_job(
            args.qas_to_embed,
            args.model_name_or_path,
            output_path,
            n_jobs=1,
            job_num=0,
            batch_size=args.batch_size,
            fp16=args.fp16,
            memory_friendly_parsing=args.memory_friendly_parsing
        )
    else:
        executor = get_submitit_executor(n_jobs=args.n_jobs, comment=args.slurm_comment, partition=args.slurm_partition)
        jobs = []
        with executor.batch():
            for jn in range(args.n_jobs):
                job = executor.submit(
                    embed_job,
                    args.qas_to_embed,
                    args.model_name_or_path,
                    output_path,
                    args.n_jobs,
                    jn,
                    args.batch_size,
                    args.fp16,
                    args.memory_friendly_parsing
                )
                jobs.append(job)

        logger.info('launched the following jobs:')
        for job in jobs:
            logger.info(job.job_id)
