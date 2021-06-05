#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import torch
import logging
import os
import time
from paq.paq_utils import is_apex_available, load_jsonl, dump_jsonl, get_submitit_executor, to_fp16
from transformers import AutoConfig, AutoTokenizer, AutoModelForMultipleChoice
if is_apex_available():
    import apex
    from apex import amp
    apex.amp.register_half_function(torch, "einsum")

logger = logging.getLogger(__name__)
CUDA = torch.cuda.is_available()


def load_reranker(model_name_or_path):
    logger.info(f'Loading model from: {model_name_or_path}')
    config = AutoConfig.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, do_lower_case=True)
    model = AutoModelForMultipleChoice.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=config,
    )
    model = model.eval()
    return model, tokenizer


def get_output_format(qas, prediction_indices, prediction_scores):
    assert len(qas) == len(prediction_indices)
    return [
        {
             'question': q['input_qa']['question'],
             'prediction': q['retrieved_qas'][p]['answer'][0],
             'score':s, 'index': int(p)
        }
        for q, p, s in zip(qas, prediction_indices, prediction_scores)
    ]


def tokenize(tokenizer, batch_qas, cuda, top_k):
    input_as, input_bs = [], []

    for item in batch_qas:
        question_a = item['input_qa']['question'] + '?'
        question_bs = [q['question'] + '? ' + q['answer'][0] for q in item['retrieved_qas']]
        question_bs = question_bs[:top_k]
        input_as += [question_a for _ in range(len(question_bs))]
        input_bs += question_bs

    inputs = tokenizer.batch_encode_plus(
        list(zip(input_as, input_bs)), return_tensors='pt', padding='longest', add_special_tokens=True
    )
    inputs = {k: v.reshape(len(batch_qas), v.shape[0]//len(batch_qas), -1) for k,v in inputs.items()}
    return {k: v.cuda() for k, v in inputs.items()} if cuda else inputs


def predict(model, tokenizer, qas, cuda=CUDA, bsz=16, fp16=False, top_k=30):

    if cuda:
        model = model.cuda()
        model = to_fp16(model) if fp16 else model

    t = time.time()

    def log_progress(j, outputs):
        t2 = time.time()
        logger.info(
            f'Reranked {j + 1} / {len(list(range(0, len(qas), bsz)))} batches in {t2 - t:0.2f} seconds '
            f'({len(outputs) / (t2 - t): 0.4f} QAs per second)')

    def forward(inputs):
        logits = model(**inputs)[0]
        scores, inds = logits.topk(1, dim=1)
        scores, inds = scores.squeeze().tolist(), inds.squeeze().tolist()
        if padded_batch:
            scores, inds = scores[:-1], inds[:-1]
        return scores, inds

    outputs = []
    output_scores = []
    logger.info(f'Embedding {len(qas)} inputs in {len(list(range(0, len(qas), bsz)))} batches:')
    with torch.no_grad():
        for j, batch_start in enumerate(range(0, len(qas), bsz)):

            batch = qas[batch_start: batch_start + bsz]
            padded_batch = len(batch) == 1
            if padded_batch: # hack for batch size 1 issues
                batch = [batch[0],batch[0]]

            inputs = tokenize(tokenizer, batch, cuda, top_k)
            scores, inds = forward(inputs)

            outputs.extend(inds)
            output_scores.extend(scores)

            log_progress(j, outputs) if j % 1 == 0 else None

    log_progress(j, outputs)

    return get_output_format(qas, outputs, output_scores)


def run_predictions(qas_to_rerank_file, output_file, model_name_or_path, batch_size, fp16, top_k):
    qas_to_rerank = load_jsonl(qas_to_rerank_file)
    reranker_model, reranker_tokenizer = load_reranker(model_name_or_path)

    predictions = predict(
        reranker_model,
        reranker_tokenizer,
        qas_to_rerank,
        bsz=batch_size,
        fp16=fp16,
        top_k=top_k
    )
    dump_jsonl(predictions, output_file)


def parse_files(args):
    infis, outfis = args.qas_to_rerank.split(','), args.output_files.split(',')
    assert len(infis) == len(outfis)
    pairs = []
    for in_fi, out_fi in zip(infis, outfis):
        if os.path.exists(out_fi):
            logging.info(f'skipping inference on {out_fi}, file exists')
        pairs.append((in_fi, out_fi))
    return pairs


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Perform RePAQ Reranking. This program will rerank retrieval results from retrieve.py.")
    parser.add_argument('--model_name_or_path', type=str,)
    parser.add_argument('--qas_to_rerank', type=str, help='comma separated list of files produced by retrieve.py to rerank')
    parser.add_argument('--output_files', type=str, help='comma separated list of filenames to write, one for each filenmae in --qas_to_rerank')
    parser.add_argument('--top_k', type=int, default=50, help='top k to rerank')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--n_jobs', type=int, required=True, help='how many parallel jobs to use in slurm (n_jobs=-1 will run locally)')
    parser.add_argument('--slurm_partition', type=str, default="learnfair", help='If using submitit to run slurm jobs, define cluster partition here')
    parser.add_argument('--slurm_comment', type=str, default="", help='If using submitit to run slurm jobs, define job comment here')
    parser.add_argument('-v', '--verbose', action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    pairs = parse_files(args)

    if args.n_jobs != -1:
        executor = get_submitit_executor(n_jobs=args.n_jobs, comment=args.slurm_comment, partition=args.slurm_partition)
        with executor.batch():
            jobs = [
                executor.submit(run_predictions, infi, outfi, args.model_name_or_path, args.batch_size, args.fp16, args.top_k)
                for infi, outfi in pairs
            ]
        logger.info('launched the following jobs:')
        [logger.info(job.job_id) for job in jobs]
    else:
        for infi, outfi in pairs:
            run_predictions(infi, outfi, args.model_name_or_path, args.batch_size, args.fp16, args.top_k)
