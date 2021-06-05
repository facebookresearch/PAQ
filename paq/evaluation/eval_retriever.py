#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import argparse
from paq.evaluation.eval_utils import metric_max_over_ground_truths, exact_match_score
from paq.paq_utils import load_jsonl


def eval_retriever(refs, preds, hits_at_k):
    for k in hits_at_k:
        scores = []
        dont_print = False
        for r, p in zip(refs, preds):
            if hits_at_k[-1] > len(p['retrieved_qas']):
                print(f'Skipping hits@{K} eval as {K} is larger than number of retrieved results')
                dont_print = True
            ref_answers = r['answer']
            em = any([
                metric_max_over_ground_truths(exact_match_score, pred_answer['answer'][0], ref_answers)
                for pred_answer in p['retrieved_qas'][:k]
            ])
            scores.append(em)

        if not dont_print:
            print(f'{k}: {100 * sum(scores) / len(scores):0.1f}% \n({sum(scores)} / {len(scores)})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=str, help="path to retrieval results to eval, in PAQ's retrieved results jsonl format")
    parser.add_argument('--references', type=str, help="path to gold answers, in jsonl format")
    parser.add_argument('--hits_at_k', type=str, help='comma separated list of K to eval hits@k for', default="1,10,50")
    args = parser.parse_args()

    refs = load_jsonl(args.references)
    preds = load_jsonl(args.predictions)
    assert len(refs) == len(preds), "number of references doesnt match number of predictions"

    hits_at_k = sorted([int(k) for k in args.hits_at_k.split(',')])
    eval_retriever(refs, preds, hits_at_k)
