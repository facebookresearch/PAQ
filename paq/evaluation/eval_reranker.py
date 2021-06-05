#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import argparse
from paq.evaluation.eval_utils import metric_max_over_ground_truths, exact_match_score
from paq.paq_utils import load_jsonl


def evaluate_exact_match(preds, refs):
    assert len(refs) == len(preds)

    scores = []
    for ref, pred in zip(refs, preds):
        score = metric_max_over_ground_truths(exact_match_score, pred, ref)
        scores.append(score)

    return sum(scores) / len(scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=str, help="path to predicted answers in jsonl format {'question': question, 'prediciton': predicted answer}")
    parser.add_argument('--references', type=str, help="path to gold answers, in jsonl format")
    args = parser.parse_args()

    refs = load_jsonl(args.references)
    preds = load_jsonl(args.predictions)
    assert len(refs) == len(preds), "number of references doesnt match number of predictions"

    assert len(refs) == len(preds)
    scores = []
    for r, p in zip(refs, preds):
        ref_answers = r['answer']
        pred_answer = p['prediction']
        score = metric_max_over_ground_truths(exact_match_score, pred_answer, ref_answers)
        scores.append(score)

    print(f'{100 * sum(scores) / len(scores):0.1f}% \n({sum(scores)} / {len(scores)})')
