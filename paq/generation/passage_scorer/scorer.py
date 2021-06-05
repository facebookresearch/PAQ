#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from typing import List, Dict, Union
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
import torch


class DummyPassageScorer:
    """
    Dummy scorer that will always return the same score for any passage.
    """
    name = "passage_scorer/dummy"

    def __init__(self, default_score=0.0):
        self.default_score = default_score

    def score_passage(self, passage: Dict) -> float:
        return self.default_score

    def score_passages(self, passages_to_label, disable_tqdm=False):
        for passage in tqdm(passages_to_label, disable=disable_tqdm):
            score = self.score_passage(passage)
            passage['metadata']['ps_score'] = score
        return passages_to_label


class LookupPassageScorer:
    """
    Lookup scorer that will return the score from a file of precomputed passage scores for passages, or if not present, return a default score.
    """
    name = "passage_scorer/lookup"

    def __init__(self, scores_file, default_score=-10000.0):
        self._load_passage_scores(scores_file)
        self.default_score = default_score

    def _load_passage_scores(self, scores_file):
        self.passage_scores = {}
        for line in open(scores_file):
            k, v = line.strip('\n').split('\t')
            self.passage_scores[k] = v

    def score_passage(self, passage: Dict) -> float:
        return self.passage_scores.get(passage['passage_id'], self.default_score)

    def score_passages(self, passages_to_label, disable_tqdm=False):
        for passage in tqdm(passages_to_label, disable=disable_tqdm):
            score = self.score_passage(passage)
            passage['metadata']['ps_score'] = score
        return passages_to_label


class LearntPassageScorer:
    """Learnt scorer"""
    name = "passage_scorer/learnt"

    def __init__(self,
                 model_path: str,
                 config_path: str,
                 tokenizer_path: str = None,
                 batch_size: int = 10,
                 device: int = 0,
                 max_seq_len: int = 256):
        self.device = torch.device(f"cuda:{device}") if device is not None else torch.device("cpu")

        config = AutoConfig.from_pretrained(config_path if config_path is not None else model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, config=config)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config)

        self.model.to(self.device)
        self.model.eval()
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

    def _tokenize(self, texts):
        input_features = self.tokenizer.batch_encode_plus(
            texts, return_tensors='pt', padding=True, add_special_tokens=True, max_length=256, truncation=True
        )
        input_features = {k: v.to(self.device) for k, v in input_features.items()}
        return input_features

    def score_passages(self, passages_to_label, disable_tqdm=False):

        def _run_batch(batch):
            inputs = self._tokenize([b['passage'] for b in batch])
            scores = self.model(**inputs)
            log_probs = torch.log_softmax(scores.logits, dim=-1)[:, 1].cpu().tolist()
            for s, b in zip(log_probs, batch):
                b['metadata']['ps_score'] = float(s)
            return scores

        batch, outputs = [], []
        for passage in tqdm(passages_to_label, disable=disable_tqdm):
            batch.append(passage)

            if len(batch) == self.batch_size:
                _run_batch(batch)
                outputs += batch
                batch = []

        if len(batch) != 0:
            _run_batch(batch)
            outputs += batch

        return outputs


def load_passage_scorer(config):
    PASSAGE_SCORER_MAP = {m.name: m for m in [LearntPassageScorer, DummyPassageScorer, LookupPassageScorer]}
    answer_extractor = PASSAGE_SCORER_MAP[config['name']](**config['config'])
    return answer_extractor
