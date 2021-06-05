#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging
import numpy as np
from typing import List, Dict
import torch
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoTokenizer
from paq.paq_utils import is_spacy_available
from paq.generation.answer_extractor.span2D_model import AnswerSpanExtractor2DModel, postprocess_span2d_output


def get_output_format(all_passages, all_answers):
    all_results = []
    assert len(all_passages) == len(all_answers)
    for passage, answers in zip(all_passages, all_answers):
        result = {
            "passage_id": passage["passage_id"],
            "passage": passage["passage"],
            "answers": answers,
            "metadata": passage["metadata"],
        }
        all_results.append(result)
    return all_results


class SpacyNERExtractor:
    """
    Spacy NER extractor
    """
    name = "answer_extractor/spacy_ner"

    def __init__(self, model="en_core_web_sm"):
        assert is_spacy_available(), "Spacy is not installed. Please install with `pip install spacy`."
        import spacy
        self.nlp = spacy.load(model)

    def extract_from_passage(self, passage: str) -> List[Dict]:
        doc = self.nlp(passage)
        answers = []
        for ent in doc.ents:
            answers.append({
                "text": ent.text,
                "start": ent.start_char,
                "end": ent.end_char,
                "score": None
            })
        return answers

    def extract_answers_from_passages(self, passages_to_label, disable_tqdm=False):
        all_answers = []
        for doc in tqdm(self.nlp.pipe([p['passage'] for p in passages_to_label], batch_size=128), disable=disable_tqdm):
            answers = []
            for ent in doc.ents:
                answers.append({
                    "text": ent.text,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "score": None
                })
            all_answers.append(answers)

        # Post-process
        all_results = get_output_format(passages_to_label, all_answers)
        return all_results


class Span2DAnswerExtractor:
    """
    Predict answer spans with their joint span probability P(start, end|context).
    """
    name = "answer_extractor/span2D"

    def __init__(
        self,
        model_path: str,
        config_path: str = None,
        tokenizer_path: str = None,
        topk: int = 5,
        max_answer_len: int = 30,
        max_seq_len: int = 256,
        doc_stride: int = 128,
        batch_size: int = 10,
        device: int = 0,
        **kwargs
    ):
        assert model_path is not None
        self.device = torch.device(f"cuda:{device}") if device is not None else torch.device("cpu")

        config = AutoConfig.from_pretrained(config_path if config_path is not None else model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path if tokenizer_path is not None else model_path)
        self.model = AnswerSpanExtractor2DModel.from_pretrained(model_path, config=config)

        self.model.to(self.device)
        self.model.eval()

        self.topk = topk
        self.max_answer_len = max_answer_len
        self.max_seq_len = max_seq_len
        self.doc_stride = doc_stride
        logging.info(f"Extract top {self.topk} answer spans with "
                     f"max_answer_len={self.max_answer_len}, max_seq_len={self.max_seq_len}, "
                     f"doc_stride={self.doc_stride}.")

        self.kwargs = kwargs
        self.batch_size = batch_size

    def _tokenize(self, passage: str):
        input_features = self.tokenizer(
            passage,
            truncation=True,
            max_length=self.max_seq_len,
            stride=self.doc_stride,
            return_overflowing_tokens = True,
            return_offsets_mapping = True,
            padding="max_length",
        )
        input_features["input_ids"] = torch.tensor(input_features["input_ids"]).to(self.device)
        input_features["token_type_ids"] = torch.tensor(input_features["token_type_ids"]).to(self.device)
        input_features["attention_mask"] = torch.tensor(input_features["attention_mask"]).to(self.device)
        return input_features

    def extract_from_passage(self, passage: str):
        input_features = self._tokenize(passage)
        model_output = self.model(**input_features, return_dict=True)
        answers = postprocess_span2d_output(model_output, input_features, self.max_answer_len, passage, self.topk)
        for answer in answers:
            answer['score'] = np.log(answer['score'])
        return answers

    def extract_answers_from_passages(self, passages_to_label, disable_tqdm=False):

        # Run the pipeline (model) to extract the answer spans
        all_answers = []
        for passage in tqdm(passages_to_label, disable=disable_tqdm):
            answers = self.extract_from_passage(passage["passage"])
            all_answers.append(answers)

        # Post-process
        all_results = get_output_format(passages_to_label, all_answers)
        return all_results


def load_answer_extractor(config):
    ANS_EXT_MAP = {m.name: m for m in [SpacyNERExtractor, Span2DAnswerExtractor]}
    answer_extractor = ANS_EXT_MAP[config['name']](**config['config'])
    return answer_extractor
