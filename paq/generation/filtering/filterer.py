#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import sys
import faiss
import os
# hack to add FID to path:
file_path = os.path.realpath(__file__)
fid_path = os.path.join(os.path.dirname(file_path), '../../../FiD')
sys.path.append(fid_path)
import torch
import transformers
import numpy as np
from torch.utils.data import DataLoader, SequentialSampler
from paq.paq_utils import load_dpr_tsv, load_jsonl
from transformers import AutoModel, AutoConfig, AutoTokenizer
import src.util
import src.data
import src.evaluation
import src.model
import logging
from paq.retrievers.embed import embed
from paq.retrievers.retrieve import mips
import pickle
from torch import nn
logger = logging.getLogger(__name__)


def _load_corpus(path):
    if 'tsv' in path or 'csv' in path:
        docs = load_dpr_tsv(path)
    else:
        docs = load_jsonl(path)
    logger.info('Parsed Corpus for retrieval')
    return {d['passage_id']: {'title': d['metadata']['title'], 'text': d['passage']} for d in docs}


class DummyFilteringRetriever:
    """Dummy filterer - does not retrieve any evidence"""
    name = "filtering/dummy_filtering_retriever"

    def retrieve_documents(self, data):
        return [{'question': d['question'], 'answers': [d['answer']], 'ctxs': [], 'metadata': d} for d in data]


class LocalFilteringRetriever:
    """Retrieves a single document (the gold context the question was generated from"""
    name = "filtering/local_filtering_retriever"
    corpus = None

    def __init__(self, corpus_path):
        self.corpus_path = corpus_path

    def retrieve_documents(self, data):
        self.corpus = _load_corpus(self.corpus_path) if self.corpus is None else self.corpus

        examples = []
        for d in data:
            assert d['passage_id'] in self.corpus
            gold_doc = self.corpus[d['passage_id']]
            examples.append(
                {'question': d['question'].strip(), 'answers': [d['answer']], 'ctxs': [gold_doc], 'metadata': d}
            )
        return examples


class DPRQuestionEncoder(nn.Module):
    """simple wrapper on DPR Question Encoder Bert model"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        seq_outputs = self.model(*args, **kwargs)['last_hidden_state']
        return seq_outputs[:, 0]


class GlobalFilteringRetriever:
    """Uses DPR to retrieve relevant documents for the question"""
    name = "filtering/global_filtering_retriever"
    corpus = None
    index_id_to_db_id = None
    index = None

    def __init__(self,
                 corpus_path,
                 index_path,
                 index_id_to_db_id_path,
                 model_path,
                 batch_size,
                 n_queries_to_parallelize,
                 max_seq_len,
                 n_docs,
                 device
                 ):
        self.corpus_path = corpus_path
        self.index_path = index_path
        self.index_id_to_db_id_path = index_id_to_db_id_path
        self.n_docs = n_docs
        self.device = torch.device(f"cuda:{device}") if device is not None else torch.device("cpu")

        config = AutoConfig.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, config=config)
        self.model = DPRQuestionEncoder(AutoModel.from_pretrained(model_path, config=config))

        self.model.to(self.device)
        self.model.eval()
        self.batch_size = batch_size
        self.n_queries_to_parallelize = n_queries_to_parallelize
        self.max_seq_len = max_seq_len

    def _load_corpus(self):
        logger.info("Loading Corpus if not already loaded...")
        self.corpus = _load_corpus(self.corpus_path) if self.corpus is None else self.corpus
        logger.info("Loading Faiss index if not already loaded...")
        self.index = faiss.read_index(self.index_path) if self.index is None else self.index
        if self.index_id_to_db_id is None:
            with open(self.index_id_to_db_id_path, 'rb') as f:
                self.index_id_to_db_id = pickle.load(f)

    def retrieve_documents(self, qa_pairs):
        self._load_corpus()
        examples = []
        for ci in range(0, len(qa_pairs), self.n_queries_to_parallelize):
            chunk_examples = qa_pairs[ci: ci + self.n_queries_to_parallelize]
            queries = embed(self.model, self.tokenizer, chunk_examples, bsz=self.batch_size)
            top_indices, _ = mips(self.index, queries, self.n_docs, self.n_queries_to_parallelize)
            for ati, d in zip(top_indices, chunk_examples):
                ctxs = [self.corpus[self.index_id_to_db_id[ati[j]]] for j in range(self.n_docs)]
                examples.append({'question': d['question'], 'answers': [d['answer']], 'ctxs': ctxs, 'metadata': d})
        return examples


class CompatableEncoderWrapper(torch.nn.Module):
    """Patched version of fid.model.EncoderWrapper to make it compatable with our version of transformers"""

    def __init__(self, encoder, use_checkpoint=False):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids=None, attention_mask=None, **kwargs, ):
        # total_length = n_passages * passage_length
        bsz, total_length = input_ids.shape
        passage_length = total_length // self.n_passages
        input_ids = input_ids.view(bsz * self.n_passages, passage_length)
        attention_mask = attention_mask.view(bsz * self.n_passages, passage_length)
        outputs = self.encoder(input_ids, attention_mask, **kwargs)
        outputs.last_hidden_state = outputs.last_hidden_state.view(bsz, self.n_passages * passage_length, -1)
        return outputs


class FIDReader:
    """FID Filterer"""
    name = "filtering/fid_reader"

    def __init__(self,
                 model_path: str,
                 batch_size: int = 10,
                 device: int = 0,
                 max_seq_len: int = 200,
                 n_docs:int = 50,
                 ):
        self.device = torch.device(f"cuda:{device}") if device is not None else torch.device("cpu")
        self.tokenizer = transformers.T5Tokenizer.from_pretrained('t5-base', return_dict=False)
        self.model = src.model.FiDT5.from_pretrained(model_path)

        self.model.to(self.device)
        self.model.eval()
        self.model.encoder = CompatableEncoderWrapper(self.model.encoder.encoder) # hack to make FID compatable with newer transformers version
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.n_docs = n_docs
        self.collator = src.data.Collator(self.max_seq_len, self.tokenizer)

    def _get_dataloader_for_examples(self, examples):
        for k, example in enumerate(examples):
            example['id'] = k
            for c in example['ctxs']:
                c['score'] = 1.0 / (k + 1)

        eval_dataset = src.data.Dataset(examples, self.n_docs)

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.batch_size,
            num_workers=20,
            collate_fn=self.collator
        )
        return eval_dataset, eval_dataloader

    def generate_answers(self, examples):

        eval_dataset, eval_dataloader = self._get_dataloader_for_examples(examples)
        total = 0
        exactmatch = []
        with torch.no_grad():
            for i, batch in enumerate(eval_dataloader):
                (idx, _, _, context_ids, context_mask) = batch

                outputs = self.model.generate(
                    input_ids=context_ids.to(self.device),
                    attention_mask=context_mask.to(self.device),
                    max_length=10,
                )
                for k, o in enumerate(outputs):
                    ans = self.tokenizer.decode(o, skip_special_tokens=True)
                    example = eval_dataset.data[idx[k]]
                    score = src.evaluation.ems(ans, example['answers'])
                    exactmatch.append(score)
                    example['consistent'] = score
                    example['filter_answer'] = ans
                    total += 1

                if (i + 1) % 10 == 0:
                    logger.info(f'FID filtering: {i+1} / {len(eval_dataloader)} | ave = {np.mean(exactmatch):.3f}')
        logger.info(f'FID filtering: {i+1} / {len(eval_dataloader)} | ave = {np.mean(exactmatch):.3f}')
        output = _get_reader_output_format(examples)
        return output


class DummyReader:
    """Dummy Reader, always returns consistent"""
    name = "filtering/dummy_reader"

    def generate_answers(self, examples):
        for example in examples:
            example['consistent'] = True
            example['filter_answer'] = "DUMMY_READER_ANSWER"
        output = _get_reader_output_format(examples)
        return output


def _get_reader_output_format(dataset):
    out = []
    for e in dataset:
        o = e['metadata']
        o['metadata'] = o.get('metadata', {})
        o['metadata']['consistent'] = e['consistent']
        o['metadata']['filter_answer'] = e['filter_answer']
        out.append(o)
    return out


def load_reader(config):
    READER_MAP = {m.name: m for m in [DummyReader, FIDReader]}
    reader = READER_MAP[config['name']](**config['config'])
    return reader


def load_retriever(config):
    RETRIEVER_MAP = {m.name: m for m in [LocalFilteringRetriever, GlobalFilteringRetriever, DummyFilteringRetriever]}
    retriever = RETRIEVER_MAP[config['name']](**config['config'])
    return retriever

