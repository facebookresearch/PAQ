#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging
from typing import List, Union, Set
from tqdm.auto import tqdm
import warnings
import torch

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.pipelines import Text2TextGenerationPipeline
from paq.paq_utils import to_fp16


logger = logging.getLogger(__name__)


def _batch_iterator(context_answer_pairs,
                    batch_size,
                    include_title: bool = True,
                    ):

    def _answer_context_pair_2_text(answer, context):
        answer_start, answer_end, answer_text = answer["start"], answer['end'], answer['text']
        return context[:answer_start] + "** " + context[answer_start:answer_end] + " **" + context[answer_end:]

    def _create_input_text(context, answer, title=None) -> str:
        text = _answer_context_pair_2_text(answer, context)

        if title is not None:
            output = f"answer: {answer['text']} | title: {title} | context: {text}"
        else:
            output = f"answer: {answer['text']} | context: {text}"
        return output

    iter_batch = []
    for context_answer_pair in context_answer_pairs:

        passage_id = context_answer_pair["passage_id"]
        context = context_answer_pair["passage"]
        answers = context_answer_pair["answers"]
        title = context_answer_pair["metadata"]["title"] if include_title else None

        for answer in answers:
            input_text = _create_input_text(context, answer, title)
            iter_batch.append((passage_id, answer, input_text))

            if len(iter_batch) >= batch_size:
                yield iter_batch
                iter_batch = []

    if len(iter_batch) > 0:
        yield iter_batch


class QuestionGenerator:
    name = "question_generator/standard"

    def __init__(
        self,
        model_path: str,
        config_path: str = None,
        tokenizer_path: str = None,
        include_title: bool = True,
        num_beams: int = None,
        num_return_sequences: int = 1,
        max_question_len: int = 30,
        batch_size: int = 1,
        device: int = 0,
        **kwargs
    ):
        assert model_path is not None

        super().__init__()

        config = AutoConfig.from_pretrained(config_path if config_path is not None else model_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path if tokenizer_path is not None else model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, config=config)

        if kwargs.get('fp16', False):
            model = model.cuda()
            model = to_fp16(model)

        self.pipeline = Text2TextGenerationPipeline(model=model, tokenizer=tokenizer, task="question-generation",
                                                    device=device)

        self.include_title = include_title  # include title in the source sequence
        self.num_beams = num_beams
        self.num_return_sequences = num_return_sequences
        self.max_question_len = max_question_len
        logger.info(
            f"Generate {self.num_return_sequences} questions for each passage with beam size {self.num_beams}.")

        self.batch_size = batch_size

        self.kwargs = kwargs

    def generate_question(self, data: Union[str, List[str]]):
        """
        Generate question for a single input sequence or a batch of input sequences.
        """
        if isinstance(data, str):
            data = [data]

        all_records = self.pipeline(
            data,
            return_text=True,
            # return_scores=True,
            clean_up_tokenization_spaces=True,
            max_length=self.max_question_len,
            min_length=3,
            num_beams=self.num_beams,
            num_return_sequences=self.num_return_sequences,
            **self.kwargs
        )

        assert len(all_records) == len(data) * self.num_return_sequences

        generated_questions = [r["generated_text"].strip() for r in all_records]
        scores = [r.get("score", None) for r in all_records]

        batched_questions = [
            generated_questions[i:i + self.num_return_sequences]
            for i in range(0, len(generated_questions), self.num_return_sequences)
        ]
        batched_scores = [
            scores[i:i + self.num_return_sequences]
            for i in range(0, len(scores), self.num_return_sequences)
        ]

        return batched_questions, batched_scores

    def generate_questions_from_passage_answer_pairs(self, passage_answer_pairs, disable_tqdm=False):
        outputs = []
        for batch in tqdm(
            _batch_iterator(passage_answer_pairs, self.batch_size, include_title=self.include_title),
            disable=disable_tqdm,
            total=len(passage_answer_pairs) // self.batch_size
        ):
            # try:
            batch_ids, batch_answers, batch_inputs = zip(*batch)
            batch_questions, batch_scores = self.generate_question(list(batch_inputs))
            # except Exception as e:
            #     logging.info('skipping Broken batch')
            #     continue

            for passage_id, answer, questions, scores in zip(batch_ids, batch_answers, batch_questions,
                                                             batch_scores):
                for question, score in zip(questions, scores):
                    output = {
                        "passage_id": passage_id,
                        "answer": answer["text"],
                        "question": question,
                        "metadata": {
                            "answer_start": answer["start"],
                            "answer_end": answer["end"],
                            "ae_score": answer["score"],
                            "qg_score": score,
                        },
                    }
                    outputs.append(output)
        return outputs


def load_question_generator(config):
    return QuestionGenerator(**config['config'])
