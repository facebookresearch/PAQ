#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
from paq.paq_utils import load_jsonl, dump_jsonl
from paq.generation.question_generator.generator import load_question_generator
import logging
import argparse

logger = logging.getLogger(__name__)


def generate_questions(config, input_file, verbose):
    question_generator = load_question_generator(config)
    passage_answer_pairs = load_jsonl(input_file)
    logger.info("Running Question Generation...")
    annotations = question_generator.generate_questions_from_passage_answer_pairs(passage_answer_pairs, disable_tqdm=not verbose)
    return annotations


def generate_questions_and_write_to_file(config, input_path, output_path, verbose):
    annotations = generate_questions(config, input_path, verbose)
    logger.info('writing generated questions to file...')
    dump_jsonl(annotations, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Extract answers from passages")
    parser.add_argument('--passage_answer_pairs_to_generate_from',
                        type=str,
                        required=True,
                        help='path to generate from (in jsonl format, produced by `answer_extractor`)')
    parser.add_argument('--output_path', type=str, required=True, help='Path to dump results to')
    parser.add_argument('--path_to_config', type=str, required=True, help='path to question generator config file')
    parser.add_argument('-v', '--verbose', action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    with open(args.path_to_config) as f:
        config = json.load(f)

    if 'question_generator' in config:
        config = config['question_generator']

        generate_questions_and_write_to_file(config, args.passage_answer_pairs_to_generate_from, args.output_path, args.verbose)
