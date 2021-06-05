#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
from paq.paq_utils import load_jsonl, dump_jsonl
from paq.generation.filtering.filterer import load_retriever, load_reader
import logging
import argparse

logger = logging.getLogger(__name__)


def retrieve_documents_for_generated_questions(config, input_file, verbose):
    retriever = load_retriever(config["retriever"])
    generated_questions = load_jsonl(input_file)
    logger.info("Running Filterer Retriever...")
    generated_questions_with_retrieved_docs = retriever.retrieve_documents(generated_questions)
    return generated_questions_with_retrieved_docs


def generate_answers_for_generated_questions_with_retrieved_docs(config, input_file, verbose):
    reader = load_reader(config["reader"])
    generated_questions_with_retrieved_docs = load_jsonl(input_file)

    logger.info("Running Filterer Reader...")
    results = reader.generate_answers(generated_questions_with_retrieved_docs)
    return results


def filter_generated_questions_and_write_to_file(config, input_path, output_path, verbose):
    results = retrieve_documents_for_generated_questions(config, input_path, verbose)
    retrieval_results_fi = output_path + '.retrieval_results'
    dump_jsonl(results, retrieval_results_fi)
    results = generate_answers_for_generated_questions_with_retrieved_docs(config, retrieval_results_fi, verbose)
    logger.info('Writing generated questions to file...')
    dump_jsonl(results, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Extract answers from passages")
    parser.add_argument('--generated_questions_to_filter',
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

    if 'filterer' in config:
        config = config['filterer']

    filter_generated_questions_and_write_to_file(config, args.generated_questions_to_filter, args.output_path, args.verbose)
