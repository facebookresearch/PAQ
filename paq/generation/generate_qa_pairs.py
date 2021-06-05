#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
import os
from collections import defaultdict
import logging
import argparse

from paq.paq_utils import load_jsonl, dump_jsonl, get_submitit_executor
from paq.generation.passage_scorer.score_passages import score_passages_and_write_to_file
from paq.generation.answer_extractor.extract_answers import extract_answers_and_write_to_file
from paq.generation.question_generator.generate_questions import generate_questions_and_write_to_file
from paq.generation.filtering.filter_questions import filter_generated_questions_and_write_to_file


logger = logging.getLogger(__name__)

CONFIG_FILE = "config.json"

FINAL_OUTPUT = "final_qas.jsonl"
FINAL_DONE = "FINAL_DONE"


def touch(path):
    """Create an empty file. Update the mtime if it exists."""
    with open(path, 'a'):
        os.utime(path, None)


def _run_pipeline_step(config, input_file, output_file, done_indicator, verbose, fun):
    if not os.path.exists(done_indicator):
        fun(config, input_file, output_file, verbose)
        touch(done_indicator)
    return output_file


def run_passage_scoring(config, input_file, output_dir, verbose=False):
    output_file = os.path.join(output_dir, "ps.jsonl")
    done_path = os.path.join(output_dir, "PS_DONE")
    func = score_passages_and_write_to_file
    return _run_pipeline_step(config['passage_scorer'], input_file, output_file, done_path, verbose, func)


def run_answer_extraction(config, input_file, output_dir, verbose=False):
    output_file = os.path.join(output_dir, "ae.jsonl")
    done_path = os.path.join(output_dir, "AE_DONE")
    func = extract_answers_and_write_to_file
    return _run_pipeline_step(config['answer_extractor'], input_file, output_file, done_path, verbose, func)


def run_question_generation(config, input_file, output_dir, verbose=False):
    output_file = os.path.join(output_dir, "qg.jsonl")
    done_path = os.path.join(output_dir, "QG_DONE")
    func = generate_questions_and_write_to_file
    return _run_pipeline_step(config['question_generator'], input_file, output_file, done_path, verbose, func)


def run_filtering(config, input_file, output_dir, verbose=False):
    output_file = os.path.join(output_dir, "filterd_qg.jsonl")
    done_path = os.path.join(output_dir, "FILTERED_DONE")
    func = filter_generated_questions_and_write_to_file
    return _run_pipeline_step(config['filterer'], input_file, output_file, done_path, verbose, func)


def combine_generated_files(document_ranker_file,
                            question_generation_file,
                            output_file
                            ):
    # Write final generated QA-pairs to an output file

    def _get_passage_score_map(doc_ranker_file):
        passage_scores = {}
        with open(doc_ranker_file, "r") as f:
            for line in f.readlines():
                row = json.loads(line)
                passage_scores[row["passage_id"]] = row["metadata"].get("ps_score", None)
        return passage_scores

    def _add_passage_metadata(questions_fi, passage_scores):
        generated_qas = load_jsonl(questions_fi)
        qas_dict = defaultdict(list)
        for qas in generated_qas:
            question, answer, passage_id = qas["question"], qas["answer"], qas["passage_id"]
            metadata = {"passage_id": passage_id, "ps_score": passage_scores[passage_id], 'answer': answer}
            metadata.update(qas["metadata"])
            qas_dict[question].append((answer, metadata))
        return qas_dict

    def _get_output_format(qas_dict):
        final_qas = []
        for question, answers_meta in qas_dict.items():
            answers, metadata_list = zip(*answers_meta)
            final_qa = {"question": question, "answer": answers, "metadata": metadata_list}
            final_qas.append(final_qa)
        return final_qas

    passage_score_map = _get_passage_score_map(document_ranker_file)
    qas_with_meta = _add_passage_metadata(question_generation_file, passage_score_map)
    final_qas = _get_output_format(qas_with_meta)
    dump_jsonl(final_qas, output_file)


def run_paq_generation_pipeline(config: dict, input_file: str, output_dir: str, verbose: bool = False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the config
    config["source"], config['output_dir'] = input_file, output_dir
    with open(os.path.join(output_dir, CONFIG_FILE), "w") as cf:
        json.dump(config, cf, indent=2)

    # Run the pipeline:
    passages_fi = run_passage_scoring(config, input_file, output_dir, verbose=verbose)
    answers_fi = run_answer_extraction(config, passages_fi, output_dir, verbose=verbose)
    questions_fi = run_question_generation(config, answers_fi, output_dir, verbose=verbose)
    filtered_questions_fi = run_filtering(config, questions_fi, output_dir, verbose=verbose)

    # Write final generated QA-pairs to an output file
    output_fi = os.path.join(output_dir, FINAL_OUTPUT)
    logging.info(f"Writing generated QA pairs to {output_fi}...")

    final_indicator = os.path.join(output_dir, FINAL_DONE)
    if not os.path.exists(final_indicator):
        output_fi = os.path.join(output_dir, FINAL_OUTPUT)
        combine_generated_files(passages_fi, filtered_questions_fi, output_fi)
        touch(final_indicator)


def _is_job_finished(job_number, output_dir):
    if os.path.exists(os.path.join(output_dir, FINAL_DONE)):
        print(f'not launching job {job_number} as its already finished: ', os.path.join(output_dir, FINAL_DONE))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_config", help="Path of config file")
    parser.add_argument("--passage_files_to_generate", help="comma separated list of files to generate QA pairs from")
    parser.add_argument("--output_dirs", help="comma separated list of directories to write the generated QA pairs to")
    parser.add_argument('--n_jobs', type=int, required=True, help='how many parallel jobs to use in slurm (n_jobs=-1 will run locally)')
    parser.add_argument('--slurm_partition', type=str, default="learnfair", help='If using submitit to run slurm jobs, define cluster partition here')
    parser.add_argument('--slurm_comment', type=str, default="", help='If using submitit to run slurm jobs, define job comment heree')
    parser.add_argument('-v', '--verbose', action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    with open(args.path_to_config) as f:
        config = json.load(f)

    input_files = args.passage_files_to_generate.split(',')
    output_dirs = args.output_dirs.split(',')

    if args.n_jobs == -1:
        # Run locally
        for i, (inf, out_dir) in enumerate(zip(input_files, output_dirs)):
            if not _is_job_finished(i, out_dir):
                logging.info(f'Running generation job {i}:\ninput file: {inf} \nSaving results to: {out_dir}')
                run_paq_generation_pipeline(config, inf, out_dir, args.verbose)
    else:
        # Run with submitit
        executor = get_submitit_executor(n_jobs=args.n_jobs, comment=args.slurm_comment, partition=args.slurm_partition)
        jobs = []
        with executor.batch():
            for i, (inf, out_dir) in enumerate(zip(input_files, output_dirs)):
                if not _is_job_finished(i, out_dir):
                    job = executor.submit(run_paq_generation_pipeline, config, inf, out_dir, args.verbose)
                    jobs.append((job, inf, out_dir))

        logging.info('Launching the following jobs:')
        for job, inf, out_dir in jobs:
            logging.info(f'{job.job_id} {inf} -> {out_dir}')
