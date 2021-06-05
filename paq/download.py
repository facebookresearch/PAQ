#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import gzip
import logging
import os
import pathlib
import wget
import tarfile

from typing import Tuple, List

logger = logging.getLogger(__name__)


NQ_LICENSE_FILES = [
    "https://dl.fbaipublicfiles.com/dpr/nq_license/LICENSE",
    "https://dl.fbaipublicfiles.com/dpr/nq_license/README",
]

RESOURCES_MAP = {
    "paq.PAQ": {
        's3_url': 'https://dl.fbaipublicfiles.com/paq/v1/PAQ.tar.gz',
        "original_ext": ".tar.gz",
        "compressed": True,
        "desc": "Full PAQ generated QA pairs (PAQ-L + PAQ-NE)",
        "skip_if_exists_path": "paq/PAQ"
    },
    "paq.PAQ_L1": {
        's3_url': 'https://dl.fbaipublicfiles.com/paq/v1/PAQ_L1.tar.gz',
        "original_ext": ".tar.gz",
        "compressed": True,
        "desc": "PAQ-L1 subset of PAQ generated QA pairs",
        "skip_if_exists_path": "paq/PAQ_L1"
    },
    "paq.PAQ_L4": {
        's3_url': 'https://dl.fbaipublicfiles.com/paq/v1/PAQ_L4.tar.gz',
        "original_ext": ".tar.gz",
        "compressed": True,
        "desc": "PAQ-L4 subset of PAQ generated QA pairs",
        "skip_if_exists_path": "paq/PAQ_L4"
    },
    "paq.PAQ_NE1": {
        's3_url': 'https://dl.fbaipublicfiles.com/paq/v1/PAQ_NE1.tar.gz',
        "original_ext": ".tar.gz",
        "compressed": True,
        "desc": "PAQ-NE1 subset of PAQ generated QA pairs",
        "skip_if_exists_path": "paq/PAQ_NE1"
    },
    "paq.TQA_TRAIN_NQ_TRAIN_PAQ": {
        's3_url': 'https://dl.fbaipublicfiles.com/paq/v1/TQA_TRAIN_NQ_TRAIN_PAQ.tar.gz',
        "original_ext": ".tar.gz",
        "compressed": True,
        "desc": "TriviaQA train set QA pairs, NQ train set QA pairs and Full PAQ generated QA pairs",
        "skip_if_exists_path": "paq/TQA_TRAIN_NQ_TRAIN_PAQ"

    },
    "paq.psgs_w100": {
        's3_url': 'https://dl.fbaipublicfiles.com/paq/v1/psgs_w100.tsv.gz',
        "original_ext": ".tsv",
        "compressed": True,
        "desc": "Preprocessed wikipedia dump, split into 100 word passages",
        "skip_if_exists_path": "paq/psgs_w100.tsv"
    },
    "paq.PASSAGE_SCORES": {
        's3_url': 'https://dl.fbaipublicfiles.com/paq/v1/PASSAGE_SCORES.tar.gz',
        "original_ext": ".tar.gz",
        "compressed": True,
        "desc": "Passage selection scores for the passages in `psgs_w100`",
        "skip_if_exists_path": "paq/PASSAGE_SCORES"
    },
    "paq.PAQ_metadata": {
        's3_url': 'https://dl.fbaipublicfiles.com/paq/v1/PAQ.metadata.jsonl.gz',
        "original_ext": ".jsonl",
        "compressed": True,
        "desc": "PAQ QA pairs metadata ",
        "skip_if_exists_path": "paq/PAQ.metadata.jsonl"
    },
    "paq.PAQ_unfiltered_metadata": {
        's3_url': 'https://dl.fbaipublicfiles.com/paq/v1/PAQ.unfiltered_metadata.jsonl.gz',
        "original_ext": ".jsonl",
        "compressed": True,
        "desc": "PAQ QA pairs metadata for unfiltered QA pairs",
        "skip_if_exists_path": "paq/PAQ.unfiltered_metadata.jsonl"
    },

    'annotated_datasets.naturalquestions': {
        's3_url': [
            'https://dl.fbaipublicfiles.com/paq/v1/annotated_datasets/NQ-open.train-train.jsonl',
            'https://dl.fbaipublicfiles.com/paq/v1/annotated_datasets/NQ-open.train-dev.jsonl',
            'https://dl.fbaipublicfiles.com/paq/v1/annotated_datasets/NQ-open.test.jsonl',
            'https://dl.fbaipublicfiles.com/paq/v1/annotated_datasets/NQ_LICENSE',
            'https://dl.fbaipublicfiles.com/paq/v1/annotated_datasets/NQ_README',
        ],
        "original_ext": [".jsonl", ".jsonl", ".jsonl", "", ""],
        "compressed": False,
        "desc": "The Open NaturalQuestions QA Pairs used in our experiments",
        "skip_if_exists_path": "annotated_datasets/naturalquestions"
    },
    'annotated_datasets.triviaqa': {
        's3_url': [
            'https://dl.fbaipublicfiles.com/paq/v1/annotated_datasets/triviaqa.train-train.jsonl',
            'https://dl.fbaipublicfiles.com/paq/v1/annotated_datasets/triviaqa.train-dev.jsonl',
            'https://dl.fbaipublicfiles.com/paq/v1/annotated_datasets/triviaqa.test.jsonl'
        ],
        "original_ext": ".jsonl",
        "compressed": False,
        "desc": "The TriviaQA QA Pairs used in our experiments",
        "skip_if_exists_path": "annotated_datasets/triviaqa"
    },

    "models.retrievers.retriever_multi_base_256": {
        's3_url': 'https://dl.fbaipublicfiles.com/paq/v1/models/retrievers/retriever_multi_base_256.tar.gz',
        "original_ext": ".tar.gz",
        "compressed": True,
        "desc": "RePAQ Retriever Albert-Base model with 256 output embedding dim, multask. Recommended RePAQ retriever",
        "skip_if_exists_path": "models/retrievers/retriever_multi_base_256"
    },
    "models.retrievers.retriever_multi_base": {
        's3_url': 'https://dl.fbaipublicfiles.com/paq/v1/models/retrievers/retriever_multi_base.tar.gz',
        "original_ext": ".tar.gz",
        "compressed": True,
        "desc": "RePAQ Retriever Albert-Base model with 768 output embedding dim, multitask",
        "skip_if_exists_path": "models/retrievers/retriever_multi_base"
    },
    "models.retrievers.retriever_multi_large": {
        's3_url': 'https://dl.fbaipublicfiles.com/paq/v1/models/retrievers/retriever_multi_large.tar.gz',
        "original_ext": ".tar.gz",
        "compressed": True,
        "desc": "RePAQ Retriever Albert-Large model with 768 output embedding dim, multitask",
        "skip_if_exists_path": "models/retrievers/retriever_multi_large"
    },
    "models.retrievers.retriever_multi_xlarge": {
        's3_url': 'https://dl.fbaipublicfiles.com/paq/v1/models/retrievers/retriever_multi_xlarge.tar.gz',
        "original_ext": ".tar.gz",
        "compressed": True,
        "desc": "RePAQ Retriever Albert-xlarge model with 768 output embedding dim, multitask",
        "skip_if_exists_path": "models/retrievers/retriever_multi_xlarge"
    },
    "models.retrievers.retriever_nq_base": {
        's3_url': 'https://dl.fbaipublicfiles.com/paq/v1/models/retrievers/retriever_nq_base.tar.gz',
        "original_ext": ".tar.gz",
        "compressed": True,
        "desc": "RePAQ Retriever Albert-base model with 768 output embedding dim, trained on NQ",
        "skip_if_exists_path": "models/retrievers/retriever_nq_base"
    },
    "models.retrievers.retriever_nq_large": {
        's3_url': 'https://dl.fbaipublicfiles.com/paq/v1/models/retrievers/retriever_nq_large.tar.gz',
        "original_ext": ".tar.gz",
        "compressed": True,
        "desc": "RePAQ Retriever Albert-large model with 768 output embedding dim, trained on NQ",
        "skip_if_exists_path": "models/retrievers/retriever_nq_large"
    },
    "models.retrievers.retriever_nq_xlarge": {
        's3_url': 'https://dl.fbaipublicfiles.com/paq/v1/models/retrievers/retriever_nq_xlarge.tar.gz',
        "original_ext": ".tar.gz",
        "compressed": True,
        "desc": "RePAQ Retriever Albert-xlarge model with 768 output embedding dim, trained on NQ",
        "skip_if_exists_path": "models/retrievers/retriever_nq_xlarge"

    },
    "models.retrievers.retriever_tqa_base": {
        's3_url': 'https://dl.fbaipublicfiles.com/paq/v1/models/retrievers/retriever_tqa_base.tar.gz',
        "original_ext": ".tar.gz",
        "compressed": True,
        "desc": "RePAQ Retriever Albert-base model with 768 output embedding dim, trained on TriviaQA",
        "skip_if_exists_path": "models/retrievers/retriever_tqa_base"
    },
    "models.retrievers.retriever_tqa_large": {
        's3_url': 'https://dl.fbaipublicfiles.com/paq/v1/models/retrievers/retriever_tqa_large.tar.gz',
        "original_ext": ".tar.gz",
        "compressed": True,
        "desc": "RePAQ Retriever Albert-large model with 768 output embedding dim, trained on TriviaQA",
        "skip_if_exists_path": "models/retrievers/retriever_tqa_large"
    },
    "models.retrievers.retriever_tqa_xlarge": {
        's3_url': 'https://dl.fbaipublicfiles.com/paq/v1/models/retrievers/retriever_tqa_xlarge.tar.gz',
        "original_ext": ".tar.gz",
        "compressed": True,
        "desc": "RePAQ Retriever Albert-xlarge model with 768 output embedding dim, trained on TriviaQA",
        "skip_if_exists_path": "models/retrievers/retriever_tqa_xlarge"
    },



    'vectors.multi_base_256_vectors': {
        "s3_url": [
            "https://dl.fbaipublicfiles.com/paq/v1/models/vectors/multi_base_256_vectors/embeddings.pt.{}".format(
                i
            )
            for i in range(50)
        ],
        "original_ext": ".pt",
        "compressed": False,
        "desc": "Precomputed vectors for tqa-train-nq-train-PAQ.jsonl, using the `multi_base_256` model",
        "skip_if_exists_path": "vectors/multi_base_256_vectors"
    },


    "indices.multi_base_256_flat_sq8": {
        's3_url': 'https://dl.fbaipublicfiles.com/paq/v1/models/indices/multi_base_256.flat.sq8.faiss',
        "original_ext": ".faiss",
        "compressed": False,
        "desc": "Precomputed Flat Faiss Index for tqa-train-nq-train-PAQ.jsonl, using the `multi_base_256` model. Slow but exact",
        "skip_if_exists_path": "indices/multi_base_256_flat_sq8"
    },
    "indices.multi_base_256_hnsw_sq8": {
        's3_url': 'https://dl.fbaipublicfiles.com/paq/v1/models/indices/multi_base_256.hnsw.sq8.faiss',
        "original_ext": ".faiss",
        "compressed": False,
        "desc": "Precomputed Flat Faiss Index for tqa-train-nq-train-PAQ.jsonl, using the `multi_base_256` model. "
                "Very Fast but slightly less accurate than `multi_base_256_flat_sq8`",
        "skip_if_exists_path": "indices/multi_base_256_hnsw_sq8"
    },

    "models.rerankers.reranker_multi_base": {
        's3_url': 'https://dl.fbaipublicfiles.com/paq/v1/models/rerankers/reranker_multi_base.tar.gz',
        "original_ext": ".tar.gz",
        "compressed": True,
        "desc": "RePAQ Reranker AlBERT-Base model, multitask",
        "skip_if_exists_path": "models/rerankers/reranker_multi_base"

    },
    "models.rerankers.reranker_multi_large": {
        's3_url': 'https://dl.fbaipublicfiles.com/paq/v1/models/rerankers/reranker_multi_large.tar.gz',
        "original_ext": ".tar.gz",
        "compressed": True,
        "desc": "RePAQ Reranker AlBERT-Large model, multitask",
        "skip_if_exists_path": "models/rerankers/reranker_multi_large"
    },
    "models.rerankers.reranker_multi_xlarge": {
        's3_url': 'https://dl.fbaipublicfiles.com/paq/v1/models/rerankers/reranker_multi_xlarge.tar.gz',
        "original_ext": ".tar.gz",
        "compressed": True,
        "desc": "RePAQ Reranker AlBERT-xlarge model, multitask",
        "skip_if_exists_path": "models/rerankers/reranker_multi_xlarge"
    },
    "models.rerankers.reranker_multi_xxlarge": {
        's3_url': 'https://dl.fbaipublicfiles.com/paq/v1/models/rerankers/reranker_multi_xxlarge.tar.gz',
        "original_ext": ".tar.gz",
        "compressed": True,
        "desc": "RePAQ Reranker AlBERT-xxlarge model, multitask",
        "skip_if_exists_path": "models/rerankers/reranker_multi_xxlarge"
    },
    "models.rerankers.reranker_tqa_xlarge": {
        's3_url': 'https://dl.fbaipublicfiles.com/paq/v1/models/rerankers/reranker_tqa_xlarge.tar.gz',
        "original_ext": ".tar.gz",
        "compressed": True,
        "desc": "RePAQ Reranker AlBERT-xlarge model, TriviaQA-trained",
        "skip_if_exists_path": "models/rerankers/reranker_tqa_xlarge"
    },
    "models.rerankers.reranker_tqa_xxlarge": {
        's3_url': 'https://dl.fbaipublicfiles.com/paq/v1/models/rerankers/reranker_tqa_xxlarge.tar.gz',
        "original_ext": ".tar.gz",
        "compressed": True,
        "desc": "RePAQ Reranker AlBERT-xxlarge model, TriviaQA-trained",
        "skip_if_exists_path": "models/rerankers/reranker_tqa_xxlarge"
    },
    "models.rerankers.reranker_nq_xlarge": {
        's3_url': 'https://dl.fbaipublicfiles.com/paq/v1/models/rerankers/reranker_nq_xlarge.tar.gz',
        "original_ext": ".tar.gz",
        "compressed": True,
        "desc": "RePAQ Reranker AlBERT-xlarge model, NQ-trained",
        "skip_if_exists_path": "models/rerankers/reranker_nq_xlarge"
    },
    "models.rerankers.reranker_nq_xxlarge": {
        's3_url': 'https://dl.fbaipublicfiles.com/paq/v1/models/rerankers/reranker_nq_xxlarge.tar.gz',
        "original_ext": ".tar.gz",
        "compressed": True,
        "desc": "RePAQ Reranker AlBERT-xxlarge model, NQ-trained",
        "skip_if_exists_path": "models/rerankers/reranker_nq_xxlarge"
    },

    "models.passage_rankers.passage_ranker_base": {
        's3_url': 'https://dl.fbaipublicfiles.com/paq/v1/models/passage_rankers/passage_ranker_base.tar.gz',
        "original_ext": ".tar.gz",
        "compressed": True,
        "desc": "Passage Ranker model, BERT-base model, trained on NQ passages with hard negatives",
        "skip_if_exists_path": "models/passage_rankers/passage_ranker_base"
    },


    "models.qgen.qgen_multi_base": {
        's3_url': 'https://dl.fbaipublicfiles.com/paq/v1/models/qgen/qgen_multi_base.tar.gz',
        "original_ext": ".tar.gz",
        "compressed": True,
        "desc": "Question Generator model. BART-base model, multitask-trained",
        "skip_if_exists_path": "models/qgen/qgen_multi_base"
    },
    "models.qgen.qgen_nq_base": {
        's3_url': 'https://dl.fbaipublicfiles.com/paq/v1/models/qgen/qgen_nq_base.tar.gz',
        "original_ext": ".tar.gz",
        "compressed": True,
        "desc": "Question Generator model. BART-base model, NQ-trained",
        "skip_if_exists_path": "models/qgen/qgen_nq_base"
    },


    "models.filtering.dpr_nq_passage_retriever": {
        's3_url': 'https://dl.fbaipublicfiles.com/paq/v1/models/filtering/dpr_nq_passage_retriever.tar.gz',
        "original_ext": ".tar.gz",
        "compressed": True,
        "desc": "DPR Passage retriever and faiss index, from the DPR Paper, used in global filtering, NQ-trained",
        "skip_if_exists_path": "models/filtering/dpr_nq_passage_retriever",

    },
    "models.filtering.fid_reader_nq_base": {
        's3_url': 'https://dl.fbaipublicfiles.com/paq/v1/models/filtering/fid_reader_nq_base.tar.gz',
        "original_ext": ".tar.gz",
        "compressed": True,
        "desc": "FID-base reader, from the Fusion-in-Decoder paper, used in global and local filtering, NQ-trained",
        "skip_if_exists_path": "models/filtering/fid_reader_nq_base",
    },

    "models.answer_extractors.answer_extractor_nq_base": {
        's3_url': 'https://dl.fbaipublicfiles.com/paq/v1/models/answer_extractors/answer_extractor_nq_base.tar.gz',
        "original_ext": ".tar.gz",
        "compressed": True,
        "desc": "Learnt Answer Span Extractor, BERT-base, NQ-trained ",
        "skip_if_exists_path": "models/answer_extractors/answer_extractor_nq_base",
    },


    "predictions.retriever_results.multi_xlarge_nq_test": {
        's3_url': 'https://dl.fbaipublicfiles.com/paq/v1/predictions/retriever_results/multi_xlarge_nq_test.jsonl.gz',
        "original_ext": ".jsonl",
        "compressed": True,
        "desc": "Learnt Answer Span Extractor, BERT-base, NQ-trained ",
        "skip_if_exists_path": "predictions/retriever_results/multi_xlarge_nq_test.jsonl",
    },


}


def untar(tar_filename: str) -> List[str]:
    logger.info("Uncompressing %s", tar_filename)
    tar = tarfile.open(tar_filename)
    tar.extractall(path=os.path.dirname(tar_filename))
    tar.close()
    tar_dir = tar_filename.split('.tar.gz.tmp')[0]
    return [os.path.join(tar_dir, f) for f in os.listdir(tar_dir)]


def unpack(gzip_file: str, out_file: str):
    logger.info("Uncompressing %s", gzip_file)
    input = gzip.GzipFile(gzip_file, "rb")
    s = input.read()
    input.close()
    output = open(out_file, "wb")
    output.write(s)
    output.close()
    logger.info(" Saved to %s", out_file)


def _get_root_dir(out_dir):
    if out_dir:
        root_dir = out_dir
    else:
        # since hydra overrides the location for the 'current dir' for every run and we don't want to duplicate
        # resources multiple times, remove the current folder's volatile part
        root_dir = os.path.abspath("./")
        if "/outputs/" in root_dir:
            root_dir = root_dir[: root_dir.index("/outputs/")]
    return root_dir

def download_resource(
    s3_url: str, original_ext: str, compressed: bool, resource_key: str, out_dir: str, use_url_fname=False,
) -> Tuple[str, str]:
    logger.info("Requested resource from %s", s3_url)
    path_names = resource_key.split(".")

    root_dir = _get_root_dir(out_dir)
    logger.info("Download root_dir %s", root_dir)
    save_root = os.path.join(root_dir, "data", *path_names[:-1])  # last segment is for file name

    pathlib.Path(save_root).mkdir(parents=True, exist_ok=True)
    if use_url_fname:
        local_file_uncompressed = os.path.abspath(
            os.path.join(save_root, s3_url.split('/')[-1])
        )
    else:
        local_file_uncompressed = os.path.abspath(
            os.path.join(save_root, path_names[-1] + original_ext)
        )
    logger.info("File to be downloaded as %s", local_file_uncompressed)

    if os.path.exists(local_file_uncompressed):
        logger.info("File already exist %s", local_file_uncompressed)
        return save_root, local_file_uncompressed

    local_file = local_file_uncompressed if not compressed else local_file_uncompressed + '.tmp'
    wget.download(s3_url, out=local_file)

    logger.info("Downloaded to %s", local_file)

    if compressed:
        if original_ext == '.tar.gz':
            local_files = untar(local_file)
            os.remove(local_file)
            local_file = ','.join(local_files)
        else:
            uncompressed_file = os.path.join(save_root, path_names[-1] + original_ext)
            unpack(local_file, uncompressed_file)
            os.remove(local_file)
            local_file = uncompressed_file
    return save_root, local_file


def download_file(s3_url: str, out_dir: str, file_name: str):
    logger.info("Loading from %s", s3_url)
    local_file = os.path.join(out_dir, file_name)

    if os.path.exists(local_file):
        logger.info("File already exist %s", local_file)
        return

    wget.download(s3_url, out=local_file)
    logger.info("Downloaded to %s", local_file)


def download(resource_key: str, out_dir: str = None):
    if resource_key not in RESOURCES_MAP:
        # match by prefix
        resources = [k for k in RESOURCES_MAP.keys() if k.startswith(resource_key)]
        if resources:
            for key in resources:
                download(key, out_dir)
        else:
            logger.info("no resources found for specified key")
        return []
    download_info = RESOURCES_MAP[resource_key]

    if "skip_if_exists_path" in download_info:
        root_dir = _get_root_dir(out_dir)
        save_root = os.path.join(root_dir, "data", download_info['skip_if_exists_path'])
        if os.path.exists(save_root):
            logger.info(f"Resource: {resource_key} already exists here: {save_root}, "
                        f"delete this directory to force re-download")
            return []


    s3_url = download_info["s3_url"]

    save_root_dir = None
    data_files = []
    if isinstance(s3_url, list):
        if isinstance(download_info["original_ext"], str):
            exts = [download_info["original_ext"] for _ in s3_url]
        else:
            exts = download_info['original_ext']
        for i, (url, ext) in enumerate(zip(s3_url, exts)):
            save_root_dir, local_file = download_resource(
                url,
                ext,
                download_info["compressed"],
                resource_key,
                # "{}_{}".format(resource_key, i),
                out_dir,
                True
            )
            data_files.append(local_file)
    else:
        save_root_dir, local_file = download_resource(
            s3_url,
            download_info["original_ext"],
            download_info["compressed"],
            resource_key,
            out_dir,
        )
        data_files.append(local_file)

    license_files = download_info.get("license_files", None)
    if license_files:
        download_file(license_files[0], save_root_dir, "LICENSE")
        download_file(license_files[1], save_root_dir, "README")
    return data_files


def main():
    NL = '\n'
    parser = argparse.ArgumentParser("Tool for downloading resources",formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "--output_dir",
        default="./",
        type=str,
        help="The output directory to download file",
    )
    parser.add_argument(
        "--name", "-n",
        type=str,
        required=True,
        help=f"Resource name. Choose between: {NL + NL.join([str(k) + ' : ' + str(v['desc']) for k, v in RESOURCES_MAP.items()])}",
    )
    parser.add_argument('-v', '--verbose', action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    if args.name:
        downloaded_files = download(args.name, args.output_dir)
        logger.info(f'\nDownloaded the following files for resource {args.name} :')
        for d in downloaded_files:
            if ',' in d:
                for d2 in d.split(','):
                    logger.info(d2)
            else:
                logger.info(f'Downloaded {d}')
    else:
        logger.error("Please specify resource value. Possible options are:")
        for k, v in RESOURCES_MAP.items():
            logger.error("Resource key=%s  :  %s", k, v["desc"])


if __name__ == "__main__":
    main()
