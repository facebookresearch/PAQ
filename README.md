# PAQ: 65 Million Probably-Asked Questions and What You Can Do With Them


This repository contains code and models to support the research paper [PAQ: 65 Million Probably-Asked Questions and What You Can Do With Them](https://arxiv.org/abs/2102.07033)

<br>
<p align="center">
  <img src="https://dl.fbaipublicfiles.com/MLQA/logos.png" alt="Facebook AI Research and UCL NLP"  width="60%"/>
  <br>
</p>
<br>

## Table of Contents

* [Table of Contents](#table-of-contents)
* [Data Downloads](#data-downloads)
  * [PAQ QA-pairs](#paq-qa-pairs)
  * [PAQ Metadata](#paq-metadata)
  * [Preprocessed Wikipedia Dump](#preprocessed-wikipedia-dump)
  * [Passage Selector Scores](#passage-selector-scores)
  * [PAQ QA-pair metadata](#paq-qa-pair-metadata)
  * [PAQ <em>unfiltered</em> QA-pair metadata](#paq-unfiltered-qa-pair-metadata)
  * [Training/Dev/Test QA Pairs](#trainingdevtest-qa-pairs)
* [Code and Models](#code-and-models)
  * [Installation and Setup:](#installation-and-setup)
  * [Download Tool](#download-tool)
  * [Question Answering with RePAQ](#question-answering-with-repaq)
     * [RePAQ Retrievers:](#repaq-retrievers)
        * [Minimal Retrieval Inference Example:](#minimal-retrieval-inference-example)
        * [Retriever Models, Precomputed Vectors and Indexes:](#retriever-models-precomputed-vectors-and-indexes)
        * [Embedding QA pairs:](#embedding-qa-pairs)
        * [Building indices:](#building-indices)
        * [Retriever Inference:](#retriever-inference)
        * [Evaluating Retriever Results:](#evaluating-retriever-results)
     * [RePAQ ReRankers:](#repaq-rerankers)
        * [Minimal Reranker Inference example:](#minimal-reranker-inference-example)
        * [Reranker Models:](#reranker-models)
        * [ReRanker Inference:](#reranker-inference)
        * [Evaluating Rerankers:](#evaluating-rerankers)
  * [Question-Answer Pair Generation](#question-answer-pair-generation)
     * [Passage Scoring/Ranking](#passage-scoringranking)
     * [Answer Extraction](#answer-extraction)
     * [Question Generation](#question-generation)
     * [Filtering Generated QA-pairs](#filtering-generated-qa-pairs)
     * [End2End Generation Tool](#end2end-generation-tool)
* [Citing](#citing)
* [LICENSE](#license)
  * [Code License:](#code-license)
  * [Data License:](#data-license)
 
## Data Downloads

PAQ QA pairs, their metadata, preprocessed wikipedia dumps and Train/dev/test QA pairs downloads are described in this section. For downloading models, indices etc, see [Code And Models](#code-and-models) section.
In addition to downloading the data here, you can use the `paq.download` tool, (recommended for downloading models, indices etc), see the [Download Tool](#download-tool) section for use.

### PAQ QA-pairs

The PAQ QA pairs can be downloaded below. We use the same format as for NQ-open (see [here](https://github.com/google-research-datasets/natural-questions/tree/master/nq_open)). The
 TQA_TRAIN_NQ_TRAIN_PAQ is the concatenation of the TriviaQA and NQ training QA-Pairs with the PAQ QA-Pairs.

| Dataset  | # QAs | Size (unzipped)| link | License |
| ------------- | ------------- | --------- | ---- | -----|
| PAQ     | 64.9M   | 5.8 GB| [download](https://dl.fbaipublicfiles.com/paq/v1/PAQ.tar.gz) |  [CC-BY-SA](https://creativecommons.org/licenses/by-sa/3.0/)|
| PAQ-L1  | 14.1M   | 1.3 GB| [download](https://dl.fbaipublicfiles.com/paq/v1/PAQ_L1.tar.gz) | [CC-BY-SA](https://creativecommons.org/licenses/by-sa/3.0/)|
| PAQ-L4  |  53.8M  | 4.9 GB| [download](https://dl.fbaipublicfiles.com/paq/v1/PAQ_L4.tar.gz) | [CC-BY-SA](https://creativecommons.org/licenses/by-sa/3.0/)|
| PAQ-NE1 | 12.0M   | 1.0 GB| [download](https://dl.fbaipublicfiles.com/paq/v1/PAQ_NE1.tar.gz) | [CC-BY-SA](https://creativecommons.org/licenses/by-sa/3.0/)|
| TQA_TRAIN_NQ_TRAIN_PAQ | 65.00M   | 5.9 GB| [download](https://dl.fbaipublicfiles.com/paq/v1/TQA_TRAIN_NQ_TRAIN_PAQ.tar.gz) | [CC-BY-SA](https://creativecommons.org/licenses/by-sa/3.0/)|


###  PAQ Metadata

Available metadata to support PAQ is available, and can be downloaded from the following table. See the descriptions below for details:

| Dataset  | Size (unzipped)| link | License |
| ------------- | ------------- | --------- |  ----|
| Preprocessed Wikipedia Dump   | 13 GB| [download](https://dl.fbaipublicfiles.com/paq/v1/psgs_w100.tsv.gz) | [CC-BY-SA](https://creativecommons.org/licenses/by-sa/3.0/)|
| Passage Selector Scores  | 560 MB| [download](https://dl.fbaipublicfiles.com/paq/v1/PASSAGE_SCORES.tar.gz) | [CC-BY-SA](https://creativecommons.org/licenses/by-sa/3.0/)|
| PAQ QA-Pair metadata  |  16 GB| [download](https://dl.fbaipublicfiles.com/paq/v1/PAQ.metadata.jsonl.gz) | [CC-BY-SA](https://creativecommons.org/licenses/by-sa/3.0/)|
| PAQ *unfiltered* QA-pairs and metadata | 95 GB| [download](https://dl.fbaipublicfiles.com/paq/v1/PAQ.unfiltered_metadata.jsonl.gz) | [CC-BY-SA](https://creativecommons.org/licenses/by-sa/3.0/)|

### Preprocessed Wikipedia Dump

This file contains the preprocessed wikipedia dump used to generate PAQ. The file consists of 100-word passages of a 2018 Wikipedia dump, and was produced by [Karphukin et al.](https://github.com/facebookresearch/DPR) for [DPR](https://github.com/facebookresearch/DPR).
The file is in TSV format, with 3 columns. The first column is passage id, the second column is the passage text, the third is the wikipedia article title. 

### Passage Selector Scores

This file contains the passage selection scores for passages, using the passage selection model described in the paper.
The file is in TSV format, with 2 columns. The first column is passage id (see "Preprocessed Wikipedia Dump"), the second column is the logprob score from the passage selector for that passage.

### PAQ QA-pair metadata

This file contains metadata for the QA pairs in PAQ. The file is in jsonl format. Each line is a json dict with metadata for one question-answer pair in PAQ.
The format is as follows:
```
{
    "question":  question string
    "subsets":   list of PAQ subsets the question appears in ("L1", "L4" or "NE")
    "answer":  the question's answer produced by the consistency filter model
    "passage_score": passage selection score of highest scoring passage that generated this question
    "answers": [
        {
            "passage_id": id of wiki passage this answer was extracted from (see "Preprocessed Wikipedia Dump")
            "offset": character offset to start of answer span
            "text": text of answer span
            "extractor": answer extractor model, either "L" (for learnt extracor), or "NE" (for Named Entity extractor)
        },
        ...
    ]
}
```
There are a small number of questions where the "subset" is "NE-legacy". These questions were generated by an earlier iteration of the "NE" generation pipeline.

### PAQ *unfiltered* QA-pair metadata

This file contains similar metadata to that described above in "PAQ QA pair metadata", but for *all* generated questions, even those that do not pass the consistency filter. 
As such, this is a very large file, and is provided for completeness, but should not be of interest to most users interested in PAQ metadata.
The file is in jsonl format. Each line is a json dict with metadata for one question-answer pair.
The format is as follows:
```
{
    "question":  question string
    "subsets":   list of PAQ subsets the question appears in ("L1", "L4" or "NE")
    "consistent_subsets":  list of PAQ subsets the question appears in, which pass the consistnency filters ("L1", "L4" or "NE")
    "canonical_answer":  the question's answer produced by the consistency filter model
    "consistent": boolean. If true, the question passes the global consistency filter
    "passage_score": passage selection score of highest scoring passage that generated this question
    "answers": [
        {
            "passage_id": id of wiki passage this answer was extracted from (see "Preprocessed Wikipedia Dump")
            "offset": character offset to start of answer span
            "text": text of answer span
            "extractor": answer extractor model, either "L" (for learnt extracor), or "NE" (for Named Entity extractor)
            "consistent": boolean. If true, this answer span is the consistent with the answer from the global consistency filter
        },
        ...
    ]
}
```

### Training/Dev/Test QA Pairs

The QA Pairs in the Open Domain NaturalQuestions and TriviaQA Train/Dev/Test sets are available below, as well as a file with the concatenation of the training sets and PAQ (useful for retrieval later).


| Dataset  | Description | Link | 
| ------------- |------------- | --------- |
| NQ-open.train-train.jsonl | Open-NaturalQuestions Training set | [download](https://dl.fbaipublicfiles.com/paq/v1/annotated_datasets/NQ-open.train-train.jsonl)|
| NQ-open.train-dev.jsonl |  Open-NaturalQuestions Development set| [download](https://dl.fbaipublicfiles.com/paq/v1/annotated_datasets/NQ-open.train-dev.jsonl)|
| NQ-open.test.jsonl |  Open-NaturalQuestions Test set| [download](https://dl.fbaipublicfiles.com/paq/v1/annotated_datasets/NQ-open.test.jsonl)|
| triviaqa.train-train.jsonl | Open-TriviaQA Training set  | [download](https://dl.fbaipublicfiles.com/paq/v1/annotated_datasets/triviaqa.train-train.jsonl)|
| triviaqa.train-dev.jsonl | Open-TriviaQA Development set| [download](https://dl.fbaipublicfiles.com/paq/v1/annotated_datasets/triviaqa.train-dev.jsonl)|
| triviaqa.test.jsonl | Open-TriviaQA Test set| [download](https://dl.fbaipublicfiles.com/paq/v1/annotated_datasets/triviaqa.test.jsonl)|
| tqa-train-nq-train-PAQ.jsonl | Concatenation of NQ-open.train-train.jsonl, triviaqa.train-train.jsonl and PAQ | [download](https://dl.fbaipublicfiles.com/paq/v1/TQA_TRAIN_NQ_TRAIN_PAQ.tar.gz)|

## Code and Models

All users should follow the instructions in [Installation and Setup](#installation-and-setup), and use the [Download Tool](#download-tool), which will make downloanding models and assets much easier.

Code to run inference for Question Answering using RePAQ and the full question generation pipeline are now available. Functionality to help train your own models is coming soon.

Users interested in running question answering with REPAQ, read [Question Answering with RePAQ](#question-answering-with-repaq).

Users interested in running Question generation using the PAQ generation pipeline, read [Question Answering with RePAQ](#question-answering-with-repaq).



### Installation and Setup:

We highly recommend you use conda environments. The requirements are pytorch, spacy, Transformers 4.1.0 (other versions unlikely to work), FID, and the packages listed in `requirements.txt`.
The following script should install all nececessary code dependencies:

```bash
conda create -n paq python=3.7
conda activate paq

# install pytorch
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
conda install -c pytorch faiss-gpu cudatoolkit=10.1

# For Spacy:
conda install -c conda-forge spacy
conda install -c conda-forge cupy
python -m spacy download en_core_web_sm
pip install -r requirements.txt

# Install FID for QA-pair consistency filtering:
git clone git@github.com:facebookresearch/FiD.git
cd FiD; git checkout baf533c3f7a26c1cac624ee9252ce5ccf344a935

```

### Download Tool

To make downloading resources easier, we've built a download tool.
This is the recommended way for downloading data, trained models, precomputed vectors and indices. 
This will download and uncompress resources to the `./data` directory, where the code will expect these resources to be, and handle path management.
Run it by supplying a resource key name (run with `-h` to see available resources): 
```bash
# Downloads a RePAQ retriever model:
$ python -m paq.download -v -n models.retrievers.retriever_multi_base_256
```

### Question Answering with RePAQ

Question Answering over PAQ with RePAQ is accomplished using [Dense Retrieval](#repaq-retrievers), optionally following by [Reranking](#repaq-rerankers). 
Reranking will improve accuracy, but is slower.

To enable wider use our work,
we have trained more compact retrievers and indices than those used in the original paper.
Thse will still give strong results, but run on machines with smaller GPUs and modest amounts of CPU RAM (64GB CPU RAM should be plenty). 
These models are only marginally less accurate than the larger ones used in the paper, and we list them as "recommended" in the tables below.



#### RePAQ Retrievers:

##### Minimal Retrieval Inference Example:
TL;DR if you just want to run retrieval:

First, download 1) A retrieval model, 2) A KB of QA Pairs (in our case, TQA train set, NQ train set and PAQ) and 3) a pre-built index for those QA Pairs.

```bash
# download retriever model
$ python -m paq.download -v -n models.retrievers.retriever_multi_base_256

# Download QA Pairs, and a corresponding faiss index:
$ python -m paq.download -v -n paq.TQA_TRAIN_NQ_TRAIN_PAQ
$ python -m paq.download -v -n indices.multi_base_256_hnsw_sq8

# Download NaturalQuestions data, we'll run inference on the test set
$ python -m paq.download -v -n annotated_datasets.naturalquestions

```

Then, run retrieval inference (here we're using the v fast but slightly less accurate HNSW faiss index):

```bash
$ python -m paq.retrievers.retrieve \
    --model_name_or_path ./data/models/retrievers/retriever_multi_base_256 \
    --qas_to_answer data/annotated_datasets/NQ-open.test.jsonl \
    --qas_to_retrieve_from ./data/paq/TQA_TRAIN_NQ_TRAIN_PAQ/tqa-train-nq-train-PAQ.jsonl \
    --top_k 50 \
    --output_file my_retrieval_results.jsonl \
    --faiss_index_path data/indices/multi_base_256_hnsw_sq8.faiss \
    --fp16 \
    --memory_friendly_parsing \
    --verbose
```


Finally, either use a reranker to rerank the top K results (see [here](#minimal-reranker-inference-example)), or evaluate retrieval performance:

```bash
$ python -m paq.evaluation.eval_retriever \
    --predictions my_retrieval_results.jsonl \
    --references data/annotated_datasets/NQ-open.test.jsonl \
    --hits_at_k 1,10,50
1: 40.0%
(1443 / 3610)
10: 55.7%
(2010 / 3610)
50: 63.9%
(2306 / 3610)
```


##### Retriever Models, Precomputed Vectors and Indexes:

The following table lists the recommended models for inference. 
For an exahustive list of models available, see [full_models_list.md](./full_models_list.md). 
We highly recommend using `retriever_multi_base_256`. 
This model has been designed to be compute and memory-friendly. 
It's embedding dimension is 256 c.f. 768 used in the original paper, saving RAM when performing retrieval.
It  outperforms the base model from the paper, and loses only 0.7% on average ove the xlarge model from the paper.


| Model  | Training data |  Architecture | Embedding Dim | NQ EM | + rerank | TQA EM | + rerank |  Download Resource Key Name |
| ------------- |----------| --- | --------- | ---------- |---- |---- | ---- | ---- |
| retriever_multi_base_256  (recommended)| NQ + TQA | AlBERT-base | 256  | 41.4 | 47.3 | 40.2 | 50.9| `models.retrievers.retriever_multi_base_256` |
| retriever_multi_base | NQ + TQA | AlBERT-base |  728 | 40.9| 47.4 | 39.7 | 51.2 | `models.retrievers.retriever_multi_base`  |
| retriever_multi_xlarge | NQ + TQA | AlBERT-xlarge| 728  | 41.7 | 47.6 | 41.3 | 52.1 |`models.retrievers.retriever_multi_xlarge`|

The table below lists available precomputed embeddings and indices for download. The embeddings are stored according to the order in tqa-train-nq-train-PAQ.jsonl, corresponding to the TQA training set, the NQ training set and PAQ.
I.e. the kth QA pair in the file is embedded in the kth vector in these files.

To download precomputed vectors, use the `paq/download.py` script, as indicated in the table. 
We recommend using the FAISS indexes for running inference, either `multi_base_256.flat.sq8.faiss`
(slower, 1-10s questions/sec, but more accurate, and has lowest memory requirement ~16GB RAM), 
or `multi_base_256.hnsw.sq8.faiss` (very fast, 100-1000s questions/sec depending on machine, slightly less accurate (0.8% on average) but higher memory requirements ~32GB RAM)


| File  | Description | Size |  Download Resource Key Name  | 
| ------------- |------------- | --------- |---- |
| tqa-train-nq-train-PAQ.jsonl (required) | Concatenation of NQ-open.train-train.jsonl, triviaqa.train-train.jsonl and PAQ | | `paq.TQA_TRAIN_NQ_TRAIN_PAQ` |
| multi_base_256_vectors | embeddings for QAS in `tqa-train-nq-train-PAQ.jsonl` using `retriever_multi_base_256` | 16GB | `vectors.multi_base_vectors_256 ` |
| multi_base_vectors | embeddings for QAS in `tqa-train-nq-train-PAQ.jsonl` using `retriever_multi_base` | 48GB |`vectors.multi_base_vectors` |
| multi_xlarge_vectors| embeddings for QAS in `tqa-train-nq-train-PAQ.jsonl` using `retriever_multi_xlarge` | 48GB| `vectors.multi_xlarge_vectors` |
| multi_base_256.flat.sq8.faiss (recommended) | Flat FAISS index for `retriever_multi_base_256` - slower (1-10s questions / sec) | 16GB | `indices.multi_base_256_flat_sq8.faiss`|
| multi_base_256.hnsw.sq8.faiss (recommended) | Fast FAISS index for `retriever_multi_base_256` - faster (100-1000s queries / sec) | 32GB | `indices.multi_base_256_hnsw_sq8.faiss`|



##### Embedding QA pairs:

To embed a set of QA pairs in the NaturalQuestions jsonl format, use the `paq/evaluation/embed.py` file.

E.g. to embed the NQ training set to vectors using the `retriever_multi_base_256` model, and write them to disk,
run the following command:

```
python -m paq.retrievers.embed \
    --model_name_or_path ./data/models/retrievers/retriever_multi_base_256 \
    --qas_to_embed data/annotated_datasets/NQ-open.train-train.jsonl \
    --output_dir ./my_vectors \
    --fp16 \
    --batch_size 128 \
    --verbose \
    --n_jobs -1 
# see below for explanation of --n_jobs
```

For very large numbers of QA pairs, you may want to run this in parallel. 
This script is set up to work with submitit, and by default, can submit a slurm job array to embed the QA pairs in parallel.
For example, to run embedding locally, set `--n_jobs -1` (As above), or to run 10 parallel jobs to embed a file, run with `--n_jobs 10`.
The full command is given below:

```
python -m paq.retrievers.embed \
    --model_name_or_path ./data/models/retrievers/retriever_multi_base_256 \
    --qas_to_embed data/annotated_datasets/NQ-open.train-train.jsonl \
    --output_dir ./my_vectors_distributed \
    --fp16 \
    --batch_size 128 \
    --verbose \
    --memory_friendly_parsing \
    --n_jobs 10 \
    --slurm_partition my_clusters_partition \
    --slurm_comment "my embedding job"
    
```

The submitit job array config can be seen and edited for your clusters needs at `paq/paq_utils.py` (the `get_submitit_executor` function)

##### Building indices:

To build faiss MIPS indices on vectors produced by `paq.retrievers.embed`, (for improved quantization and speed over raw exact search in pytorch), use the `paq/retreiver/build_index.py`.
This will allow you to build indices like the ones used in the paper (specifically, Flat and HNSW indices, optionally with scalar quantization).

```
# build a flat index with Scaler quantization (slower queries, but slightly more accurate)
python -m paq.retrievers.build_index \
    --embeddings_dir ./my_vectors \
    --output_path ./my_index.faiss \
    --SQ8 \
    --verbose

# or, build an hnsw index with scaler (mcuh much faster qurerying, slightly less accurate)
python -m paq.retrievers.build_index \
    --embeddings_dir ./my_vectors \
    --output_path ./my_index.hnsw.faiss \
    --hnsw \
    --SQ8 \
    --store_n 32 \
    --ef_construction 128 \
    --ef_search 128 \
    --verbose

```

Building indices is a deep, nuanced and complex area. The scripts we provide to build indices is mostly a convenience and reproduciblity wrapper.
It's likely that stronger compression is possible without losing performance (e.g. by using Product Quantization), as is faster inference. 
If the indexes we provide are too large or slow, consider building your own by referring the the [faiss documentation](https://github.com/facebookresearch/faiss) directly.

##### Retriever Inference:

Run QA-pair Retrieval using `paq/retrievers/retrieve.py`. You can see argument help by passing `-h`. 
You must pass in a jsonl file of QA pairs to retrieve from, using the `--qas_to_retrieve_from` argument. 
You can also pass in either a directory of embeddings for the qa-pairs to retrieve from using the `--precomputed_embeddings_dir` (e.g. the output of `paq.retrievers.embed`) 
or a faiss index of the qa-pairs to retrieve from, using the `--faiss_index_path`. If neither `--faiss_index_path` or `--precomputed_embeddings_dir` are given, the QA-pairs to retrieve from will be embedded on-the-fly. This may be slow for large QA-pair KBs.

The following command will run retrieve the top 50 QA-pairs from the PAQ KB for the NQ-test set, using the fast HNSW faiss index, and write the results to `my_retrieval_results.jsonl`

```bash
#Download the relevant artefacts
$ python -m paq.download -v -n models.retrievers.retriever_multi_base_256
$ python -m paq.download -v -n paq.TQA_TRAIN_NQ_TRAIN_PAQ
$ python -m paq.download -v -n indices.multi_base_256_hnsw_sq8
$ python -m paq.download -v -n annotated_datasets.naturalquestions

$ python -m paq.retrievers.retrieve \
    --model_name_or_path ./data/models/retrievers/retriever_multi_base_256 \
    --qas_to_answer data/annotated_datasets/NQ-open.test.jsonl \
    --qas_to_retrieve_from ./data/paq/TQA_TRAIN_NQ_TRAIN_PAQ/tqa-train-nq-train-PAQ.jsonl \
    --top_k 50 \
    --output_file my_retrieval_results.jsonl \
    --faiss_index_path data/indices/multi_base_256_hnsw_sq8.faiss \
    --fp16 \
    --memory_friendly_parsing \
    --verbose
```
    
##### Evaluating Retriever Results:

Evaluate retrieval performance using the `paq.evaluation.eval_retriever` tool. 
It will return the hits@k (whether the correct answer is in the top K retrieved questions' answers). Hits@1 is equivalent to Exact Match score

```bash
$ python -m paq.evaluation.eval_retriever \
    --predictions my_retrieval_results.jsonl \
    --references data/annotated_datasets/NQ-open.test.jsonl \
    --hits_at_k 1,10,50
1: 40.0%
(1443 / 3610)
10: 55.7%
(2010 / 3610)
50: 63.9%
(2306 / 3610)
```


#### RePAQ ReRankers:

##### Minimal Reranker Inference example:
Tl;DR for if you just want to run reranking:
First, download a reranker model, (and if you dont already have retrieval results you want to rerank, download some)

```bash
# download reranker model (here we're using the albert xxlarge model, smaller ones are available)
$ python -m paq.download -v -n models.rerankers.reranker_multi_xxlarge

# download some retrieval results to rerank if you dont already have some
$ python -m paq.download -v -n predictions.retriever_results.multi_xlarge_nq

```
Next, run reranking:
```
$ python -m paq.rerankers.rerank \
    --model_name_or_path data/models/rerankers/reranker_multi_xxlarge  \
    --qas_to_rerank data/predictions/retriever_results/multi_xlarge_nq_test.jsonl \
    --output_file my_reranker_results.jsonl \
    --top_k 50 \
    --fp16 \
    --batch_size 4 --verbose --n_jobs -1
```

Then calculate results:
```
$ python -m paq.evaluation.eval_reranker --predictions my_reranker_results.jsonl --references data/annotated_datasets/NQ-open.test.jsonl
47.6%
(1699 / 3610)
```


##### Reranker Models:

The following table lists the recommended models for inference. 
For an exahustive list of models available, see [full_models_list.md](./full_models_list.md). 

| Model  | Training data |  Architecture | NQ EM | TQA EM |  Download Resource Key Name |
| ------------- |----------| --- | --------- | ---------- |---- |
|reranker_multi_base| NQ + TQA| AlBERT-base |46.0 |48.9 | `models.rerankers.reranker_multi_base`| 
|reranker_multi_large| NQ + TQA|AlBERT-large | 46.2| 49.4|`models.rerankers.reranker_multi_large`| 
|reranker_multi_xlarge| NQ + TQA|AlBERT-xlarge | 46.0| 49.1| `models.rerankers.reranker_multi_xlarge`| 
|reranker_multi_xxlarge| NQ + TQA|AlBERT-xxlarge | 47.7| 52.1 | `models.rerankers.reranker_multi_xxlarge`| 

##### ReRanker Inference:

Run QA-pair Retrieval using `paq/rerankers/rerank.py`. You can see argument help by passing `-h`. 
Pass retrieval results files of the format produced by `paq/retrievers/retrieve.py` into the `--qas_to_rerank` file.

If you have many retrieval results files to rerank, it might be useful to submit them to a cluster using `submitit` to run in parallel rather than run them one by one locally.

You can pass in a comma-separated list of retrieval results filepaths to `--qas_to_rerank` (and corresponding comma-separated list of output paths to `--output_file`) to do this, and specify the number of parallel jobs to schedule uing `--n_jobs`. To run reranking locally, pass in `--n_jobs -1`

An example of reranking the top 50 retrieved QA pairs on the NQ test set, using the ALBERT-xxlarge model running locally is shown below:
```bash
# download resources if needed:
python -m paq.download -v -n annotated_datasets.naturalquestions
python -m paq.download -v -n models.rerankers.reranker_multi_xxlarge
python -m paq.download -v -n predictions.retriever_results.multi_xlarge_nq

# run reranking
python -m paq.rerankers.rerank \
    --model_name_or_path data/models/rerankers/reranker_multi_xxlarge  \
    --qas_to_rerank data/predictions/retriever_results/multi_xlarge_nq_test.jsonl \
    --output_file my_reranker_results.jsonl \
    --top_k 50 \
    --fp16 \
    --batch_size 4 --verbose --n_jobs -1
```

##### Evaluating Rerankers:
Evalute the results of reranking using the `eval_reranker.py` file, this will return the Exact Match Score:

```
$ python -m paq.evaluation.eval_reranker --predictions my_reranker_results.jsonl --references data/annotated_datasets/NQ-open.test.jsonl
47.6%
(1699 / 3610)
```

### Question-Answer Pair Generation

The following sections details how to run the PAQ QA-Pair generation.

TL;DR for users who just want to generate QA pairs: The easiest way to generate QA-pairs is to use the [End2End Generation Tool](#end2end-generation-tool) section.

Each step in the pipeline can be run by itself, as described in the [Passage Scoring/Ranking](#passage-scoringranking), [Answer Extraction](#answer-extraction), [Question Generation](#question-generation) and [Filtering Generated QA-pairs](#filtering-generated-qa-pairs) section,
or the generation pipeline can be run fully end2end (from passages to filtered QA pairs), as described in the [End2End Generation Tool](#end2end-generation-tool) section.

Training code for training your own models is coming soon.

The pipelines have a lot of configurations and options, so to keep track of these, we use json config files to specify pipeline behaviours.
A number of example configs are listed in the `generator_configs` directory, or you can adapt them or write your own to fit your own needs.

#### Passage Scoring/Ranking

To perform passage ranking, use the `paq.generation.passage_scorer.score_passages` program, which takes as input a config json file and file of passages formatted as a tsv (passage id, passage text, passage title).

There are three passage rankers implemented:
* `DummyPassageScorer`: Applies the same score to all documents. An example config for this scorer is `generator_configs/passage_ranker_configs/dummy_passage_scorer_config.json`
* `LookupPassageScorer`: Looks up precomputed scores based on passage id (useful if you run the same passages through the pipeline a lot, and want to save compute). An example config for this scorer is `generator_configs/passage_ranker_configs/lookup_passage_scorer_config.json`
* `LearntPassageScorer`: Use a trained Passage Scorer (as done in the Paper). An example config for this scorer is `generator_configs/passage_ranker_configs/learnt_passage_scorer_config.json`

A trained passage scorer is available for download: 

| Model  | Training data |  Architecture |  Download Resource Key Name |
| ------------- |---------- |---- | ---- |
| passage_ranker_base| NQ | BERT-base |  `models.passage_rankers.passage_ranker_base`| 

Note, the original Passage ranker model used in the paper was unfortunately lost due to a storage corruption issue.
The model available here is a reproduction using the same hardware and HPs, but differs a little due to the stochastic training sampling procedure.

Below is an example to get passage scores for the the first 1000 passages of wikipedia:

```bash
# download the passage scorer model, and wikipedia text
python -m paq.download -v -n models.passage_rankers.passage_ranker_base
python -m paq.download -v -n paq.psgs_w100

# get 1000 passages to score
head -n 1000 data/paq/psgs_w100.tsv > data/paq/psgs_w100.first_1000.tsv

# run scoring
python -m paq.generation.passage_scorer.score_passages \
    --passages_to_score data/paq/psgs_w100.first_1000.tsv \
    --output_path my_passages_with_scores.jsonl \
    --path_to_config generator_configs/passage_ranker_configs/learnt_passage_scorer_config.json \
    --verbose

```

This will output a jsonl file with the following format (which is accepted by the [Answer Extraction](#answer-extraction) component below)
```json
{
  "passage_id": "ID for passage", 
  "passage": "Main text of passage.",
  "metadata": {"title": "Title of passage", "ps_score": "passage score"}
}
```

#### Answer Extraction

To perform answer extraction on passages, use the `paq.generation.answer_extractor.extract_answers` program, which takes as input a config file and passages formatted in the output format of the [Passage Scoring/Ranking](#passage-scoringranking) functionality.

There are two answer extractors implemented:
* `SpacyNERExtractor`: This answer extractor will extract named entities from passages as answers (as used in PAQ-NE). An example config for this extractor is `generator_configs/answer_extractor_configs/named_entity_answer_extractor_config.json` 
* `Span2DAnswerExtractor`: This answer extractor uses a learnt answer span extractor to extract answers (as used in PAQ-L).  An example config for this extractor is `generator_configs/answer_extractor_configs/learnt_answer_extractor_config.json`

The learnt answer span extractor model used in the paper is available for download:


| Model  | Description | Training data |  Architecture |  Download Resource Key Name |
| ----------| --- |---------- |---- | ---- |
| answer_extractor_nq_base| Learnt Answer Span Extractor, BERT-base, NQ-trained | NQ | BERT-base |  `models.answer_extractors.answer_extractor_nq_base`| 


Below is an example to extract answers from passages, using the learnt extractor:

```bash

# download the span extractor model:
python -m paq.download -v -n models.answer_extractors.answer_extractor_nq_base

# run answer extraction
python -m paq.generation.answer_extractor.extract_answers \
    --passages_to_extract_from my_passages_with_scores.jsonl \
    --output_path my_pasages_with_answers.jsonl \
    --path_to_config generator_configs/answer_extractor_configs/learnt_answer_extractor_config.json \
    --verbose
```
This will output a jsonl file with the following format (which is accepted by the [Question Generation](#question-generation) component below)

```json
{
  "passage_id": "ID for passage", 
  "passage": "Main text of passage.",
  "metadata": {"title": "Title of passage", "ps_score": "passage score"},
  "answers": [{"text": "Main", "start": 0, "end": 5, "score": "score for answer"}, {"text": "passage", "start": 13, "end": 20, "score": "score for answer"}]
}
```

#### Question Generation

To perform Question Generation on passages with extracted answers, use the `paq.generation.question_generator.generate_questions` program, which takes as input a config file and passages with answers formatted in the output format of the [Answer Extraction](#answer-extraction) functionality. 

An example config for question generation can be found here: `generator_configs/question_generator_configs/question_generation_config.json`

The following trained question generators are available:

| Model  | Training data |  Architecture |  Download Resource Key Name |
| ------------- |---------- |---- | ---- |
| qgen_nq_base| NQ | BART-base |  `models.qgen.qgen_nq_base`| 
| qgen_multi_base| Multitask | BART-base |  `models.qgen.qgen_multi_base`| 


Below is an example to generate questions from passages with extracted answers, using the multitask generator:

```bash

# download the qgen model:
python -m paq.download -v -n models.qgen.qgen_multi_base

# run question generation extraction
python -m paq.generation.question_generator.generate_questions \
    --passage_answer_pairs_to_generate_from my_pasages_with_answers.jsonl \
    --output_path my_generated_questions.jsonl \
    --path_to_config generator_configs/question_generator_configs/question_generation_config.json \
    --verbose
```

This will output a jsonl file with the following format (which is accepted by the [Filtering Generated QA-pairs](#filtering-generated-qa-pairs) component below)

```json
{
  "passage_id": "ID for passage", 
  "answer": "Benedict", 
  "question": "which pope has the middle name gregory",
  "metadata": {"answer_start": 617, "answer_end": 625, "ae_score": "score for answer", "qg_score": "currently not implemented, but score for question can go here"}
}
```

#### Filtering Generated QA-pairs

Generated questions can be inconsistent, or poor quality, or overly ambiguous. 
Empirically, we find it important to filter the generated questions for answer consistency. 
To perform filtering on generated questions, use the `paq.generation.filtering.filter_questions` program, which takes as input a config file, and generated questions formatted in the output format of the [Question Generation](#question-generation) functionality.

Filtering is split into two parts: retrieval and reading. 
The retriever retrieves passages from a corpus using the generated question, and the reader reads the passages and computes an answer.

We have implemented the following filterers:
* *Dummy filtering*: uses a `DummyFilteringRetriever` and `DummyReader`, assigns all answers as consistent. An example config is `generator_configs/filterer_configs/dummy_filtering_config.json` 
* *Local filtering* (fast but not as good): essentially performs reading comprehension. uses a `LocalFilteringRetriever` to "retrieve" the passage the question was generated from. The reader (`FiDReader`) generates an answer using only this single gold passage. We use FID supplied with a single passage as the reader, which worked as well as standard readers in our experiments. An example config is `generator_configs/filterer_configs/local_filtering_config.json`.
* *Global Filtering* (slow but important for strong performance): Uses A `GlobalFilteringRetriever` to retrieve relevant passages for the question (this uses DPR under the hood). The reader is a `FiDReader`, (this is FID under the hood). An example config is `generator_configs/filterer_configs/global_filtering_config.json`

The following trained models are available for download:

| Model  | Description | Training data |  Architecture |  Download Resource Key Name |
| ----------| --- |---------- |---- | ---- |
| dpr_nq_passage_retriever| DPR Passage retriever and faiss index, from the DPR Paper, used for retrieving passage for the reader in global filtering, NQ-trained| NQ | BERT-base |  `models.filtering.dpr_nq_passage_retriever`| 
| fid_reader_nq_base| FID-base reader, from the Fusion-in-Decoder paper, used in global and local filtering, NQ-trained | NQ | t5-base |  `models.filtering.fid_reader_nq_base`| 

Below is an example of how to filter questions (both with local and global filtering):

```bash
# download the corpus to retrieve from, the DPR retriever and the reader:
python -m paq.download -v -n paq.psgs_w100
python -m paq.download -v -n models.filtering.dpr_nq_passage_retriever
python -m paq.download -v -n models.filtering.fid_reader_nq_base

# run filtering using local filtering...
python -m paq.generation.filtering.filter_questions \
    --generated_questions_to_filter my_generated_questions.jsonl \
    --output_path my_locally_filtered_questions.jsonl \
    --path_to_config generator_configs/filterer_configs/local_filtering_config.json \
    --verbose

# or, run filtering using global filtering 
python -m paq.generation.filtering.filter_questions \
    --generated_questions_to_filter my_generated_questions.jsonl \
    --output_path my_globally_filtered_questions.jsonl \
    --path_to_config generator_configs/filterer_configs/global_filtering_config.json \
    --verbose

```

This will output a jsonl file with the following format:

```json
{
  "passage_id": "ID for passage", 
  "answer": "Benedict", 
  "question": "which pope has the middle name gregory",
  "metadata": {"filter_answer": "benedict", "consistent": true, "answer_start": 617, "answer_end": 625, "ae_score": "score for answer", "qg_score": "currently not implemented, but score for question can go here"}
}
```

#### End2End Generation Tool

To run all the steps in the pipeline end2end, use the `paq.generation.generate_qa_pairs` program.
This will run passage ranking, then answer extraction, then generation, then finally filtering automatically.

The tool takes as input a config json file, and a file passages to generate QA pairs from, formatted as a tsv (passage id, passage text, passage title).
The tool will create out output directory, and write intermediate results to it, including the final generated QA-pairs in the `final_qas.jsonl` file.

The following example configs can be used with this tool to replicate the generation pipelines used in the paper:
* `generator_configs/paq_L1_config.json`: run a generation pipeline replicating PAQ-L1
* `generator_configs/paq_L4_config.json`: run a generation pipeline replicating PAQ-L4
* `generator_configs/paq_NE_config.json`: run a generation pipeline replicating PAQ-NE
* `generator_configs/paq_L1_local_filtering_config.json`: run a generation pipeline replicating PAQ-L1, but with local rather than global filtering.

Or, write your own config to fit your generation needs.

The following code will run the PAQ-L1 generation pipeline on the first 1000 passages in the preprocssed wikipedia dump:

```bash
# Download the models and data we need:
python -m paq.download -v -n models.passage_rankers.passage_ranker_base
python -m paq.download -v -n models.answer_extractors.answer_extractor_nq_base
python -m paq.download -v -n models.qgen.qgen_multi_base
python -m paq.download -v -n paq.psgs_w100
python -m paq.download -v -n models.filtering.dpr_nq_passage_retriever
python -m paq.download -v -n models.filtering.fid_reader_nq_base

head -n 1000 data/paq/psgs_w100.tsv > data/paq/psgs_w100.first_1000.tsv

python -m paq.generation.generate_qa_pairs \
    --passage_files_to_generate data/paq/psgs_w100.first_1000.tsv \
    --output_dirs my_generated_qas \
    --path_to_config generator_configs/paq_L1_config.json\
    --verbose --n_jobs -1
```

The paq.generation.generate_qa_pairs can use submitit to run generation on a cluster.
The `--n_jobs` flag indicates how many concurrent submitit jobs to submit, use --n_jobs -1 to run locally.
To run generation in several jobs in parallel, you can pass in a comma-separated list of input files to `--passage_files_to_generate`
 and a corresponding comma separated list of output directories to create.



## Citing

To cite this work, please use the following bibtex:
```
@article{lewis2021paq,
      title={PAQ: 65 Million Probably-Asked Questions and What You Can Do With Them}, 
      author={Patrick Lewis and Yuxiang Wu and Linqing Liu and Pasquale Minervini and Heinrich Küttler and Aleksandra Piktus and Pontus Stenetorp and Sebastian Riedel},
      year={2021},
      eprint={2102.07033},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## LICENSE

### Code License:

The majority of the PAQ code is licensed under [CC-BY-NC](./LICENSE), however portions of the project are available under separate license terms: HuggingFace Transformers is licensed under Apache License 2.0; spaCy and wandb are licensed under the MIT License.
The code in this repository is licenced according the [LICENSE](./LICENSE) file.

### Data License:

The PAQ QA-pairs and metadata is licensed under [CC-BY-SA](https://creativecommons.org/licenses/by-sa/3.0/). 
Other data is licensed according to the accompanying license files.
