# PAQ: 65 Million Probably-Asked Questions and What You Can Do With Them


This repository contains code and models to support the research paper [PAQ: 65 Million Probably-Asked Questions and What You Can Do With Them](https://arxiv.org/abs/2102.07033)

<br>
<p align="center">
  <img src="https://dl.fbaipublicfiles.com/MLQA/logos.png" alt="Facebook AI Research and UCL NLP"  width="60%"/>
  <br>
</p>
<br>

## Data

The PAQ QA pairs can be downloaded below. We use the same format as for NQ-open (available [here](https://github.com/google-research-datasets/natural-questions/tree/master/nq_open)):

| Dataset  | # QAs | Size (unzipped)| link | License |
| ------------- | ------------- | --------- | ---- | -----|
| PAQ     | 64.9M   | 5.8 GB| [download](https://dl.fbaipublicfiles.com/paq/v1/PAQ.tar.gz) |  [CC-BY-SA](https://creativecommons.org/licenses/by-sa/3.0/)|
| PAQ-L1  | 14.1M   | 1.3 GB| [download](https://dl.fbaipublicfiles.com/paq/v1/PAQ_L1.tar.gz) | [CC-BY-SA](https://creativecommons.org/licenses/by-sa/3.0/)|
| PAQ-L4  |  53.8M  | 4.9 GB| [download](https://dl.fbaipublicfiles.com/paq/v1/PAQ_L4.tar.gz) | [CC-BY-SA](https://creativecommons.org/licenses/by-sa/3.0/)|
| PAQ-NE1 | 12.0M   | 1.0 GB| [download](https://dl.fbaipublicfiles.com/paq/v1/PAQ_NE1.tar.gz) | [CC-BY-SA](https://creativecommons.org/licenses/by-sa/3.0/)|


### Metadata:

Available metadata to support PAQ is available, and can be downloaded from the following table. See the descriptions below for details:

| Dataset  | Size (unzipped)| link | License |
| ------------- | ------------- | --------- |  ----|
| Preprocessed Wikipedia Dump   | 13 GB| [download](https://dl.fbaipublicfiles.com/paq/v1/psgs_w100.tsv.gz) | [CC-BY-SA](https://creativecommons.org/licenses/by-sa/3.0/)|
| Passage Selector Scores  | 560 MB| [download](https://dl.fbaipublicfiles.com/paq/v1/PASSAGE_SCORES.tar.gz) | [CC-BY-SA](https://creativecommons.org/licenses/by-sa/3.0/)|
| PAQ QA-Pair metadata  |  16 GB| [download](https://dl.fbaipublicfiles.com/paq/v1/PAQ.metadata.jsonl.gz) | [CC-BY-SA](https://creativecommons.org/licenses/by-sa/3.0/)|
| PAQ *unfiltered* QA-pairs and metadata | 95 GB| [download](https://dl.fbaipublicfiles.com/paq/v1/PAQ.unfiltered_metadata.jsonl.gz) | [CC-BY-SA](https://creativecommons.org/licenses/by-sa/3.0/)|

#### Preprocessed Wikipedia Dump

This file contains the preprocessed wikipedia dump used to generate PAQ. The file consists of 100-word passages of a 2018 Wikipedia dump, and was produced by [Karphukin et al.](https://github.com/facebookresearch/DPR) for [DPR](https://github.com/facebookresearch/DPR).
The file is in TSV format, with 3 columns. The first column is passage id, the second column is the passage text, the third is the wikipedia article title. 

#### Passage Selector Scores

This file contains the passage selection scores for passages, using the passage selection model described in the paper.
The file is in TSV format, with 2 columns. The first column is passage id (see "Preprocessed Wikipedia Dump"), the second column is the logprob score from the passage selector for that passage.

#### PAQ QA pair metadata

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

#### PAQ *unfiltered* QA-pair metadata 

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


## Code

Code to run experiments will be uploaded soon.

## Models

Models will be uploaded soon.


## Citation

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

### Code:

The majority of the PAQ code is licensed under [CC-BY-NC](./LICENSE), however portions of the project are available under separate license terms: HuggingFace Transformers is licensed under Apache License 2.0; spaCy and wandb are licensed under the MIT License.
The code in this repository is licenced according the [LICENSE](./LICENSE) file.

### Data:

The PAQ data is licensed under [CC-BY-SA](https://creativecommons.org/licenses/by-sa/3.0/)
