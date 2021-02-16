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

| Dataset  | # QAs | Size (unzipped)| link |
| ------------- | ------------- | --------- | ---- |
| PAQ     | 64.9M   | 5.8 GB| [download](https://dl.fbaipublicfiles.com/paq/v1/PAQ.tar.gz) |
| PAQ-L1  | 14.1M   | 1.3 GB| [download](https://dl.fbaipublicfiles.com/paq/v1/PAQ_L1.tar.gz) |
| PAQ-L4  |  53.8M  | 4.9 GB| [download](https://dl.fbaipublicfiles.com/paq/v1/PAQ_L4.tar.gz) |
| PAQ-NE1 | 12.0M   | 1.0 GB| [download](https://dl.fbaipublicfiles.com/paq/v1/PAQ_NE1.tar.gz) |


Metadata for the PAQ QA-pairs are coming soon.

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

The PAQ dataset is licensed under [CC-BY-SA](https://creativecommons.org/licenses/by-sa/3.0/)
