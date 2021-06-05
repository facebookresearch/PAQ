# Full List of Models Available for Download

## BiEncoder Retrievers


| Model  | Training data |  Architecture | Embedding Dim | NQ EM | + rerank | TQA EM | + rerank |  Download Resource Key Name |
| ------------- |----------| --- | --------- | ---------- |---- |---- | ---- | ---- |
| retriever_multi_base_256  (recommended)| NQ + TQA | AlBERT-base | 256  | 41.4 | 47.3 | 40.2 | 50.9| `models.retrievers.retriever_multi_base_256` |
| retriever_multi_base | NQ + TQA | AlBERT-base  | 728 | 40.9| 47.4 | 39.7 | 51.2 | `models.retrievers.retriever_multi_base`  |
| retriever_multi_large | NQ + TQA | AlBERT-large | 728 | 41.2 | 47.5 | 41.0| 51.9 |`models.retrievers.retriever_multi_large`|
| retriever_multi_xlarge | NQ + TQA | AlBERT-xlarge| 728  | 41.7 | 47.6 | 41.3 | 52.1 |`models.retrievers.retriever_multi_xlarge`|
| retriever_nq_base | NQ | AlBERT-base | 728 | 41.0 | 47.2 |35.6 | 49.0 |`models.retrievers.retriever_nq_base`|
| retriever_nq_large | NQ | AlBERT-large | 728 | 40.4 | 47.3| 34.1|48.1 |`models.retrievers.retriever_nq_large`|
| retriever_nq_xlarge | NQ | AlBERT-xlarge | 728 | 41.1 |47.7 | 35.7| 48.9|`models.retrievers.retriever_nq_xlarge`|
| retriever_tqa_base | TQA | AlBERT-base | 728 | 37.5| 46.8 | 38.7| 51.0| `models.retrievers.retriever_tqa_base`|
| retriever_tqa_large | TQA | AlBERT-large |  728 | 38.2| 47.0| 39.6|51.4 |`models.retrievers.retriever_tqa_large`|
| retriever_tqa_xlarge | TQA | AlBERT-xlarge | 728 | 38.0| 46.5 | 38.9|51.2 |`models.retrievers.retriever_tqa_xlarge`|

(Rerank scores calculated with `reranker_multi_xxlarge`)

## QA Rerankers


| Model  | Training data |  Architecture | NQ EM | TQA EM |  Download Resource Key Name |
| ------------- |----------| --- | --------- | ---------- |---- |
|reranker_multi_base| NQ + TQA| AlBERT-base |46.0 |48.9 | `models.rerankers.reranker_multi_base`| 
|reranker_multi_large| NQ + TQA|AlBERT-large | 46.2| 49.4|`models.rerankers.reranker_multi_large`| 
|reranker_multi_xlarge| NQ + TQA|AlBERT-xlarge | 46.0| 49.1| `models.rerankers.reranker_multi_xlarge`| 
|reranker_multi_xxlarge| NQ + TQA|AlBERT-xxlarge | 47.7| 52.1 | `models.rerankers.reranker_multi_xxlarge`| 
|reranker_nq_xlarge| NQ | AlBERT-xlarge | 45.2| 46.7 | `models.rerankers.reranker_nq_xlarge`| 
|reranker_nq_xxlarge| NQ| AlBERT-xxlarge |46.4 | 49.6| `models.rerankers.reranker_nq_xxlarge`| 
|reranker_tqa_xlarge| TQA | AlBERT-xlarge | 45.0|49.7 | `models.rerankers.reranker_tqa_xlarge`| 
|reranker_tqa_xxlarge| TQA | AlBERT-xxlarge | 46.0|51.7 | `models.rerankers.reranker_tqa_xxlarge`| 

(EM scores in this table calculated using  `retriever_multi_xlarge` retriever)

## Qgen Models

| Model  | Training data |  Architecture |  Download Resource Key Name |
| ------------- |---------- |---- | ---- |
| qgen_nq_base| NQ | BART-base |  `models.qgen.qgen_nq_base`| 
| qgen_multi_base| Multitask | BART-base |  `models.qgen.qgen_multi_base`| 


## Passage Ranker Models

Models used for selecting passages to generate questions from:

| Model  | Training data |  Architecture |  Download Resource Key Name |
| ------------- |---------- |---- | ---- |
| passage_ranker_base| NQ | BERT-base |  `models.passage_rankers.passage_ranker_base`| 

Note, the original Passage ranker model used in the paper was unfortunately lost due to a storage corruption issue.
The model here is a reproduction using the same hardware and HPs, but differs a little due to the stochastic training sampling procedure.

## Answer Extractor Models


| Model  | Description | Training data |  Architecture |  Download Resource Key Name |
| ----------| --- |---------- |---- | ---- |
| answer_extractor_nq_base| Learnt Answer Span Extractor, BERT-base, NQ-trained | NQ | BERT-base |  `models.answer_extractors.answer_extractor_nq_base`| 

## Filterer Models

| Model  | Description | Training data |  Architecture |  Download Resource Key Name |
| ----------| --- |---------- |---- | ---- |
| dpr_nq_passage_retriever| DPR Passage retriever and faiss index, from the DPR Paper, used for retrieving passage for the reader in global filtering, NQ-trained| NQ | BERT-base |  `models.filtering.dpr_nq_passage_retriever`| 
| fid_reader_nq_base| FID-base reader, from the Fusion-in-Decoder paper, used in global and local filtering, NQ-trained | NQ | t5-base |  `models.filtering.fid_reader_nq_base`| 
