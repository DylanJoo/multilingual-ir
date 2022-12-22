# multilingual-ir
Experimental multilingual IR repo for WSDM MIRACL.

# Dataset
- MIRACL
Use the huggingface API to access data. 
ISO language codes: 'ar', 'bn', 'en', 'es', 'fa', 'fi', 'fr', 'hi', 'id', 'ja', 'ko', 'ru', 'sw', 'te', 'th', 'zh'. 
Check [MIRACL in huggingface](https://huggingface.co/datasets/miracl/miracl).

* queryid, query, positive passages and negative passages
```
dataset=load_dataset('miracl/miracl', f"{lang}")
qid=dataset['train']['query_id]
qtext=dataset['train']['query']
neg_p=dataset['train']['positive_passages']
pos_p=dataset['train']['negative_passages']
```

# Requirements
- pyserini
```
(bash) export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
```

# Baseline
- BM25 sparse retrieval
    * prebuilt miracl indexes from pyserini, 16 languges
    ```
    from pyserini.search.lucene import LuceneSearcher
    LuceneSearcher.from_prebuilt_index(f'miracl-v1.0-f{lang}')
    ```
    * Retrieval
    ```

    ```

