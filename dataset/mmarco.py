import os
import collections
from datasets import load_dataset

# run the following command to fetch mmarco 
LANGUAGES = {
        "arabic": 'ar', 
        "chinese": 'zh', 
        "french": 'fr', 
        "hindi": 'hi', 
        "indonesian": 'id', 
        "japanese": 'ja', 
        "russian": 'ru', 
        "spanish": 'es',
    # "dutch",
    # "english",
    # "german",
    # "italian",
    # "portuguese",
    # "vietnamese",
}

def download_dataset():
    for lang in LANGUAGES:
        dataset = load_dataset('unicamp-dl/mmarco', lang)
        dataset['train'].select(range(400000)).to_json(f'mmarco.400k.{lang}.json', num_proc=10)
        cmd = f'rm -rvf /tmp2/jhju/huggingface/datasets/unicamp-dl___m_marco/${lang}'
        os.system(cmd)

def join_dataset(english_file, mixing_type='english-x', do_eval=False):

    import random
    import numpy as np

    # rich language
    dataset = load_dataset('json', data_files=english_file, keep_in_memory=True)['train']
    # low language
    dataset_low = {}
    for lang in LANGUAGES:
        file_name = english_file.replace('english', lang)
        dataset_low[lang] = load_dataset('json', data_files=file_name, keep_in_memory=True)['train']

    # random sampling parallel dataset
    languages = [l for l in LANGUAGES if l != 'english']
    dataset_mixing = collections.defaultdict(list)
    for lang, lang_split_indices in zip(
            languages, np.array_split(range(len(dataset)), len(languages))
        ):
        dataset_lang_split = dataset_low[lang][lang_split_indices]
        dataset_mixing['query'] += dataset_lang_split['query']
        dataset_mixing['positive'] += dataset_lang_split['positive']
        dataset_mixing['negative'] += dataset_lang_split['negative']
        dataset_mixing['lang'] += [lang] * len(lang_split_indices)

    for column in dataset_mixing:
        dataset = dataset.add_column(f"{column}_low", dataset_mixing[column])

    dataset = dataset.class_encode_column('lang_low')
    dataset = dataset.train_test_split(test_size=0.01, stratify_by_column='lang_low')

    return dataset
