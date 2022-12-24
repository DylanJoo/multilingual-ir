import os
from datasets import load_dataset
# from datasets import disable_caching
# disable_caching()

LANGUAGES = [
    "arabic",
    "chinese",
    # "dutch",
    # "english",
    "french",
    # "german",
    "hindi",
    "indonesian",
    # "italian",
    "japanese",
    # "portuguese",
    "russian",
    "spanish",
    # "vietnamese",
]

for lang in LANGUAGES:
    # dataset = load_dataset('unicamp-dl/mmarco', lang, keep_in_memory=True)
    dataset = load_dataset('unicamp-dl/mmarco', lang)
    dataset['train'].select(range(400000)).to_json(f'mmarco.400k.{lang}.json', num_proc=10)
    cmd = f'rm -rvf /tmp2/jhju/huggingface/datasets/unicamp-dl___m_marco/${lang}'
    os.system(cmd)


