import os
from huggingface_hub import hf_hub_download

# run the following command to fetch mmarco 
LANGUAGES = {
        'train': ['ar', 'bn', 'en', 'es', 'fa', 'fi', 'fr', 'hi', 'id', 'ja', 'ko', 'ru', 'sw', 'te', 'th', 'zh'],
        'dev': ['ar', 'bn', 'en', 'es', 'fa', 'fi', 'fr', 'hi', 'id', 'ja', 'ko', 'ru', 'sw', 'te', 'th', 'zh'],
        'testA': ['ar', 'bn', 'en', 'fi', 'id', 'ja', 'ko', 'ru', 'sw', 'te', 'th']
}
# /home/jhju/.cache/huggingface/hub/ar.tsv

# download MIRACL testA topic files
for lang in LANGUAGES['testA']:
    hf_hub_download('miracl/miracl', 
            filename=f'topics.miracl-v1.0-{lang}-test-a.tsv', 
            subfolder=f'miracl-v1.0-{lang}/topics', 
            repo_type='dataset', 
            cache_dir='/tmp2/jhju/multilingual-ir/dataset/',
            force_filename=f'topics.miracl-v1.0-{lang}-test-a.tsv')

    cmd = f'rm -rvf /tmp2/jhju/multilingual-ir/dataset/*lock'
    os.system(cmd)


