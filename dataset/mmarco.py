from datasets import load_dataset

# fetch dataset and cached
for lang in ['arabic', 'bengali', 'english', 'indonesian', 'finnish', 'korean', 'russian', 'swahili', 'telugu', 'thai', 'japanese', 'combined']:
    dataset = load_dataset('castorini/mr-tydi', lang, 'train')
