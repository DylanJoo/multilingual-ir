import numpy as np
import torch
if torch.cuda.is_available():
    from torch.cuda.amp import autocast
from transformers import BertModel, BertTokenizer, BertTokenizerFast

from pyserini.encode import DocumentEncoder, QueryEncoder


class BertEncoder(DocumentEncoder):
    def __init__(self, model_name: str, tokenizer_name=None, device='cuda:0'):
        self.device = device
        self.model = BertModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name or model_name)

    def encode(self, texts=None, inputs=None, titles=None, fp16=False,  max_length=512, **kwargs):
        # if input the raw text 
        if inputs is None:
            if titles is not None:
                texts = [f'[CLS] {title} {text}' for title, text in zip(titles, texts)]
            else:
                texts = ['[CLS] ' + text for text in texts]
            inputs = self.tokenizer(
                texts,
                max_length=max_length,
                padding="longest",
                truncation=True,
                add_special_tokens=False,
                return_tensors='pt'
            )

        inputs.to(self.device)
        if fp16:
            with autocast():
                with torch.no_grad():
                    outputs = self.model(**inputs)
        else:
            outputs = self.model(**inputs)

        embeddings = self._mean_pooling(outputs["last_hidden_state"][:, 1:, :], inputs['attention_mask'][:, 1:])
        if kwargs.pop('return_tensors', False):
            return embeddings
        else:
            return embeddings.detach().cpu().numpy()

