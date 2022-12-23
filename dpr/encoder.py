import torch
if torch.cuda.is_available():
    from torch.cuda.amp import autocast
from transformers import BertModel, BertTokenizer, BertTokenizerFast

class BertEncoder:
    def __init__(self, model_name: str, tokenizer_name=None, device='cuda:0'):
        self.device = device
        self.model = BertModel.from_pretrained(model_name, add_pooling_layer=False)
        self.model.to(self.device)
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name or model_name)

    @staticmethod
    def _mean_pooling(last_hidden_state, attention_mask):
        token_embeddings = last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(self, texts=None, inputs=None, titles=None, fp16=False, max_length=512, **kwargs):
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
        else:
            for k in inputs:
                inputs[k] = inputs[k].to(self.device)
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

    def forward(self, ):

