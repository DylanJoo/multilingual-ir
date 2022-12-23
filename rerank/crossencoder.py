import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertForSequenceClassification
from transformers import AutoTokenizer, MT5ForConditionalGeneration

# point-wise reranker
class monoMBERT(BertForSequenceClassification):
    softmax = nn.Softmax(dim=1)

    def predict(self, batch):
        for k in batch:
            batch[k] = batch[k].to(self.device)

        batch_logits = self.forward(**batch, return_dict=False).logits
        return self.softmax(batch_logits[:, 1]).detach().cpu().numpy()

    def set(self, device='cpu', tokenizer_name=None):
        self.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or self.name_or_path)

class monoMT5(MT5ForConditionalGeneration):
    softmax = nn.Softmax(dim=1)

    def predict(self, batch):
        for k in batch:
            batch[k] = batch[k].to(self.device)

        # Perpare BOS labels
        dummy_labels = torch.full(
                batch['input_ids'].size(), 
                self.config.decoder_start_token_id
        ).to(self.device)

        batch_logits = self.forward(**batch, labels=dummy_labels).logits
        return self.softmax(batch_logits[:, 0, self.targeted_ids]).detach().cpu().numpy()

    def set(self, device='cpu', tokenizer_name=None, targeted_tokens=['yes', 'no']):
        self.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or self.name_or_path)
        tokenized_tokens = self.tokenizer(targeted_tokens, add_special_tokens=False)
        self.targeted_ids = [x for xs in tokenized_tokens.input_ids for x in xs]
