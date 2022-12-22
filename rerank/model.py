import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Tokenizer, MT5ForConditionalGeneration
from transformers import BertForSequenceClassification, BertModel

# point-wise reranker
class monoBERT(BertForSequenceClassification):
    """
    init ckpt: bert-base-multilingual-cased
    fine-tuninng required.
    """
    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
    device = 'cpu'
    softmax = nn.Softmax(dim=1)

    def predict(self, batch):
        for k in batch:
            batch[k] = batch[k].to(self.device)

        batch_logits = self.forward(**batch, return_dict=False).logits
        return self.softmax(batch_logits[:, 1]).detach().cpu().numpy()

class monoMT5(MT5ForConditionalGeneration):
    targeted_tokens = ['yes', 'no']
    device = 'cpu'
    softmax = nn.Softmax(dim=1)

    def predict(self, batch):
        # Perpare BOS labels
        for k in batch:
            batch[k] = batch[k].to(self.device)

        dummy_labels = torch.full(
                batch.input_ids.size(), 
                self.config.decoder_start_token_id
        ).to(self.device)

        batch_logits = self.forward(**batch, labels=dummy_labels).logits
        return self.softmax(batch_logits[:, 0, self.targeted_ids]).detach().cpu().numpy()

