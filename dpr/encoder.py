import torch
import torch.nn as nn
if torch.cuda.is_available():
    from torch.cuda.amp import autocast
from transformers import BertModel, AutoTokenizer
from loss import InBatchNegativeCELoss
from typing import Optional

class BiEncoderForRelevanceTransfer(nn.Module):
    def __init__(self, model_name: str, tokenizer_name=None, device='cuda', freeze_document_encoder=False):
        super().__init__()
        self.device = device
        self.query_encoder = BertModel.from_pretrained(model_name, add_pooling_layer=False)
        self.query_encoder.to(self.device)
        self.document_encoder = BertModel.from_pretrained(model_name, add_pooling_layer=False)
        self.document_encoder.to(self.device)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name)
        except:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name, use_fast=False)
        # configuration
        if freeze_document_encoder:
            for p in self.document_encoder.parameters():
                p.requires_grad = False

        # required for huggingface trainer
        # self.name_

    @staticmethod
    def _mean_pooling(last_hidden_state, attention_mask):
        token_embeddings = last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def forward(self, 
                q_inputs, 
                d_inputs,
                labels: Optional[torch.Tensor] = None,
                keep_d_dims: bool = True,
                **kwargs):
        # Sentence embeddings 
        d_outputs = self.document_encoder(**d_inputs)
        d_embeddings = self._mean_pooling(d_outputs.last_hidden_state[:, 1:, :], d_inputs['attention_mask'][:, 1:])
        q_outputs = self.query_encoder(**q_inputs)

        q_embeddings = self._mean_pooling(q_outputs.last_hidden_state[:, 1:, :], q_inputs['attention_mask'][:, 1:])
        q_outputs = self.query_encoder(**q_inputs)
        relevance_embeddings = d_embeddings * q_embeddings # Lang*B H

        # language relevance transfer 
        language_border = relevance_embeddings.size(0) // 2
        rich_lang_rel_embeddings = relevance_embeddings[:language_border, :] # B H
        low_lang_rel_embeddings = relevance_embeddings[language_border:, :]  # B H
        lang_rel_cosine = rich_lang_rel_embeddings @ low_lang_rel_embeddings.permute(1, 0) # B B 
        loss_xfer = InBatchNegativeCELoss(lang_rel_cosine)

        # constrastive learning with in-batch negative training
        ## [TODO] add hard negative 
        ## [TODO] add knowledge distilation
        ranking_cosine = d_embeddings @ q_embeddings.permute(1, 0) # Lang*B Lang*B
        loss_rank = InBatchNegativeCELoss(ranking_cosine)

        loss = loss_xfer + loss_rank
        # print('loss:', loss.detach().cpu().numpy(), 
        #       'loss-xfer:', loss_xfer.detach().cpu().numpy(), 
        #       'loss-rank', loss_rank.detach().cpu().numpy())

        return {'loss': loss, 'score': torch.diag(ranking_cosine)}


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

