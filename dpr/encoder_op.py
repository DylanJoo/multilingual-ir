import torch
import torch.nn as nn
if torch.cuda.is_available():
    from torch.cuda.amp import autocast
from transformers import BertModel, AutoTokenizer
from loss import InBatchNegativeCELoss, InBatchKLLoss
from typing import Optional

class BiEncoderForRelevanceTransfer(nn.Module):
    def __init__(self, model_name: str, tokenizer_name=None, device='cuda', freeze_document_encoder=False, pooling='cls'):
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

        self.pooling = pooling

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
        q_outputs = self.query_encoder(**q_inputs)
        if self.pooling == 'mean':
            d_embeddings = self._mean_pooling(d_outputs.last_hidden_state[:, 1:, :], d_inputs['attention_mask'][:, 1:])
            q_embeddings = self._mean_pooling(q_outputs.last_hidden_state[:, 1:, :], q_inputs['attention_mask'][:, 1:])
        else:
            d_embeddings = d_outputs.last_hidden_state[:, 0, :]
            q_embeddings = q_outputs.last_hidden_state[:, 0, :]

        # OBJ1: separate the positive and negative:w
        relevance_embeddings = d_embeddings * q_embeddings # Lang*B H

        # language relevance transfer 
        language_border = relevance_embeddings.size(0) // 2
        rich_lang_rel_embeddings = relevance_embeddings[:language_border, :] # B H
        low_lang_rel_embeddings = relevance_embeddings[language_border:, :]  # B H
        lang_rel_cosine = rich_lang_rel_embeddings @ low_lang_rel_embeddings.T # B B 
        loss_rel_xfer = InBatchNegativeCELoss(lang_rel_cosine)

        # constrastive learning with in-batch negative training
        ## [TODO] add hard negative 
        ## [TODO] add knowledge distilation
        ranking_cosine_rich = q_embeddings[:ranking_border, :] @ d_embeddings[:language_border, :].T # Lang*B Lang*B
        ranking_cosine_low = q_embeddings[language_border:, :] @ d_embeddings[language_border:, :].T # Lang*B Lang*B
        loss_rank_rich = InBatchNegativeCELoss(ranking_cosine_rich)
        loss_rank_low = InBatchNegativeCELoss(ranking_cosine_low)
        loss_rank_xfer = InBatchKLLoss(ranking_cosine_low, ranking_cosine_rich)
        loss = loss_rel_xfer + loss_rank_rich + loss_rank_low + loss_rank_xfer

        return {'loss': loss, 'score': torch.diag(ranking_cosine_rich)}

