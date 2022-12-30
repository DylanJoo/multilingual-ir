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
                qr_inputs, dr_inputs, ql_inputs, dl_inputs,
                labels: Optional[torch.Tensor] = None,
                keep_d_dims: bool = True,
                **kwargs):

        # Sentence embeddings 
        qr_outputs = self.query_encoder(**qr_inputs)
        dr_outputs = self.document_encoder(**dr_inputs)
        ql_outputs = self.query_encoder(**ql_inputs)
        dl_outputs = self.document_encoder(**dl_inputs)

        if self.pooling == 'mean':
            qr_embeddings = self._mean_pooling(qr_outputs.last_hidden_state[:, 1:, :], qr_inputs['attention_mask'][:, 1:])
            dr_embeddings = self._mean_pooling(dr_outputs.last_hidden_state[:, 1:, :], dr_inputs['attention_mask'][:, 1:])
            ql_embeddings = self._mean_pooling(ql_outputs.last_hidden_state[:, 1:, :], ql_inputs['attention_mask'][:, 1:])
            dl_embeddings = self._mean_pooling(d1_outputs.last_hidden_state[:, 1:, :], d1_inputs['attention_mask'][:, 1:])
        else:
            qr_embeddings = qr_outputs.last_hidden_state[:, 0, :]
            dr_embeddings = dr_outputs.last_hidden_state[:, 0, :]
            ql_embeddings = ql_outputs.last_hidden_state[:, 0, :]
            dl_embeddings = dl_outputs.last_hidden_state[:, 0, :]

        # OBJ1: separate the positive and negative
        rich_lang_rel_embeddings = qr_embeddings * dr_embeddings # Bx2 H
        low_lang_rel_embeddings = ql_embeddings * dl_embeddings # Bx2 H
        positive_border = qr_embeddings.size(0) // 2 # B//2
        ## setting 0: only use positive relevance constrastive learning # B(en) B(other)
        # lang_rel_cosine = rich_lang_rel_embeddings[:positive_border, :] @ low_lang_rel_embeddings[:positive_border, :].T 
        ## setting 1: add soft-negative relevance constrastive # B(en-pos) Bx2(other-pos;other-neg)
        lang_rel_cosine = rich_lang_rel_embeddings[:positive_border, :] @ low_lang_rel_embeddings.T # B B 
        ## setting 2: add hard-negative relevance constrastive # B(en) Bx3(other-pos;other-neg;en-neg)
        # hard_negative = torch.concat((low_lang_rel_embeddings, rich_lang_rel_embeddings[positive_border:, :]), 0)
        # lang_rel_cosine = rich_lang_rel_embeddings[:positive_border, :] @ hard_negative.T # B B 

        loss_rel_xfer = InBatchNegativeCELoss(lang_rel_cosine)

        # OBJ2: separate the positive and negative
        # constrastive learning with in-batch negative training
        ranking_cosine_rich = qr_embeddings @ d_embeddings.T # B Bx2
        ranking_cosine_low = qr_embeddings @ d_embeddings.T # B Bx2
        loss_rank_rich = InBatchNegativeCELoss(ranking_cosine_rich)
        loss_rank_low = InBatchNegativeCELoss(ranking_cosine_low)
        loss_rank_xfer = InBatchKLLoss(ranking_cosine_low, ranking_cosine_rich)

        loss_rank = loss_rank_rich + loss_rank_low 
        loss_xfer = loss_rel_xfer + loss_rank_xfer

        loss = loss_rank + loss_xfer
        return {'loss': loss, 'score': torch.diag(ranking_cosine_rich)}

