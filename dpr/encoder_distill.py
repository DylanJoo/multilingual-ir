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

        ## OBJ1: in-batch negative training for ranking 
        ranking_cosine_rich = qr_embeddings @ dr_embeddings.T # B B
        ranking_cosine_low = ql_embeddings @ dl_embeddings.T # B B
        loss_rank_rich = InBatchNegativeCELoss(ranking_cosine_rich)
        loss_rank_low = InBatchNegativeCELoss(ranking_cosine_low)

        ## OBJ2: language relevance transfer (distilation)
        loss_rank_xfer = InBatchKLLoss(ranking_cosine_low, ranking_cosine_rich)
        loss = loss_rank_xfer + loss_rank_low + loss_rank_rich 
        #[TODO] control these w/ hyperparms 
        # Personally, loss_rank_rich should be the smallest;
        # and the other two should be scheduled

        # verbose
        # print('loss:', loss.detach().cpu().numpy(), 
        #       'loss-xfer:', loss_rank_xfer.detach().cpu().numpy(), 
        #       'loss-rank (low) ', loss_rank_low.detach().cpu().numpy(),
        #       'loss-rank (rich) ', loss_rank_rich.detach().cpu().numpy())

        return {'loss': loss, 'score': torch.diag(ranking_cosine_low)}


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

