import torch
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple, Any
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase

@dataclass
class DataCollatorFormonoBERT:
    tokenizer: Union[PreTrainedTokenizerBase] = None
    padding: Union[bool, str, PaddingStrategy] = True
    truncation: Union[bool, str] = True
    max_length: Optional[int] = 512
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    padding: Union[bool, str] = True
    # spec
    # istrain: Union[bool] = False
    language: str = "en"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        # text and id info 
        texts_q = [batch['query'] for batch in features]
        texts_p = [batch['passage'] for batch in features]
        ids = [(batch['qid'], batch['did']) for batch in features]

        inputs = self.tokenizer(
                texts_q, texts_p,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors=self.return_tensors
        )

        return inputs, ids

@dataclass
class DataCollatorFormDPR:
    """for biencoder models """
    tokenizer: Union[PreTrainedTokenizerBase] = None
    padding: Union[bool, str, PaddingStrategy] = True
    truncation: Union[bool, str] = True
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    padding: Union[bool, str] = True
    # spec
    istrain: Union[bool] = False
    source_language: str = "en"
    target_language: str = "x"
    # max_length: Optional[int] = 512
    max_q_length: Optional[int] = None
    max_d_length: Optional[int] = None
    # language_relxfer: str = 'distill'
    use_only_positive: Union[bool, str] = True

    def relevance_transfer(self, features, only_positive=True):
        # rich lang and low lang query
        texts_q_rich = [batch['query'] for batch in features]
        texts_q_low = [batch['query_low'] for batch in features]

        # rich lang and low land positive passage
        texts_d_rich = [batch['positive'] for batch in features] 
        texts_d_low = [batch['positive_low'] for batch in features] 

        if only_positive is False:
            texts_d_rich += [batch['negative'] for batch in features] 
            texts_d_low += [batch['negative_low'] for batch in features] 

        return texts_q_rich, texts_d_rich, texts_q_low, texts_d_low

    def _tokenize(self, text, max_length):
        inputs = self.tokenizer(
                text,
                max_length=max_length,
                truncation='only_first',
                padding=True,
                add_special_tokens=True,
                return_token_type_ids=False,
                return_tensors=self.return_tensors
        )
        return inputs

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Returns: 
        q*_inputs: batch size of query (B, Lq)
        d*_inputs: batch size of doucment (english and low-resource lang) (Bx2, Ld)
        """
        # input # if istrain, the 'label' should be zero or one.

        texts_q_en, texts_d_en, texts_q_low, texts_d_low = self.relevance_transfer(features, self.use_only_positive)
        q_rich_inputs = self._tokenize(texts_q_en, self.max_q_length) 
        d_rich_inputs = self._tokenize(texts_d_en, self.max_d_length)
        q_low_inputs = self._tokenize(texts_q_low, self.max_q_length)
        d_low_inputs = self._tokenize(texts_d_low, self.max_d_length)
        return {'qr_inputs': q_rich_inputs, 'dr_inputs': d_rich_inputs, 
                'ql_inputs': q_low_inputs, 'dl_inputs': d_low_inputs}

@dataclass
class DataCollatorForDPR:
    """for biencoder models """
    tokenizer: Union[PreTrainedTokenizerBase] = None
    padding: Union[bool, str, PaddingStrategy] = True
    truncation: Union[bool, str] = True
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    padding: Union[bool, str] = True
    # spec
    istrain: Union[bool] = False
    # max_length: Optional[int] = 512
    max_q_length: Optional[int] = None
    max_p_length: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        # text and id info 
        texts_q = [batch['query'] for batch in features]
        texts_p_pos = [batch['positive'] for batch in features]
        # positive/negative
        texts_p_neg = [batch['negative'] for batch in features]

        # input # if istrain, the 'label' should be zero or one.
        q_inputs = self.tokenizer(
                texts_q, 
                max_length=self.max_q_length,
                truncation=True,
                padding=True,
                return_tensors=self.return_tensors
        )
        p_inputs = self.tokenizer(
                texts_p_pos + texts_p_neg,
                max_length=self.max_p_length,
                truncation=True,
                padding=True,
                return_tensors=self.return_tensors
        )

        if istrain:
            ids = [(batch['qid'], batch['did']) for batch in features]
            labels = torch.tensor([batch['label'] for batch in features])
            # do sth ...

        return (q_inputs, d_inputs)

@dataclass
class DataCollatorFormonoT5:
    tokenizer: Union[PreTrainedTokenizerBase] = None
    padding: Union[bool, str, PaddingStrategy] = True
    truncation: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    padding: Union[bool, str] = True
    # spec
    istrain: Union[bool] = False

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        texts_qp = [f"Query: {batch['query']} Document: {batch['passage']} Relevant:" \
                for batch in features]

        inputs = self.tokenizer(
                texts_qp,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors=self.return_tensors
        )

        # qid-did pairs 
        ids = [(batch['qid'], batch['did']) for batch in features]

        # labeling (if training)
        if self.istrain:
            targets = self.tokenizer(
                    [text['label'] for text in features],
                    truncation=True,
                    return_tensors=self.return_tensors
            ).input_ids
            inputs['labels'] = target

        return inputs, ids

@dataclass
class DataCollatorForSeq2Seq:
    tokenizer: Union[PreTrainedTokenizerBase] = None
    padding: Union[bool, str, PaddingStrategy] = True
    truncation: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    padding: Union[bool, str] = True
    # spec
    istrain: Union[bool] = False

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        text = [batch['text'] for batch in features]

        inputs = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors=self.return_tensors
        )

        return inputs
