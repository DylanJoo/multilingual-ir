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
    istrain: Union[bool] = False
    language: str = "en"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        # text and id info 
        q_texts = [batch['query']} for batch in features]
        p_texts = [batch['passage']} for batch in features]
        ids = [(batch['qid'], batch['pid']) for batch in features]

        # input # if istrain, the 'label' should be zero or one.
        inputs = self.tokenizer(
                q_texts, p_texts,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors=self.return_tensors
        )

        return inputs, ids

@dataclass
class DataCollatorForDPR:
    """for biencoder models """
    tokenizer: Union[PreTrainedTokenizerBase] = None
    padding: Union[bool, str, PaddingStrategy] = True
    truncation: Union[bool, str] = True
    max_length: Optional[int] = 512
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    padding: Union[bool, str] = True
    # spec
    istrain: Union[bool] = False
    language: str = "en"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        # text and id info 
        q_texts = [batch['query']} for batch in features]
        p_texts = [batch['passage']} for batch in features]
        ids = [(batch['qid'], batch['pid']) for batch in features]

        # input # if istrain, the 'label' should be zero or one.
        q_inputs = self.tokenizer(
                q_texts, 
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors=self.return_tensors
        )
        p_inputs = self.tokenizer(
                p_texts,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors=self.return_tensors
        )

        return q_inputs, p_inputs, ids

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
    return_text: Union[bool] = False

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        qp_texts = [f"Query: {batch['query']} Document: {batch['passage']} Relevant:" \
                for batch in features]

        inputs = self.tokenizer(
                qp_texts,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors=self.return_tensors
        )

        # qid-pid pairs 
        ids = [(batch['qid'], batch['pid']) for batch in features]

        # labeling (if training)
        if self.istrain:
            targets = self.tokenizer(
                    [text['label'] for text in features],
                    truncation=True,
                    return_tensors=self.return_tensors
            ).input_ids
            inputs['labels'] = target

            return inputs, ids
