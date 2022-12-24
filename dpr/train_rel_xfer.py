# hacky import
import sys
sys.path.append('../')
import sys
import multiprocessing
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    AutoConfig,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    HfArgumentParser,
    DefaultDataCollator,
)

from datasets import load_dataset, DatasetDict
from models import TctColBertForIR
from datacollator import IRTripletCollator

import os
os.environ["WANDB_DISABLED"] = "true"

# Arguments: (1) Model arguments (2) DataTraining arguments (3)
@dataclass
class OurModelArguments:
    # Huggingface's original arguments
    # model_name_or_path: Optional[str] = field(default='bert-base-uncased')
    config_name: Optional[str] = field(default='bert-base-uncased')
    tokenizer_name: Optional[str] = field(default='bert-base-uncased')
    cache_dir: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=True)
    model_revision: str = field(default="main")
    use_auth_token: bool = field(default=False)
    # Cutomized arguments
    query_encoder_model_name_or_path: Optional[str] = field(default='bert-base-uncased')
    document_encoder_model_name_or_path: Optional[str] = field(default='bert-base-uncased')
    # colbert_type: Optional[str] = field(default="colbert")
    # dim: Optional[int] = field(default=128)

@dataclass
class OurDataArguments:
    # Huggingface's original arguments.
    dataset_name: Optional[str] = field(default=None)
    dataset_config_name: Optional[str] = field(default=None)
    overwrite_cache: bool = field(default=False)
    validation_split_percentage: Optional[int] = field(default=5)
    preprocessing_num_workers: Optional[int] = field(default=None)
    # train_file: Optional[str] = field(default=None)
    # eval_file: Optional[str] = field(default=None)
    # test_file: Optional[str] = field(default=None)
    # Customized arguments
    # max_q_length: Optional[int] = field(default=32)
    # max_p_length: Optional[int] = field(default=128)

@dataclass
class OurTrainingArguments(TrainingArguments):
    # Huggingface's original arguments.
    output_dir: str = field(default='./temp')
    seed: int = field(default=42)
    data_seed: int = field(default=None)
    do_train: bool = field(default=False)
    do_eval: bool = field(default=False)
    max_steps: int = field(default=100)
    save_steps: int = field(default=100)
    eval_steps: int = field(default=10)
    evaluation_strategy: Optional[str] = field(default='steps')
    per_device_train_batch_size: int = field(default=8)
    per_device_eval_batch_size: int = field(default=8)
    weight_decay: float = field(default=0.0)
    logging_dir: Optional[str] = field(default='./logs')
    warmup_ratio: float = field(default=0.1)
    warmup_steps: int = field(default=0)
    resume_from_checkpoint: Optional[str] = field(default=None)
    learning_rate: float = field(default=5e-5)

def main():

    # Parseing argument for huggingface packages
    parser = HfArgumentParser((OurModelArguments, OurDataArguments, OurTrainingArguments))
    # model_args, data_args, training_args = parser.parse_args_into_datalcasses()
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = \
                parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        # [CONCERN] Deprecated? or any parser issue.
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    config = AutoConfig.from_pretrained(
            model_args.config_name, 
            output_hidden_states=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, 
            cache_dir=model_args.cache_dir,
            user_fast=model_args.use_fast_tokenizer
    )

    # model
    model_teacher = TctColBertForIR.from_pretrained(
            pretrained_model_name_or_path=model_args.kd_teacher_model_name_or_path,
            config=config,
            colbert_type='colbert-inbatch',
    ) if model_args.colbert_type == 'tctcolbert' else None

    model_kwargs = {
            'dim': model_args.dim,
            'similarity_metric': 'cosine',
            'mask_punctuation': True,
            'kd_teacher': model_teacher,
            'colbert_type': model_args.colbert_type
    }
    model = TctColBertForIR.from_pretrained(
            pretrained_model_name_or_path=model_args.model_name_or_path,
            config=config,
            **model_kwargs
    )

    # Dataset
    ## Loading form json
    if training_args.do_eval:
        dataset = DatasetDict.from_json({
            "train": data_args.train_file,
            "eval": data_args.eval_file
        })
    else:
        dataset = DatasetDict.from_json({"train": data_args.train_file,})
        data['eval'] = None

    # data collator (transform the datset into the training mini-batch)
    ## Preprocessing
    triplet_collator = IRTripletCollator(
            tokenizer=tokenizer,
            query_maxlen=data_args.max_q_seq_length,
            doc_maxlen=data_args.max_p_seq_length,
            in_batch_negative=(model_args.colbert_type != 'colbert')
    )

    # Trainer
    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['eval'],
            data_collator=triplet_collator
    )

    # ***** strat training *****
    results = trainer.train(
            resume_from_checkpoint=training_args.resume_from_checkpoint
    )

    return results

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ == '__main__':
    main()

