import multiprocessing
from dataclasses import dataclass, field
from typing import Optional
from transformers import (
    AutoConfig,
    AutoTokenizer,
    TrainingArguments,
    HfArgumentParser,
    DefaultDataCollator,
)

from datasets import load_dataset, DatasetDict

# hacky import
import sys
sys.path.append('/tmp2/jhju/multilingual-ir/')
from encoder import BiEncoderForRelevanceTransfer
from dataset.mmarco import join_dataset
from datacollator import DataCollatorFormDPR
from trainer import TrainerForBiEncoder

import os
os.environ["WANDB_DISABLED"] = "false"

# Arguments: (1) Model arguments (2) DataTraining arguments (3)
@dataclass
class OurModelArguments:
    # Huggingface's original arguments
    model_name_or_path: Optional[str] = field(default='bert-base-uncased')
    config_name: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=True)
    model_revision: str = field(default="main")
    use_auth_token: bool = field(default=False)
    # Cutomized arguments
    freeze_document_encoder: Optional[bool] = field(default=True)
    pooler: str = field(default='cls')

@dataclass
class OurDataArguments:
    # Huggingface's original arguments.
    dataset_name: Optional[str] = field(default=None)
    dataset_config_name: Optional[str] = field(default=None)
    overwrite_cache: bool = field(default=False)
    validation_split_percentage: Optional[int] = field(default=5)
    preprocessing_num_workers: Optional[int] = field(default=None)
    train_file: Optional[str] = field(default=None)
    # eval_file: Optional[str] = field(default=None)
    # test_file: Optional[str] = field(default=None)
    # Customized arguments
    max_q_length: Optional[int] = field(default=None)
    max_d_length: Optional[int] = field(default=256)

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
    learning_rate: float = field(default=1e-5)
    # Customized arguments
    place_model_on_device: bool = field(default=False)
    remove_unused_columns: bool = field(default=False)

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

    # model
    model = BiEncoderForRelevanceTransfer(
            model_name=model_args.model_name_or_path,
            tokenizer_name=model_args.tokenizer_name,
            freeze_document_encoder=model_args.freeze_document_encoder,
            pooling=model_args.pooler
    )

    # Dataset
    ## adjusted to multilingual (for relevance transfer)
    dataset = join_dataset(data_args.train_file)

    ## Preprocessing
    datacollator = DataCollatorFormDPR(
            tokenizer=model.tokenizer,
            padding=True,
            truncation=True,
            max_q_length=data_args.max_q_length,
            max_d_length=data_args.max_d_length,
    )

    # Trainer
    trainer = TrainerForBiEncoder(
            model=model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'],
            data_collator=datacollator
    )
            # eval_dataset=dataset['test'],

    # ***** strat training *****
    results = trainer.train(
            resume_from_checkpoint=training_args.resume_from_checkpoint
    )

    # [check]
    state_dict = model.document_encoder.state_dict()
    model.document_encoder.save_pretrained('checkpoint/document_encoder/', state_dict=state_dict)

    return results

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ == '__main__':
    main()

