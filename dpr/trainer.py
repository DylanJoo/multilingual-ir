# inheritance required
import os
import torch
from transformers.utils import logging, WEIGHTS_NAME
from transformers.modeling_utils import unwrap_model, PreTrainedModel
logger = logging.get_logger(__name__)
from transformers import Trainer
from typing import Optional

TRAINING_ARGS_NAME = "training_args.bin"

class TrainerForBiEncoder(Trainer):
    """
    save_model: instead of saving the huggingface's `BertModel` model,
    I here save the query encoder since I didnt inherit the basic huggingface class.
    """
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`

        # [MODIFIED] replaced the model by model.query_encoder, since adopting bi-encoder training process
        # [MODIFIED] reload the state_dict for model.query_encoder
        if not isinstance(self.model.query_encoder, PreTrainedModel):
            if isinstance(unwrap_model(self.model.query_encoder), PreTrainedModel):
                # if state_dict is None:
                state_dict = self.model.query_encoder.state_dict()
                unwrap_model(self.model.query_encoder).save_pretrained(output_dir, state_dict=state_dict)
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                # if state_dict is None:
                state_dict = self.model.query_encoder.state_dict()
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.query_encoder.save_pretrained(output_dir, state_dict=state_dict)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
