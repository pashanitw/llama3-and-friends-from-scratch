from functools import partial
from warnings import warn

from lance.dependencies import torch
from pyannote.audio.models.embedding.wespeaker.convert import state_dict
from pydantic_cli import run_and_exit
from typer.cli import utils_app

import common as utils
import training as tr
from attention import CasualSelfAttention
from llama3_tokenizer import Tokenizer
from training import get_optimizer

logger = utils_app.get_logger("DEBUG")


class FullFineTuning(utils.BaseFineTuning):
    def __init__(self, training_args: utils.TrainingArgs):
        self.device = utils.get_device()
        self.dtype = utils.get_dtype()

        if self.dtype == torch.float16:
            raise RuntimeError("only supporting fp 32 and bf16")

        self.training_args = training_args

        self.current_epoch = 0
        self.global_step = 0

        torch.manual_seed(args.seed)

    def setup(self) -> None:
        training_args = self.training_args
        # step 1: Load checkpoint
        self.model = self._setup_model()

        # step 2: load tokenizer
        # Todo: add configuration for this later
        tokenizer = Tokenizer(model_path="./Meta-Llama-3-8B-Instruct/tokenizer.model")

        pass

    def _setup_model(self):
        training_args = self.training_args
        with tr.set_default_dtype(self.dtype), self.device:
            model, state_dict = tr.load_checkpoint(training_args.checkpoint_dir)

        if training_args.compile:
            tr.compile_model(model)
        if training_args.enable_activation_checkpointing:
            tr.set_activation_checkpointing(model, CasualSelfAttention)

        model.load_state_dict(state_dict)

        return model

    def _setup_optimizer(self):
        training_args = self.training_args
        if training_args.optimizer_in_bwd:
            optim_dict = {p: tr.get_optimizer([p]) for p in self.model.parameters()}
            tr.register_bwd_hook(self.model, optim_dict)

        else:
            optimizer = get_optimizer(training_args.optimizer)
            if optimizer:
                optimizer = optimizer(self.model.parameters())

            return optimizer


def main(args: utils.TrainingArgs):
    print(f"Training with:")
    print(f"  Model: {args.model}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Device: {args.device}")


if __name__ == "__main__":
    args = utils.TrainingArgs.from_args()
    main(args)
