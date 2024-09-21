from functools import partial
from warnings import warn

from lance.dependencies import torch
from pydantic_cli import run_and_exit
from typer.cli import utils_app

import common as utils

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

        pass

# Usage example

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
