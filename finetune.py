from functools import partial
from warnings import warn
from pydantic_cli import run_and_exit

from common import get_logger, BaseFineTuning, TrainingArgs

logger = get_logger("DEBUG")


class FullFineTuning(BaseFineTuning):
    def __init__(self, config):
        pass

# Usage example

def main(args: TrainingArgs):
    print(f"Training with:")
    print(f"  Model: {args.model}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Device: {args.device}")


if __name__ == "__main__":
    args = TrainingArgs.from_args()
    main(args)
