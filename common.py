from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import torch
from typing import Optional

@dataclass
class BaseFineTuning:
    """
    This dataclass provides a structure for LLM fine-tuning recipes.
    It includes fields for storing necessary components and methods for key operations.

    Attributes:
        model: The model to be fine-tuned
        optimizer: The optimizer for training
        loss_fn: The loss function
        dataloader: The data loader for training data
        training_params: A dictionary to store training parameters
        checkpoint: Optional field to store checkpoint data

    Methods are defined for loading checkpoints, setup, training, saving checkpoints, and cleanup.
    These methods should be implemented by updating the dataclass or subclassing it.
    """

    model: Any = field(default=None)
    optimizer: Any = field(default=None)
    loss_fn: Any = field(default=None)
    dataloader: Any = field(default=None)
    training_params: Dict[str, Any] = field(default_factory=dict)
    checkpoint: Optional[Any] = field(default=None)

    def load_checkpoint(self, **kwargs) -> None:
        """
        Load all state for the recipe from the checkpoint file.
        This includes state for the model, optimizer, dataloader, and training parameters.
        """
        raise NotImplementedError("Subclass must implement abstract method")

    def setup(self, **kwargs) -> None:
        """
        Set up all components necessary for training.
        This includes model, optimizer, loss function, and dataloader.
        """
        raise NotImplementedError("Subclass must implement abstract method")

    def train(self, **kwargs) -> None:
        """
        Implement the core training logic.
        This includes the training loop, loss computation, gradient accumulation, and backward pass.
        """
        raise NotImplementedError("Subclass must implement abstract method")

    def save_checkpoint(self, **kwargs) -> None:
        """
        Save all state for the recipe.
        This includes state for the model, optimizer, dataloader, and training parameters.
        """
        raise NotImplementedError("Subclass must implement abstract method")

    def cleanup(self, **kwargs) -> None:
        """
        Perform any necessary cleanup for the recipe.
        """
        raise NotImplementedError("Subclass must implement abstract method")


def get_device() -> torch.device:

    device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)

    if device.type == "cuda":
        local_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        if device.index is None:
            device = torch.device("cuda", index=local_rank)

        if device.index >= torch.cuda.device_count():
            raise RuntimeError(f"The local rank {local_rank} is larger than the available devices {torch.cuda.device_count()}.")

        torch.cuda.set_device(device)

        if torch.distributed.is_initialized() and device.index != local_rank:
            raise RuntimeError(f"The device rank {device.index} is does not match the local rank {local_rank}.")


    # validate the device availability
    try:
        torch.empty(0, device=device)
    except RuntimeError as e:
        raise RuntimeError(f"The device {device} is not a supported device.")

    return device


import logging
from typing import Optional


def get_logger(level: Optional[str] = "INFO") -> logging.Logger:

    logger = logging.getLogger()
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(level.upper())
    return logger


import argparse
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

import argparse
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class TrainingArgs:
    # Dataset
    max_seq_len: Optional[int] = None
    dataset: str = "alpaca_dataset"
    seed: Optional[int] = None
    shuffle: bool = True

    # Model
    model: str = "llama_3"
    checkpoint_dir: Path = Path("/tmp/Llama-2-7b-hf")
    output_dir: Path = Path("/tmp/alpaca-llama2-finetune")
    resume_from_checkpoint: bool = False

    # Training
    batch_size: int = 2
    epochs: int = 3
    optimizer: str = "bitsandbytes.optim.PagedAdamW"
    learning_rate: float = 2e-5
    optimizer_in_bwd: bool = True
    max_steps_per_epoch: Optional[int] = None
    gradient_accumulation_steps: int = 1
    clip_grad_norm: Optional[float] = None
    compile: bool = False

    # Hardware
    device: str = "cuda"
    enable_activation_checkpointing: bool = True
    dtype: str = "bf16"

    # Logging
    metric_logger_component: str = "torchtune.training.metric_logging.DiskLogger"
    log_every_n_steps: int = 1
    log_peak_memory_stats: bool = False

    @classmethod
    def from_args(cls):
        parser = argparse.ArgumentParser(description="Training arguments")
        for field_name, field_def in cls.__dataclass_fields__.items():
            field_type = field_def.type
            default = field_def.default

            if field_type == Optional[int]:
                parser.add_argument(f"--{field_name}", type=int, default=default)
            elif field_type == bool:
                parser.add_argument(f"--{field_name}", type=lambda x: x.lower() == 'true', default=default)
            elif field_type == Path:
                parser.add_argument(f"--{field_name}", type=Path, default=default)
            elif field_type == float:
                parser.add_argument(f"--{field_name}", type=float, default=default)
            else:
                parser.add_argument(f"--{field_name}", type=field_type, default=default)

        args = parser.parse_args()
        return cls(**vars(args))


def verify_bf16_support() -> bool:
    """
    Check that bf16 is available on this hardware. Requirements:
        - CUDA is available and supports bf16
            - CUDA version >= 11
            - CUDA compute capability >= 8
        - NCCL is available and version >= 2.10

    Returns:
        bool: True if bf16 is available, False otherwise.

    """
    return (
        torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
        and torch.distributed.is_nccl_available()
        and torch.cuda.nccl.version() >= (2, 10)
    )



import torch
from typing import Optional

VALID_DTYPES = {'fp16', 'bf16', 'fp32', 'fp64'}



def get_dtype(dtype: Optional[str] = None, device: Optional[torch.device] = None) -> torch.dtype:
    if dtype is None:
        return torch.float32

    dtype = dtype.lower()
    if dtype not in VALID_DTYPES:
        raise ValueError(f"Unsupported dtype: {dtype}. Must be one of {', '.join(VALID_DTYPES)}.")

    # Map abbreviated format to full name
    dtype_map = {'fp16': 'float16', 'bf16': 'bfloat16', 'fp32': 'float32', 'fp64': 'float64'}
    full_dtype_name = dtype_map[dtype]

    torch_dtype = getattr(torch, full_dtype_name)

    if torch_dtype == torch.bfloat16 and device != torch.device("cpu"):
        if not verify_bf16_support():
            raise RuntimeError("bf16 precision not supported on this hardware")

    return torch_dtype