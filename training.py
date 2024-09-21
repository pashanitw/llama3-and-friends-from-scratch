import torch
import contextlib
from load_llama_weights import convert_weights
from llama import llama3_8b, TransformerDecoder
import common as utils
from attention import CasualSelfAttention
import os
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)
from torch import nn
from typing import Dict, Generator
import bitsandbytes

logger = utils.get_logger("INFO")


@contextlib.contextmanager
def set_default_dtype(dtype: torch.dtype) -> Generator[None, None, None]:
    """
    Context manager to set torch's default dtype.

    Args:
        dtype (torch.dtype): The desired default dtype inside the context manager.

    Returns:
        ContextManager: context manager for setting default dtype.

    Example:
        >>> with set_default_dtype(torch.bfloat16):
        >>>     x = torch.tensor([1, 2, 3])
        >>>     x.dtype
        torch.bfloat16


    """
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


def load_checkpoint(checkpoint_path):
    # Proceed to load the file assuming it's correctly formatted
    state_dict = torch.load(
        checkpoint_path, map_location="cpu", mmap=True, weights_only=True
    )
    convert_model_state_dict = convert_weights(state_dict)
    model = llama3_8b()
    # model.load_state_dict(convert_model_state_dict)
    print("Loaded checkpoint '{}'".format(checkpoint_path))
    return model, convert_model_state_dict


def compile_model(
    model: TransformerDecoder,
    verbose: bool = False,
):
    backend = os.environ.get("TORCH_COMPILE_BACKEND", "inductor")

    if verbose:
        logger.info("Compiling model layers with torch.compile")

    counter = 0
    for m in reversed(list(model.modules())):
        if isinstance(m, CasualSelfAttention):
            counter += 1
            print(counter)
            m.compile(backend=backend)


def set_activation_checkpointing(
    model: nn.Module, auto_wrap_policy: any, **kwargs
) -> None:
    if isinstance(auto_wrap_policy, set):
        auto_wrap_policy = ModuleWrapPolicy(auto_wrap_policy)

    apply_activation_checkpointing(model, auto_wrap_policy)


def get_optimizer(optimizer):
    optimizer_dict = {
        "bitsandbytes.optim.PagedAdamW": bitsandbytes.optim.PagedAdamW,
        "AdamW": torch.optim.AdamW,
        "torch.optim.AdamW": torch.optim.AdamW,
    }

    if optimizer in optimizer_dict:
        return optimizer_dict[optimizer]
    else:
        raise RuntimeError(f"given optimizer '{optimizer}' is not supported")


def register_bwd_hook(
    model: TransformerDecoder,
    optim_dict: Dict[torch.nn.Parameter, torch.optim.Optimizer],
):

    def optim_step(param) -> None:
        optim_dict[param].step()
        optim_dict[param].zero_grad()

    for p in model.parameters():
        p.register_post_accumulate_grad_hook(optim_step)


# file_path = 'Meta-Llama-3-8B-Instruct/consolidated.00.pth'  # Update this path
# model, ckpt = load_checkpoint(file_path)
#
# compile_model(model, verbose=True)
