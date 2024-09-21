import torch
from load_llama_weights import convert_weights
from llama import llama3_8b, TransformerDecoder
import common as utils
from attention import CasualSelfAttention
import os

logger = utils.get_logger("INFO")


def load_checkpoint(checkpoint_path):
    # Proceed to load the file assuming it's correctly formatted
    state_dict = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=True)
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


file_path = 'Meta-Llama-3-8B-Instruct/consolidated.00.pth'  # Update this path
model, ckpt  = load_checkpoint(file_path)

compile_model(model, verbose=True)