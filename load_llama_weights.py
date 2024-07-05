import torch
from typing import Dict
import re
# state dict key mappings from Meta's format to Our implementation
_FROM_META = {
    "tok_embeddings.weight": "tok_embeddings.weight",
    "norm.weight": "norm.scale",
    "output.weight": "output.weight",
    "layers.{}.attention.wk.weight": "layers.{}.attn.k_proj.weight",
    "layers.{}.attention.wq.weight": "layers.{}.attn.q_proj.weight",
    "layers.{}.attention.wv.weight": "layers.{}.attn.v_proj.weight",
    "layers.{}.attention.wo.weight": "layers.{}.attn.o_proj.weight",
    "layers.{}.attention_norm.weight": "layers.{}.attn_norm.scale",
    "layers.{}.ffn_norm.weight": "layers.{}.mlp_norm.scale",
    "layers.{}.feed_forward.w1.weight": "layers.{}.mlp.gate_proj.weight",
    "layers.{}.feed_forward.w2.weight": "layers.{}.mlp.down_proj.weight",
    "layers.{}.feed_forward.w3.weight": "layers.{}.mlp.up_proj.weight",
}

def _get_mapped_key(key: str, mapping_dict: Dict[str, str]) -> str:
    try:
        if "layers" in key:
            # Replace layer number with "{}" to create key for lookup
            abstract_key = re.sub(r"(\.\d+)", ".{}", key)
            layer_num = re.search(r"\d+", key).group(0)
            new_key = mapping_dict[abstract_key]
            new_key = new_key.format(layer_num)
        else:
            new_key = mapping_dict[key]
    except KeyError as e:
        raise Exception(
            f'Error converting the state dict. Found unexpected key: "{key}". '
            "Please make sure you're loading a checkpoint with the right format. "
        ) from e

    return new_key


def convert_weights(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Args:
        state_dict (Dict[str, torch.Tensor]): State dict in Meta's format.

    Returns:
        Dict[str, torch.Tensor]: State dict in TorchTune's format.
    """
    converted_state_dict = {}
    for key, value in state_dict.items():
        if key not in ["rope.freqs"]:  # Skip loading the position embeddings
            new_key = _get_mapped_key(key, _FROM_META)
            converted_state_dict[new_key] = value

    return converted_state_dict

def load_and_print_keys(file_path):
    """
    Load a PyTorch model's state dictionary from the given file path and print its keys.

    Args:
    file_path (str): The path to the PyTorch model file.
    """
    try:
        # Open the file in binary mode to check the first few bytes
        with open(file_path, 'rb') as file:
            magic = file.read(2)
            if not (magic == b'PK' or magic.startswith(b'\x80\x02')):  # Check for ZIP or pickle headers
                print("File does not start with expected magic numbers for a PyTorch model.")
                return None

        # Proceed to load the file assuming it's correctly formatted
        state_dict = torch.load(file_path, map_location=torch.device('cpu'))
        print("Keys in the state dictionary:")
        for key in state_dict.keys():
            print(key)
        return state_dict
    except Exception as e:
        print(f"Failed to load the model from {file_path}. Error: {str(e)}")
        return None

