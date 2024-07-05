import torch
from torch import nn, Tensor
from dataclasses import dataclass
from typing import Optional
import torch.nn.functional as F
from rope import RotaryPositionalEmbeddings

# Define the necessary arguments for developing LLaMA-3 from scratch


@dataclass
class LLM_Args:
    num_layers: int  # Number of transformer layers in the model
    num_heads: int  # Number of attention heads in each transformer layer
    embed_dim: int  # Dimensionality of the embeddings and transformer hidden states
    intermediate_dim: (
        int  # Dimensionality of the intermediate layer in the feed-forward network
    )
    vocab_size: int  # Size of the vocabulary (number of unique tokens)
    max_seq_len: int  # Maximum sequence length that the model can handle
    attn_dropout: float  # Dropout rate for attention layers
    norm_eps: float  # Epsilon value for layer normalization to avoid division by zero
    rope_base: int  # Base value for Rotary Position Embedding (RoPE)
    num_kv_heads: Optional[int] = None  # Number of key-value attention heads (optional)
    attention_bias = False  # Boolean flag to indicate whether to use attention bias


# kv cache is useful for lowering the memory requirement while doing the inference
class KVCache(nn.Module):
    def __init__(self, max_batch_size: int, dtype, args: LLM_Args):
        super().__init__()
        num_heads = (
            args.num_heads
        )  # we are using num heads because we are saving expanded version of k, v
        head_dim = args.embed_dim // args.num_heads
        # Shape of the cache: (batch_size, num_heads, max_seq_len, head_dim)
        cache_shape = (max_batch_size, num_heads, args.max_seq_len, head_dim)

        # Initialize key and value caches with zeros
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos: Tensor, k_val: Tensor, v_val: Tensor):
        num_positions = input_pos.shape[0]
        bsz, _, seq_len, _ = (
            k_val.shape
        )  # k_val shape: (batch_size, num_heads, seq_len, head_dim)

        assert (
            num_positions == seq_len
        )  # Ensure input positions match the sequence length

        k_out = self.k_cache  # Retrieve the key cache
        v_out = self.v_cache  # Retrieve the value cache

        # Update the caches with the new key and value tensors at the specified positions
        k_out[:, :, input_pos, :] = k_val
        v_out[:, :, input_pos, :] = v_val

        return k_out, v_out  # Return the updated caches


# Next Define the Attention Block


class CasualSelfAttention(nn.Module):
    def __init__(
        self,
        args: LLM_Args,
        pos_embeddings: nn.Module,
        kv_cache: Optional[KVCache] = None,
    ):
        super().__init__()

        # Validate that the number of attention heads is divisible by the number of key-value heads
        if args.num_heads % args.num_kv_heads != 0:
            raise ValueError(
                f"Number of attention heads ({args.num_heads}) must be divisible by the number of key-value attention heads ({args.num_kv_heads})"
            )

        # Validate that the embedding dimension is divisible by the number of attention heads
        if args.embed_dim % args.num_heads != 0:
            raise ValueError(
                f"Embedding dimension ({args.embed_dim}) must be divisible by the number of attention heads ({args.num_heads})"
            )

        # Validate that the attention dropout value is within the valid range
        if args.attn_dropout < 0.0 or args.attn_dropout > 1.0:
            raise ValueError(
                f"Attention dropout value must be between 0.0 and 1.0 (inclusive), but got {args.attn_dropout}"
            )

        self.head_dim = args.embed_dim // args.num_heads
        self.num_heads = args.num_heads
        self.num_kv_heads = args.num_kv_heads if args.num_kv_heads else args.num_heads
        self.embed_dim = args.embed_dim
        self.attn_dropout = args.attn_dropout
        self.max_seq_len = args.max_seq_len

        # Linear layers for projecting inputs to query, key, and value vectors
        self.q_proj = nn.Linear(
            args.embed_dim, self.head_dim * args.num_heads, bias=args.attention_bias
        )
        self.k_proj = nn.Linear(
            args.embed_dim, self.head_dim * self.num_kv_heads, bias=args.attention_bias
        )
        self.v_proj = nn.Linear(
            args.embed_dim, self.head_dim * self.num_kv_heads, bias=args.attention_bias
        )
        self.o_proj = nn.Linear(
            args.embed_dim, self.embed_dim, bias=args.attention_bias
        )

        # Positional embeddings module
        self.pos_embeddings = pos_embeddings
        # Key-value cache
        self.kv_cache = kv_cache

    def forward(self, x: Tensor, mask: Optional[Tensor], input_pos: Optional[Tensor]):
        bsz, seq_len, _ = x.shape  # x shape: (batch_size, seq_len, embed_dim)

        # Validate that the sequence length does not exceed the maximum allowed length
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length ({seq_len}) exceeds the maximum allowed length ({self.max_seq_len})"
            )

        q = self.q_proj(x)  # q shape: (batch_size, seq_len, head_dim * num_heads)
        k = self.k_proj(x)  # k shape: (batch_size, seq_len, head_dim * num_kv_heads)
        v = self.v_proj(x)  # v shape: (batch_size, seq_len, head_dim * num_kv_heads)

        q_per_kv = (
            self.num_heads // self.num_kv_heads
        )  # Number of queries per value head

        # Reshape queries, keys, and values for multi-head attention
        q = q.view(
            bsz, seq_len, self.num_kv_heads, q_per_kv, self.head_dim
        )  # q shape: (batch_size, seq_len, num_kv_heads, q_per_kv, head_dim)
        k = k.view(
            bsz, seq_len, self.num_kv_heads, 1, self.head_dim
        )  # k shape: (batch_size, seq_len, num_kv_heads, 1, head_dim)
        v = v.view(
            bsz, seq_len, self.num_kv_heads, 1, self.head_dim
        )  # v shape: (batch_size, seq_len, num_kv_heads, 1, head_dim)

        if self.num_kv_heads < self.num_heads:
            # Expand keys and values if necessary
            k = k.expand(
                bsz, seq_len, self.num_kv_heads, q_per_kv, self.head_dim
            )  # Expand k shape: (batch_size, seq_len, num_kv_heads, q_per_kv, head_dim)
            v = v.expand(
                bsz, seq_len, self.num_kv_heads, q_per_kv, self.head_dim
            )  # Expand v shape: (batch_size, seq_len, num_kv_heads, q_per_kv, head_dim)

        # Reshape for multi-head attention computation
        q = q.reshape(
            bsz, seq_len, -1, self.head_dim
        )  # q shape: (batch_size, seq_len, num_heads, head_dim)
        k = k.reshape(
            bsz, seq_len, -1, self.head_dim
        )  # k shape: (batch_size, seq_len, num_heads, head_dim)
        v = v.reshape(
            bsz, seq_len, -1, self.head_dim
        )  # v shape: (batch_size, seq_len, num_heads, head_dim)

        # Apply positional embeddings
        q = self.pos_embeddings(
            q, input_pos
        )  # q shape after pos embeddings: (batch_size, seq_len, num_heads, head_dim)
        k = self.pos_embeddings(
            k, input_pos
        )  # k shape after pos embeddings: (batch_size, seq_len, num_heads, head_dim)

        # Transpose for attention computation
        q = q.transpose(1, 2)  # q shape: (batch_size, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2)  # k shape: (batch_size, num_heads, seq_len, head_dim)
        v = v.transpose(1, 2)  # v shape: (batch_size, num_heads, seq_len, head_dim)

        if self.kv_cache:
            # Update the KV cache
            k, v = self.kv_cache.update(input_pos, k, v)

        # Compute attention scores
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.attn_dropout,
            is_causal=self.kv_cache is None,
        )  # output shape: (batch_size, num_heads, seq_len, head_dim)

        # Reshape the output back to the original shape
        output = (
            output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        )  # output shape: (batch_size, seq_len, embed_dim)
        return self.o_proj(
            output
        )  # Final output shape: (batch_size, seq_len, embed_dim)


# Transformer Decoder Layer we are gonna repeat the many times this is the major block of the transformer:
# ┌──────────────────────────────┐
# │                              │
# │  ┌──────────────────────────┐│
# │  │                          ││
# │  │        +-------------+   ││
# │  │        │             │   ││
# │  │  +-----▼-----+       │   ││
# │  │  │ MLP block │       │   ││
# │  │  +-----------+       │   ││
# │  │        │             │   ││
# │  │  +-----▼-----+       │   ││
# │  │  │ RMSNorm   │       │   ││
# │  │  +-----------+       │   ││
# │  │        │             │   ││
# │  │        +-------------+   ││
# │  │        │             │   ││
# │  │  +-----▼-----+       │   ││
# │  │  │ Casual    │       │   ││
# │  │  │ self-attn │       │   ││
# │  │  │ block     │       │   ││
# │  │  +-----------+       │   ││
# │  │        │             │   ││
# │  │  +-----▼-----+       │   ││
# │  │  │ RMSNorm   │       │   ││
# │  │  +-----------+       │   ││
# │  │        │             │   ││
# │  │        +-------------+   ││
# │  │                          ││
# │  └──────────────────────────┘│
# │                              │
# └──────────────────────────────┘


class TransformerDecoderLayer(nn.Module):
    def __init__(self, attn, mlp, attn_norm, mlp_norm):
        super().__init__()

        self.attn = attn  # Casual self-attention block
        self.mlp = mlp  # MLP block
        self.attn_norm = attn_norm  # RMSNorm before attention
        self.mlp_norm = mlp_norm  # RMSNorm before MLP

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        input_pos: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass for the Transformer decoder layer.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, embed_dim)
            mask (Optional[Tensor]): Attention mask of shape (batch_size, seq_len)
            input_pos (Optional[Tensor]): Input positions for positional encoding

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        # Pre-normalization before attention block
        pre_norm_attn = self.attn_norm(x)  # Shape: (batch_size, seq_len, embed_dim)

        # Attention block
        attn_out = self.attn(
            pre_norm_attn, mask, input_pos
        )  # Shape: (batch_size, seq_len, embed_dim)

        # Residual connection after attention block
        x = attn_out + x  # Shape: (batch_size, seq_len, embed_dim)

        # Pre-normalization before MLP block
        pre_norm_mlp = self.mlp_norm(x)  # Shape: (batch_size, seq_len, embed_dim)

        # MLP block
        mlp_out = self.mlp(pre_norm_mlp)  # Shape: (batch_size, seq_len, embed_dim)

        # Residual connection after MLP block
        x = mlp_out + x  # Shape: (batch_size, seq_len, embed_dim)

        return x


# let's implement the mlp block that we are passing to the above transformer decoder layer block
"""
'''
        ┌─────────────────────────────────────┐
        │                 Linear              │
        │                   ▲                 │
        │                   │                 │
  ──────┼───►─────────────► + ─────────────── ┼──────►
        │     ▲             ▲                 │
        │     │             │                 │
        │     |             │                 │
        │    SiLU          Linear             │
        │     ▲             ▲                 │
        │     |             │                 │
        │   Linear ─────────┘                 │
        │              ▲                      |
        |              │                      |  
        └─────────────────────────────────────┘
                Gated MLP block
'''
"""


class FeedForward(nn.Module):
    """
    Gated MLP block for feedforward operation

    Parameters:
        gate_proj (nn.Module): Linear projection for the gating mechanism.
        up_proj (nn.Module): Linear projection for the upper branch.
        down_proj (nn.Module): Linear projection for combining the gated results.
        activation (nn.Module): Activation function, default is SiLU.
    """

    def __init__(
        self,
        gate_proj: nn.Module,
        up_proj: nn.Module,
        down_proj: nn.Module,
        activation: nn.Module = nn.SiLU(),
    ) -> None:
        super().__init__()
        self.gate_proj = gate_proj  # Gating projection
        self.up_proj = up_proj  # Upper branch projection
        self.down_proj = down_proj  # Downstream projection to combine results
        self.activation = activation  # Activation function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the gated MLP.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after gated MLP operations.
        """
        gated_output = self.gate_proj(x)
        activated_output = self.activation(
            gated_output
        )  # Apply activation to gated output
        up_proj_output = self.up_proj(x)  # Compute upper branch
        return self.down_proj(
            activated_output * up_proj_output
        )  # Combine and project outputs


def llama_mlp(dim: int, hidden_dim: int) -> FeedForward:
    """
    Constructs a Gated MLP block with specific dimensions.

    Args:
        dim (int): Dimensionality of the input and output.
        hidden_dim (int): Dimensionality of the hidden layer.

    Returns:
        FeedForward: The constructed Gated MLP block.
    """
    gate_proj = nn.Linear(dim, hidden_dim, bias=False)  # Gating projection without bias
    up_proj = nn.Linear(dim, hidden_dim, bias=False)  # Upper projection without bias
    down_proj = nn.Linear(
        hidden_dim, dim, bias=False
    )  # Downstream projection without bias
    return FeedForward(
        gate_proj, up_proj, down_proj, nn.SiLU()
    )  # Create the FeedForward block with SiLU activation


def scale_mlp_hidden_dim(dim: int, multiple_of: int = 256) -> int:
    """
    Adjust the hidden dimension for an MLP layer to be a multiple of a specified number.
    This ensures the dimension aligns well with hardware optimizations like those used in GPUs.

    Args:
        dim (int): The original dimension of the MLP layer.
        multiple_of (int): The number that the hidden dimension should be a multiple of (default is 256).

    Returns:
        int: The adjusted hidden dimension that is a multiple of `multiple_of`.
    """
    # Calculate an initial hidden dimension as about 1.33 times the input dimension
    initial_estimate = 4 * int(2 * dim / 3)

    # Adjust the dimension to the nearest multiple of `multiple_of`
    adjusted_dimension = multiple_of * (
        (initial_estimate + multiple_of - 1) // multiple_of
    )

    return adjusted_dimension


# let's implement the final transformer decoder block which basically nothing creating many layers of above transformer decoder block

import torch
from torch import nn, Tensor
import copy


def _get_clones(module: nn.Module, N: int) -> nn.ModuleList:
    """Creates N deep copies of a given module."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        tok_embeddings: nn.Embedding,
        layer: nn.Module,
        norm: nn.Module,
        output: nn.Linear,
        args: LLM_Args,
    ):
        super().__init__()
        self.tok_embeddings = tok_embeddings  # Embedding layer for token embeddings
        self.norm = norm  # Normalization layer
        self.output = output  # Output linear layer
        self.layers = _get_clones(
            layer, args.num_layers
        )  # Clone the transformer layer multiple times
        self.max_seq_len = args.max_seq_len
        # Pre-compute a causal mask to avoid future computation in each forward pass
        self.casual_mask = torch.tril(
            torch.ones((args.max_seq_len, args.max_seq_len), dtype=torch.bool)
        )

        self.args = args

    def setup_caches(self, max_batch_size: int, dtype: torch.dtype):
        """
        Initialize the cache for key-value pairs in the attention mechanism and set up the causal mask.

        Args:
            max_batch_size (int): Maximum batch size expected, which determines the cache size.
            dtype (torch.dtype): Data type of the tensors to be used for the cache.
        """
        # Set up caches for all layers in the transformer. This cache is used to store key and value
        # pairs across multiple forward passes, which is particularly useful in incremental decoding.
        for layer in self.layers:
            layer.attn.kv_cache = KVCache(max_batch_size, dtype, self.args)

        # Initialize the causal mask that is used to prevent attention to future tokens.
        # This mask is lower triangular, indicating that a position can only attend to previous positions.
        self.casual_mask = torch.tril(
            torch.ones(self.args.max_seq_len, self.args.max_seq_len, dtype=torch.bool)
        )

    def forward(self, tokens: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        # Get batch size and sequence length from tokens tensor
        bsz, seq_len = tokens.shape

        h = self.tok_embeddings(tokens)  #  [batch_size, seq_length, embedding_dim])

        if input_pos is not None:
            # Create a mask for causal attention based on input positions (Shape: [1, 1, seq_length, seq_length])
            mask = self.casual_mask[None, None, input_pos]
        else:
            mask = None

        # Apply transformer layers with attention masking
        for layer in self.layers:
            h = layer(h, mask, input_pos)  #  [batch_size, seq_length, embedding_dim]

        # Apply normalization
        h = self.norm(h)  #  [batch_size, seq_length, embedding_dim]

        # convert the final output to float 32
        return self.output(
            h
        ).float()  # The shape remains [batch_size, seq_length, embedding_dim]


# we are norm norm many places which is a RMS norm let's implement it


import torch
from torch import nn, Tensor


class RMS_Norm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:

        super().__init__()
        self.scale = nn.Parameter(
            torch.ones(dim)
        )  # Scale parameter, learnable during training.
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply Root Mean Square Normalization to the input tensor.

        Args:
            x (Tensor): Input tensor to be normalized.

        Returns:
            Tensor: RMS normalized tensor.
        """
        input_dtype = x.dtype  # Save the original data type of the input tensor

        x = x.to(torch.float32)  # Convert input to float32 for stable computation

        # Compute the reciprocal of the root mean square of the input
        rms_a = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

        # Normalize the input by multiplying by the RMS value
        div = x * rms_a

        # Convert back to the original data type and apply the scale
        norm = div.to(input_dtype) * self.scale

        return norm


# okay we finished implementing all the blocks required for llama implementation from scratch except rope which i am directly copying from original llama implementation
def llama3(args: LLM_Args) -> TransformerDecoder:
    """
    Constructs a TransformerDecoder model based on provided arguments.

    Args:
        args (LLM_Args): Configuration object containing model config parameters

    Returns:
        TransformerDecoder: The constructed TransformerDecoder model.
    """
    # Calculate the dimension of each attention head
    head_dim = args.embed_dim // args.num_heads

    # Create Rotary Positional Embeddings for the model
    rope = RotaryPositionalEmbeddings(
        dim=head_dim, max_seq_len=args.max_seq_len, base=args.rope_base
    )

    # Define the self-attention mechanism with causal masking and rotary positional embeddings
    self_attn = CasualSelfAttention(args, rope)

    # Compute the dimension of the intermediate layer of the MLP, use provided or calculate based on embedding dimension
    hidden_dim = (
        args.intermediate_dim
        if args.intermediate_dim
        else scale_mlp_hidden_dim(dim=args.embed_dim)
    )

    # Construct the MLP using the calculated or provided intermediate dimension
    mlp = llama_mlp(dim=args.embed_dim, hidden_dim=hidden_dim)

    # Create a transformer decoder layer with attention, mlp and normalization
    layer = TransformerDecoderLayer(
        attn=self_attn,
        mlp=mlp,
        attn_norm=RMS_Norm(
            dim=args.embed_dim, eps=args.norm_eps
        ),  # Normalization for attention
        mlp_norm=RMS_Norm(
            dim=args.embed_dim, eps=args.norm_eps
        ),  # Normalization for MLP
    )

    # Embedding layer for converting token indices to vectors
    tok_embeddings = nn.Embedding(args.vocab_size, args.embed_dim)

    # Output projection layer from embedding dimension to vocabulary size
    output_proj = nn.Linear(args.embed_dim, args.vocab_size, bias=False)

    # Assemble the complete Transformer Decoder model
    model = TransformerDecoder(
        tok_embeddings=tok_embeddings,
        layer=layer,
        norm=RMS_Norm(
            dim=args.embed_dim, eps=args.norm_eps
        ),  # Apply normalization across the model output
        output=output_proj,
        args=args,
    )

    return model


# now lets create the config for lamm3 8b llm args defined above


def llama3_8b() -> TransformerDecoder:
    args = LLM_Args(
        num_layers=32,
        num_heads=32,
        embed_dim=4096,
        intermediate_dim=14336,
        vocab_size=128_256,
        max_seq_len=8192,
        num_kv_heads=8,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500000,
    )
    return llama3(args)
