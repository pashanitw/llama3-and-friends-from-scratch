import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional
from dataclasses import dataclass
from torch import Tensor

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


class CasualSelfAttention2(nn.Module):
    def __init__(
        self,
        args: LLM_Args,
        pos_embeddings: nn.Module,
        kv_cache: Optional[KVCache] = None,
    ):
        super().__init__()

        assert (
            args.num_heads % args.num_kv_heads == 0
        ), "Number of attention heads must be divisible by the number of key-value attention heads"
        assert (
            args.embed_dim % args.num_heads == 0
        ), "Embedding dimension must be divisible by the number of attention heads"
        assert (
            0 <= args.attn_dropout <= 1
        ), "Attention dropout value must be between 0.0 and 1.0 (inclusive)"

        self.head_dim = args.embed_dim // args.num_heads
        self.num_heads = args.num_heads
        self.num_kv_heads = args.num_kv_heads or args.num_heads
        self.embed_dim = args.embed_dim
        self.attn_dropout = args.attn_dropout
        self.max_seq_len = args.max_seq_len

        self.q_proj = nn.Linear(
            args.embed_dim, self.embed_dim, bias=args.attention_bias
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

        self.pos_embeddings = pos_embeddings
        self.kv_cache = kv_cache

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        input_pos: Optional[torch.Tensor],
    ):
        bsz, seq_len, _ = x.shape
        assert (
            seq_len <= self.max_seq_len
        ), f"Sequence length ({seq_len}) exceeds the maximum allowed length ({self.max_seq_len})"

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = rearrange(q, "B S (H D) -> B H S D", H=self.num_heads)
        k = rearrange(k, "B S (H D) -> B H S D", H=self.num_kv_heads)
        v = rearrange(v, "B S (H D) -> B H S D", H=self.num_kv_heads)

        if self.num_kv_heads < self.num_heads:
            k = repeat(
                k, "B H S D -> B (H R) S D", R=self.num_heads // self.num_kv_heads
            )
            v = repeat(
                v, "B H S D -> B (H R) S D", R=self.num_heads // self.num_kv_heads
            )

        q = self.pos_embeddings(q, input_pos)
        k = self.pos_embeddings(k, input_pos)

        if self.kv_cache:
            k, v = self.kv_cache.update(input_pos, k, v)

        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.attn_dropout,
            is_causal=self.kv_cache is None,
        )

        output = rearrange(output, "B H S D -> B S (H D)")
        return self.o_proj(output)


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
