"""Tests for ModernBERT components."""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional
import pytest
import torch
import torch.nn as nn
import flax.nnx as nnx

from jax_layers.models.modernbert import (
    RoPEPositionalEmbedding,
    create_sinusoidal_positions,
    apply_rotary_pos_emb,
)


class RotaryPositionalEmbeddingsTorch(nn.Module):
    """PyTorch reference implementation from torchtun."""

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10_000,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.rope_init()

    def rope_init(self):
        theta = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache()

    def build_rope_cache(self) -> None:
        seq_idx = torch.arange(self.max_seq_len, dtype=torch.float32)
        idx_theta = torch.einsum("i,j->ij", seq_idx, self.theta)
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x: torch.Tensor, *, input_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_len = x.size(1)
        rope_cache = self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )
        x_out = x_out.flatten(3)
        return x_out.type_as(x)


def test_create_sinusoidal_positions():
    """Test creation of sinusoidal position embeddings."""
    max_length = 16
    dim = 64
    base = 10000.0

    # Create embeddings using our implementation
    cache_jax = create_sinusoidal_positions(max_length, dim, base)

    # Create embeddings using PyTorch implementation
    rope_torch = RotaryPositionalEmbeddingsTorch(dim=dim, max_seq_len=max_length, base=base)
    cache_torch = rope_torch.cache.numpy()

    # Compare results
    np.testing.assert_allclose(cache_jax, cache_torch, atol=1e-6)


def test_apply_rotary_pos_emb():
    """Test application of rotary position embeddings."""
    batch_size = 2
    seq_len = 16
    num_heads = 8
    head_dim = 64

    # Create random input
    key = jax.random.PRNGKey(0)
    x_jax = jax.random.normal(key, (batch_size, seq_len, num_heads, head_dim))
    x_np = np.array(x_jax)  # Convert to numpy first

    # Apply RoPE using our implementation
    cache_jax = create_sinusoidal_positions(seq_len, head_dim)
    out_jax = np.array(apply_rotary_pos_emb(x_jax, cache_jax))  # Convert to numpy

    # Apply RoPE using PyTorch implementation
    rope_torch = RotaryPositionalEmbeddingsTorch(dim=head_dim, max_seq_len=seq_len)
    x_torch = torch.from_numpy(x_np).float()  # Ensure float32
    out_torch = rope_torch(x_torch)

    # Compare results
    np.testing.assert_allclose(out_jax, out_torch.numpy(), atol=1e-6)


def test_rope_module():
    """Test the full RoPE module."""
    batch_size = 2
    seq_len = 16
    num_heads = 8
    head_dim = 64

    # Create random input
    key = jax.random.PRNGKey(0)
    x_jax = jax.random.normal(key, (batch_size, seq_len, num_heads, head_dim))
    x_np = np.array(x_jax)  # Convert to numpy first

    # Initialize and apply our RoPE
    rope_jax = RoPEPositionalEmbedding(rngs=nnx.Rngs(0), dim=head_dim)
    out_jax = np.array(rope_jax(x_jax))  # Convert to numpy

    # Apply PyTorch RoPE
    rope_torch = RotaryPositionalEmbeddingsTorch(dim=head_dim, max_seq_len=seq_len)
    x_torch = torch.from_numpy(x_np).float()  # Ensure float32
    out_torch = rope_torch(x_torch)

    # Compare results
    np.testing.assert_allclose(out_jax, out_torch.numpy(), atol=1e-6)


def test_rope_with_positions():
    """Test RoPE with custom position indices."""
    batch_size = 2
    seq_len = 16
    num_heads = 8
    head_dim = 64

    # Create random input and positions
    key = jax.random.PRNGKey(0)
    x_jax = jax.random.normal(key, (batch_size, seq_len, num_heads, head_dim))
    x_np = np.array(x_jax)  # Convert to numpy first

    positions_jax = jnp.array([[3, 7, 1, 4] + [0] * (seq_len - 4)] * batch_size)
    positions_np = np.array(positions_jax)  # Convert to numpy

    # Initialize and apply our RoPE
    rope_jax = RoPEPositionalEmbedding(rngs=nnx.Rngs(0), dim=head_dim)
    out_jax = np.array(rope_jax(x_jax, positions=positions_jax))  # Convert to numpy

    # Apply PyTorch RoPE
    max_pos = int(positions_np.max()) + 1  # Convert to int for PyTorch
    rope_torch = RotaryPositionalEmbeddingsTorch(dim=head_dim, max_seq_len=max_pos)
    x_torch = torch.from_numpy(x_np).float()  # Ensure float32
    positions_torch = torch.from_numpy(positions_np).long()  # Convert to long for indexing
    out_torch = rope_torch(x_torch, input_pos=positions_torch)

    # Compare results
    np.testing.assert_allclose(out_jax, out_torch.numpy(), atol=1e-6)
