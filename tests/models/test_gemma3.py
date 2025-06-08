"""Tests for Gemma3 model."""

import jax.numpy as jnp
import numpy as np
from flax import nnx

from jaxgarden.models.gemma3 import (
    Gemma3Attention,
    Gemma3Config,
    Gemma3DecoderLayer,
    Gemma3ForCausalLM,
    Gemma3MLP,
    Gemma3RMSNorm,
    Gemma3RotaryEmbedding,
)


def test_gemma3_config():
    """Test Gemma3Config initialization and validation."""
    config = Gemma3Config()
    assert config.num_attention_heads % config.num_key_value_heads == 0
    assert config.head_dim == 256  # Default head_dim


def test_gemma3_rms_norm():
    """Test RMSNorm layer."""
    rng = nnx.Rngs(0)
    dim = 32
    batch_size = 2
    seq_len = 4

    norm = Gemma3RMSNorm(dim, eps=1e-6, rngs=rng)
    x = jnp.ones((batch_size, seq_len, dim), dtype=jnp.float32)
    out = norm(x)

    assert out.shape == x.shape
    # Output should be normalized
    variance = jnp.mean(jnp.square(out), axis=-1)
    np.testing.assert_allclose(variance, 1.0, rtol=1e-5)


def test_gemma3_rotary_embedding():
    """Test RotaryEmbedding module."""
    dim = 32
    batch_size = 2
    seq_len = 4
    num_heads = 2

    rope = Gemma3RotaryEmbedding(dim=dim, max_position_embeddings=8192)
    x = jnp.ones((batch_size, seq_len, num_heads, dim), dtype=jnp.float32)
    position_ids = jnp.arange(seq_len, dtype=jnp.int32)[None, :]  # [1, seq_len]
    position_ids = jnp.broadcast_to(position_ids, (batch_size, seq_len))

    out = rope(x, position_ids)
    assert out.shape == (batch_size, seq_len, num_heads, dim)
