"""Tests for the RoPEMultiHeadAttention class."""

import flax.linen as nn
import jax
import jax.numpy as jnp
import pytest

from jaxgarden.attention.rope_multi_head_attention import (
    RoPEMultiHeadAttention,
    apply_rotary_pos_emb,
    precompute_rotary_embeddings,
    rotate_half,
)


def test_rotate_half():
    """Tests the rotate_half function."""
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (2, 4, 6, 8))  # batch, seq, heads, dim
    rotated_x = rotate_half(x)

    assert rotated_x.shape == x.shape
    # Check specific values after rotation
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    expected = jnp.concatenate((-x2, x1), axis=-1)
    assert jnp.allclose(rotated_x, expected)


def test_precompute_rotary_embeddings():
    """Tests the precompute_rotary_embeddings function."""
    seq_len = 16
    head_dim = 8
    base = 10000.0

    cos_emb, sin_emb = precompute_rotary_embeddings(seq_len, head_dim, base)

    assert cos_emb.shape == (1, seq_len, 1, head_dim)
    assert sin_emb.shape == (1, seq_len, 1, head_dim)

    # Check properties - e.g., cos^2 + sin^2 = 1
    assert jnp.allclose(cos_emb**2 + sin_emb**2, jnp.ones_like(cos_emb), atol=1e-6)

    # Check different base value
    cos_emb_b2, sin_emb_b2 = precompute_rotary_embeddings(seq_len, head_dim, base=500.0)
    assert not jnp.allclose(cos_emb, cos_emb_b2)

    # Test with odd head_dim (should raise error)
    with pytest.raises(ValueError, match="head_dim must be even"):
        precompute_rotary_embeddings(seq_len, head_dim=7)


def test_apply_rotary_pos_emb():
    """Tests the apply_rotary_pos_emb function."""
    key = jax.random.PRNGKey(1)
    batch, seq_len, num_heads, head_dim = 2, 16, 4, 8
    x = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))

    cos_emb, sin_emb = precompute_rotary_embeddings(seq_len, head_dim)

    rotated_x = apply_rotary_pos_emb(x, cos_emb, sin_emb)

    assert rotated_x.shape == x.shape
    # Applying RoPE again should not give the original x (unless pos=0, which isn't the whole seq)
    assert not jnp.allclose(rotated_x, x)


# --- Test RoPEMultiHeadAttention Module ---


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_rope_mha_forward_pass(dtype):
    """Tests the forward pass of RoPEMultiHeadAttention."""
    key = jax.random.PRNGKey(2)
    batch_size = 2
    seq_len = 16
    num_heads = 4
    head_dim = 8
    embed_dim = num_heads * head_dim

    x = jax.random.normal(key, (batch_size, seq_len, embed_dim), dtype=dtype)

    rope_mha = RoPEMultiHeadAttention(num_heads=num_heads, head_dim=head_dim, dtype=dtype)
    params = rope_mha.init(key, x)["params"]
    output = rope_mha.apply({"params": params}, x)

    assert output.shape == (batch_size, seq_len, embed_dim)
    assert output.dtype == dtype


def test_rope_mha_masking():
    """Tests causal masking in RoPEMultiHeadAttention."""
    key = jax.random.PRNGKey(3)
    batch_size = 1
    seq_len = 4
    num_heads = 2
    head_dim = 4
    embed_dim = num_heads * head_dim

    x = jax.random.normal(key, (batch_size, seq_len, embed_dim))
    # Create a causal mask (True means masked)
    causal_mask = nn.make_causal_mask(x[:, :, 0])  # Gets (batch, seq, seq) or (1, seq, seq)

    rope_mha = RoPEMultiHeadAttention(num_heads=num_heads, head_dim=head_dim)
    params = rope_mha.init(key, x, causal_mask)["params"]

    # Apply without mask
    output_unmasked = rope_mha.apply({"params": params}, x)

    # Apply with mask
    output_masked = rope_mha.apply({"params": params}, x, mask=causal_mask)

    # Basic check: outputs should differ if mask has an effect
    assert not jnp.allclose(output_unmasked, output_masked, atol=1e-5)

    # More rigorous check (requires inspecting attention weights, omitted for brevity)


def test_rope_mha_errors():
    """Tests error conditions for RoPEMultiHeadAttention."""
    key = jax.random.PRNGKey(4)
    rope_mha_odd_dim = RoPEMultiHeadAttention(num_heads=8, head_dim=7)
    x_dummy_odd = jax.random.normal(key, (2, 16, 8 * 7))
    # Test with odd head_dim (should raise error during initialization/setup)
    with pytest.raises(ValueError, match=r"head_dim \(\d+\) must be even"):
        rope_mha_odd_dim.init(key, x_dummy_odd)

    # Test with mismatched embed_dim (should raise error during forward pass / init)
    rope_mha = RoPEMultiHeadAttention(num_heads=4, head_dim=8)  # Expects embed_dim 32
    x_mismatch = jax.random.normal(key, (2, 16, 100))  # Incorrect embed_dim

    with pytest.raises(ValueError, match=r"embed_dim \(\d+\) must equal"):
        rope_mha.init(key, x_mismatch)
