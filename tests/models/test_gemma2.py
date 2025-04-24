"""
Unit tests for Gemma2 model (jaxgarden/models/gemma2.py)
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from jaxgarden.models.gemma2 import (
    Gemma2Attention,
    Gemma2Config,
    Gemma2ForCausalLM,
    Gemma2MLP,
    Gemma2RMSNorm,
    Gemma2RotaryEmbedding,
)


# Helper: minimal config for fast tests
@pytest.fixture(scope="module")  # Use module scope for efficiency
def tiny_config():
    return Gemma2Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,  # Must be even for GeGLU
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        context_length=16,
        param_dtype=jnp.float32,
        dtype=jnp.float32,
        attn_logits_soft_cap=50.0,
        final_logit_soft_cap=30.0,
    )


@pytest.fixture(scope="module")
def dummy_rngs():
    return nnx.Rngs(params=jax.random.PRNGKey(0))


# --- Test Core Modules ---


def test_rmsnorm_output(dummy_rngs):
    dim = 4
    norm = Gemma2RMSNorm(dim=dim, eps=1e-6, rngs=dummy_rngs)
    # Initialize dummy weights
    norm.weight = nnx.Param(jnp.ones((dim,), dtype=jnp.float32))  # Use ones for non-trivial test

    x = jnp.array([[0.1, 0.2, 0.3, 0.4]])
    out = norm(x)
    expected = jnp.array([[0.7302919, 1.4605838, 2.1908758, 2.9211676]])

    assert out.shape == (1, dim)
    assert jnp.allclose(out, expected, atol=1e-5)

    # Test with zero weights (as initialized in the module)
    norm.weight = nnx.Param(jnp.zeros((dim,), dtype=jnp.float32))
    x_ones = jnp.ones((2, dim))
    out_zeros_w = norm(x_ones)
    assert out_zeros_w.shape == (2, dim)
    # With input=ones and weight=zeros, output should be normalized input
    # (close to input / sqrt(mean(square(input))))
    # For ones input, mean(square(input)) = 1, sqrt = 1, output = input / 1 = 1
    assert jnp.allclose(out_zeros_w, jnp.ones_like(x_ones), atol=1e-6)


def test_rope_embedding():
    dim = 4
    batch = 2
    seq = 1
    num_heads = 2
    head_dim = dim // num_heads
    rope = Gemma2RotaryEmbedding(dim=dim // num_heads)

    # Shape [B, L, N, H] - Use head_dim here!
    x = jnp.ones((batch, seq, num_heads, head_dim))
    positions = jnp.array([[1], [0]])  # Example positions

    out = rope(x, positions)
    assert out.shape == x.shape

    # For pos=1, head=0: sin(1/100^(0/2))=sin(1)=0.841, cos(1)=0.540
    # RoPE applies rotations like [x0*c - x1*s, x0*s + x1*c]
    # Input is all 1s.
    # Expected head 0, pos 1: [1*c0-1*s0, 1*s0+1*c0] = [cos(1)-sin(1), cos(1)+sin(1)]
    expected_pos1 = jnp.array([-0.30116867, 1.38177329])
    # For pos=0, sin(0)=0, cos(0)=1. Output should be unchanged [1, 1]
    expected_pos0 = jnp.ones((head_dim,))

    # Check head 0 output
    np.testing.assert_allclose(out[0, 0, 0, :], expected_pos1, atol=1e-5)
    np.testing.assert_allclose(out[1, 0, 0, :], expected_pos0, atol=1e-5)


def test_attention_shape_and_gqa(tiny_config, dummy_rngs):
    # Initialize Gemma2Attention layer
    attn = Gemma2Attention(
        layer_idx=0,
        config=tiny_config,
        attention_type="global",
        rngs=dummy_rngs,
    )
    batch = 2
    seq = 8
    x = jnp.ones((batch, seq, tiny_config.hidden_size))
    pos_ids = jnp.arange(seq)[None, :].repeat(batch, axis=0)
    mask = jnp.tril(jnp.ones((seq, seq), dtype=jnp.bool_))[None, None, :, :]
    mask = jnp.where(mask, 0.0, -jnp.inf)  # Additive mask

    # Gemma2Attention.__call__ always returns (output, cache)
    out, _ = attn(x, pos_ids, attention_mask=mask, cache=None)
    assert out.shape == (batch, seq, tiny_config.hidden_size)

    # Test repeat_kv helper
    kv_heads = tiny_config.num_key_value_heads
    head_dim = tiny_config.head_dim
    num_attn_heads = tiny_config.num_attention_heads
    n_rep = tiny_config.num_attention_heads // kv_heads
    # Input shape should be (batch, kv_heads, seq, head_dim)
    hidden_kv = jnp.ones((batch, kv_heads, seq, head_dim))
    repeated = attn._repeat_kv(hidden_kv, n_rep)
    # Expected output shape: (batch, num_attn_heads, seq, head_dim)
    assert repeated.shape == (batch, num_attn_heads, seq, head_dim)


def test_attention_soft_cap(tiny_config, dummy_rngs):
    cap_value = 50.0
    config_with_cap = dataclasses.replace(tiny_config, attn_logits_soft_cap=cap_value)
    attn = Gemma2Attention(
        layer_idx=0,
        config=config_with_cap,
        attention_type="global",
        rngs=dummy_rngs,
    )

    logits = jnp.array([-100.0, -10.0, 0.0, 10.0, 100.0])
    capped_logits = attn.apply_soft_cap(logits, cap_value)

    expected = cap_value * jnp.tanh(logits / cap_value)
    np.testing.assert_allclose(capped_logits, expected, atol=1e-6)


def test_mlp_geglu_shape(tiny_config, dummy_rngs):
    mlp = Gemma2MLP(config=tiny_config, rngs=dummy_rngs)
    batch = 2
    seq = 8
    x = jnp.ones((batch, seq, tiny_config.hidden_size))
    out = mlp(x)
    assert out.shape == (batch, seq, tiny_config.hidden_size)

    # Test static geglu method
    intermediate_x = jnp.ones((batch, seq, tiny_config.intermediate_size * 2))
    geglu_out = Gemma2MLP.geglu(intermediate_x)
    assert geglu_out.shape == (batch, seq, tiny_config.intermediate_size)


# --- Test Decoder Layer ---


def test_decoder_layer_structure(tiny_config, dummy_rngs):
    layer_idx = 0
    model = Gemma2ForCausalLM(config=tiny_config, rngs=dummy_rngs)
    decoder = model.layers[layer_idx]

    assert isinstance(decoder.pre_attn_norm, Gemma2RMSNorm)
    assert isinstance(decoder.attn, Gemma2Attention)
    assert isinstance(decoder.post_attn_norm, Gemma2RMSNorm)
    assert isinstance(decoder.pre_mlp_norm, Gemma2RMSNorm)
    assert isinstance(decoder.mlp, Gemma2MLP)
    assert isinstance(decoder.post_mlp_norm, Gemma2RMSNorm)

    # Check attention type alternation
    assert decoder.attn.attention_type == "global"
    layer_idx = 1
    decoder1 = model.layers[layer_idx]
    assert decoder1.attn.attention_type == "local"


# --- Test Full Model ---


def test_gemma2_init(tiny_config, dummy_rngs):
    model = Gemma2ForCausalLM(config=tiny_config, rngs=dummy_rngs)
    assert isinstance(model, Gemma2ForCausalLM)
    assert hasattr(model, "embed_tokens")
    assert hasattr(model, "layers")
    assert len(model.layers) == tiny_config.num_hidden_layers
    assert hasattr(model, "norm")


def test_gemma2_forward_shape_dtype(tiny_config, dummy_rngs):
    model = Gemma2ForCausalLM(config=tiny_config, rngs=dummy_rngs)
    batch = 2
    seq = 8
    input_ids = jnp.ones((batch, seq), dtype=jnp.int32)
    pos_ids = jnp.arange(seq)[None, :].repeat(batch, axis=0)
    # Pass None for attention_mask; model should handle creation
    attn_mask = None

    out, cache = model(input_ids, pos_ids, attn_mask)

    assert out.shape == (batch, seq, tiny_config.vocab_size)
    assert out.dtype == tiny_config.dtype
    assert cache is not None  # Cache returned in forward pass


def test_final_logit_soft_cap(tiny_config, dummy_rngs):
    # Test with and without final soft cap
    config_no_cap = dataclasses.replace(tiny_config, final_logit_soft_cap=None)
    config_with_cap = dataclasses.replace(tiny_config, final_logit_soft_cap=30.0)

    model_no_cap = Gemma2ForCausalLM(config=config_no_cap, rngs=dummy_rngs)
    model_with_cap = Gemma2ForCausalLM(config=config_with_cap, rngs=dummy_rngs)

    nnx.update(model_with_cap, nnx.state(model_no_cap, nnx.Param))

    batch = 1
    seq = 4
    input_ids = jnp.ones((batch, seq), dtype=jnp.int32)
    pos_ids = jnp.arange(seq)[None, :].repeat(batch, axis=0)
    attn_mask = jnp.ones((batch, seq), dtype=jnp.bool_)

    out, _ = model_with_cap(input_ids, pos_ids, attn_mask)
    final_logits = out[:, -1, :]

    assert out.shape == (batch, seq, tiny_config.vocab_size)
    assert jnp.max(jnp.abs(final_logits)) <= config_with_cap.final_logit_soft_cap + 1e-6
