"""Tests for the jax_layers/functional/attention.py implementations."""

from unittest.mock import MagicMock

import jax
import jax.numpy as jnp

from jax_layers.functional import dot_product_attention


def test_dot_product_attention():
    """Test that the dot_product_attention function works with different implementations."""
    batch_size = 2
    seq_len = 16
    num_heads = 4
    head_dim = 8

    key = jax.random.PRNGKey(0)
    key1, key2, key3 = jax.random.split(key, 3)

    query = jax.random.normal(key1, (batch_size, seq_len, num_heads, head_dim))
    key_tensor = jax.random.normal(key2, (batch_size, seq_len, num_heads, head_dim))
    value = jax.random.normal(key3, (batch_size, seq_len, num_heads, head_dim))

    # Execute using default (XLA) implementation
    output_default = dot_product_attention(query, key_tensor, value)
    output_xla = dot_product_attention(query, key_tensor, value, implementation="xla")

    # Verify that the output shapes and values are identical
    assert output_default.shape == output_xla.shape
    assert jnp.allclose(output_default, output_xla, rtol=1e-5, atol=1e-5)


def test_dot_product_attention_flash_mapping(monkeypatch):
    """Test that when implementation is 'flash', it is remapped to 'cudnn'
    before calling jax.nn.dot_product_attention."""

    mock_fn = MagicMock(return_value=jnp.array(0))
    monkeypatch.setattr(jax.nn, "dot_product_attention", mock_fn)

    key = jax.random.PRNGKey(42)
    key1, key2, key3 = jax.random.split(key, 3)

    query = jax.random.normal(key1, (2, 16, 4, 8))
    key_tensor = jax.random.normal(key2, (2, 16, 4, 8))
    value = jax.random.normal(key3, (2, 16, 4, 8))

    _ = dot_product_attention(query, key_tensor, value, implementation="flash")

    mock_fn.assert_called_once_with(
        query=query,
        key=key_tensor,
        value=value,
        bias=None,
        scale=None,
        is_causal=False,
        query_seq_lengths=None,
        key_value_seq_lengths=None,
        local_window_size=None,
        implementation="cudnn",
    )


def test_dot_product_attention_mask_without_bias():
    """Test the mask branch when bias is None:
    bias should be created as jnp.where(mask, 0.0, -1e10)."""
    batch_size = 2
    seq_len = 16
    num_heads = 4
    head_dim = 8

    key = jax.random.PRNGKey(1)
    key1, key2, key3 = jax.random.split(key, 3)

    query = jax.random.normal(key1, (batch_size, seq_len, num_heads, head_dim))
    key_tensor = jax.random.normal(key2, (batch_size, seq_len, num_heads, head_dim))
    value = jax.random.normal(key3, (batch_size, seq_len, num_heads, head_dim))

    # Create a causal mask with boolean values.
    mask = jnp.tril(jnp.ones((batch_size, 1, seq_len, seq_len), dtype=bool))

    # Compute output using the mask (without providing bias explicitly).
    output_with_mask = dot_product_attention(query, key_tensor, value, mask=mask)

    # Manually create the bias as per conversion: if mask is True -> 0.0, else -1e10.
    bias_manual = jnp.where(mask, 0.0, -1e10)
    output_with_manual_bias = dot_product_attention(query, key_tensor, value, bias=bias_manual)

    assert output_with_mask.shape == output_with_manual_bias.shape
    assert jnp.allclose(output_with_mask, output_with_manual_bias, rtol=1e-5, atol=1e-5)


def test_dot_product_attention_mask_with_bias():
    """Test the mask branch when bias is provided:
    bias should be converted using jnp.where(mask, bias, -1e10)."""
    batch_size = 2
    seq_len = 16
    num_heads = 4
    head_dim = 8

    key = jax.random.PRNGKey(2)
    key1, key2, key3 = jax.random.split(key, 3)

    query = jax.random.normal(key1, (batch_size, seq_len, num_heads, head_dim))
    key_tensor = jax.random.normal(key2, (batch_size, seq_len, num_heads, head_dim))
    value = jax.random.normal(key3, (batch_size, seq_len, num_heads, head_dim))

    # Create a causal mask with boolean values.
    mask = jnp.tril(jnp.ones((batch_size, 1, seq_len, seq_len), dtype=bool))
    # Provide a custom bias (e.g., constant value 5.0).
    custom_bias = jnp.full(mask.shape, 5.0)

    # Expected bias after conversion: jnp.where(mask, custom_bias, -1e10)
    bias_manual = jnp.where(mask, custom_bias, -1e10)

    output_with_mask_bias = dot_product_attention(
        query,
        key_tensor,
        value,
        mask=mask,
        bias=custom_bias,
    )
    output_with_manual_bias = dot_product_attention(query, key_tensor, value, bias=bias_manual)

    assert output_with_mask_bias.shape == output_with_manual_bias.shape
    assert jnp.allclose(output_with_mask_bias, output_with_manual_bias, rtol=1e-5, atol=1e-5)
