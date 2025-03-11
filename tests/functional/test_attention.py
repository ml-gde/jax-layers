"""Tests for the jax_layers/functional/attention.py implementations."""

import jax
import jax.numpy as jnp

from jax_layers.functional import dot_product_attention


def test_dot_product_attention():
    """Test that the dot_product_attention function works with different implementations."""
    # Set up parameters
    batch_size = 2
    seq_len = 16
    num_heads = 4
    head_dim = 8

    # Create random input data
    key = jax.random.PRNGKey(0)
    key1, key2, key3 = jax.random.split(key, 3)
    query = jax.random.normal(key1, (batch_size, seq_len, num_heads, head_dim))
    key_tensor = jax.random.normal(key2, (batch_size, seq_len, num_heads, head_dim))
    value = jax.random.normal(key3, (batch_size, seq_len, num_heads, head_dim))

    # Apply dot_product_attention with different implementations
    output_default = dot_product_attention(query, key_tensor, value)
    output_xla = dot_product_attention(query, key_tensor, value, implementation="xla")

    # Check that the outputs are the same (should be deterministic with same inputs)
    assert jnp.allclose(output_default, output_xla, rtol=1e-5, atol=1e-5)
