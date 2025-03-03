"""Tests for the MultiHeadAttention class."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from jax_layers.attention import MultiHeadAttention
from jax_layers.functional import dot_product_attention


def test_multi_head_attention_shape():
    """Test that the MultiHeadAttention module produces the expected output shape."""
    # Set up parameters
    batch_size = 2
    seq_len = 16
    num_heads = 4
    head_dim = 8
    hidden_dim = num_heads * head_dim

    # Create random input data
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    x = jax.random.normal(key1, (batch_size, seq_len, hidden_dim))

    # Create the MultiHeadAttention module
    attention = MultiHeadAttention(
        num_heads=num_heads,
        in_features=hidden_dim,
        decode=False,
        rngs=nnx.Rngs(key2),
    )

    # Apply the attention
    output = attention(x, x, x)

    # Check the output shape
    assert output.shape == (batch_size, seq_len, hidden_dim)


def test_multi_head_attention_mask():
    """Test that the MultiHeadAttention module correctly applies attention masks."""
    # Set up parameters
    batch_size = 2
    seq_len = 16
    num_heads = 4
    head_dim = 8
    hidden_dim = num_heads * head_dim

    # Create random input data
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    x = jax.random.normal(key1, (batch_size, seq_len, hidden_dim))

    # Create a causal attention mask
    mask = jnp.tril(jnp.ones((batch_size, 1, seq_len, seq_len)))

    # Create the MultiHeadAttention module
    attention = MultiHeadAttention(
        num_heads=num_heads,
        in_features=hidden_dim,
        decode=False,
        rngs=nnx.Rngs(key2),
    )

    # Apply the attention with and without mask
    output_with_mask = attention(x, x, x, mask=mask)
    output_without_mask = attention(x, x, x)

    # Check that the outputs are different
    assert not jnp.allclose(output_with_mask, output_without_mask)


def test_multi_head_attention_implementation():
    """Test that the MultiHeadAttention module works with different implementations."""
    # Set up parameters
    batch_size = 2
    seq_len = 16
    num_heads = 4
    head_dim = 8
    hidden_dim = num_heads * head_dim

    # Create random input data
    key = jax.random.PRNGKey(0)
    key1, key2, key3 = jax.random.split(key, 3)
    x = jax.random.normal(key1, (batch_size, seq_len, hidden_dim))

    # Create the MultiHeadAttention modules with different implementations
    attention_default = MultiHeadAttention(
        num_heads=num_heads,
        in_features=hidden_dim,
        decode=False,
        rngs=nnx.Rngs(key2),
    )

    attention_xla = MultiHeadAttention(
        num_heads=num_heads,
        in_features=hidden_dim,
        decode=False,
        implementation="xla",
        rngs=nnx.Rngs(key3),
    )

    # Apply the attention with different implementations
    output_default = attention_default(x, x, x)
    output_xla = attention_xla(x, x, x)

    # Check that the outputs are similar (not exactly the same due to different initializations)
    assert output_default.shape == output_xla.shape, f"{type(output_xla)}, {len(output_xla)}"


def test_dot_product_attention_implementation():
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


def test_multi_head_attention_self_attention():
    """Test that the MultiHeadAttention module works as a self-attention module."""
    # Set up parameters
    batch_size = 2
    seq_len = 16
    num_heads = 4
    head_dim = 8
    hidden_dim = num_heads * head_dim

    # Create random input data
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    x = jax.random.normal(key1, (batch_size, seq_len, hidden_dim))

    # Create the MultiHeadAttention module
    attention = MultiHeadAttention(
        num_heads=num_heads,
        in_features=hidden_dim,
        decode=False,
        rngs=nnx.Rngs(key2),
    )

    # Apply the attention as self-attention
    output = attention(x)

    # Check the output shape
    assert output.shape == (batch_size, seq_len, hidden_dim)


def test_multi_head_attention_cross_attention():
    """Test that the MultiHeadAttention module works as a cross-attention module."""
    # Set up parameters
    batch_size = 2
    q_seq_len = 16
    kv_seq_len = 32
    num_heads = 4
    head_dim = 8
    hidden_dim = num_heads * head_dim

    # Create random input data
    key = jax.random.PRNGKey(0)
    key1, key2, key3, key4 = jax.random.split(key, 4)
    q = jax.random.normal(key1, (batch_size, q_seq_len, hidden_dim))
    k = jax.random.normal(key2, (batch_size, kv_seq_len, hidden_dim))
    v = jax.random.normal(key3, (batch_size, kv_seq_len, hidden_dim))

    # Create the MultiHeadAttention module
    attention = MultiHeadAttention(
        num_heads=num_heads,
        in_features=hidden_dim,
        decode=False,
        rngs=nnx.Rngs(key4),
    )

    # Apply the attention as cross-attention
    output = attention(q, k, v)

    # Check the output shape
    assert output.shape == (batch_size, q_seq_len, hidden_dim)
