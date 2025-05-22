import jax.numpy as jnp
import pytest
from flax import nnx

from jaxgarden.models.t5 import (
    T5MLP,
    T5Attention,
    T5Config,
    T5CrossAttention,
    T5LayerNorm,
    T5SelfAttention,
)


@pytest.mark.parametrize(("dim", "dtype"), [(1024, jnp.float32), (2048, jnp.float16)])
def test_t5_layer_norm(dim, dtype):
    layer_norm = T5LayerNorm(dim=dim, dtype=dtype, rngs=nnx.Rngs(0))
    x = jnp.ones((1, dim))
    output = layer_norm(x)

    assert output.shape == (1, dim)
    assert output.dtype == dtype


@pytest.mark.parametrize(
    ("dim", "intermediate_dim", "dtype"), [(512, 2048, jnp.float32), (1024, 4096, jnp.float16)]
)
def test_t5_mlp(dim, intermediate_dim, dtype):
    # small sequence length for testing
    seq_len = 128

    mlp = T5MLP(dim=dim, intermediate_dim=intermediate_dim, dtype=dtype, rngs=nnx.Rngs(0))
    x = jnp.ones((1, seq_len, dim))
    output = mlp(x)

    assert output.shape == (1, seq_len, dim)
    assert output.dtype == dtype


@pytest.mark.parametrize(
    ("hidden_size", "dim_kv", "dtype", "masked"),
    [(768, 64, jnp.float32, True), (1024, 128, jnp.float16, False)],
)
def test_t5_attention(hidden_size, dim_kv, dtype, masked):
    # small sequence length for testing
    seq_len = 128

    attention = T5Attention(
        hidden_size=hidden_size,
        dim_kv=dim_kv,
        num_heads=hidden_size // dim_kv,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        dtype=dtype,
        rngs=nnx.Rngs(0),
    )

    hidden_states = jnp.ones((1, seq_len, hidden_size))
    attention_mask = jnp.ones((1, seq_len)) if masked else None
    output = attention(hidden_states, attention_mask=attention_mask)

    assert output.shape == (1, seq_len, hidden_size)
    assert output.dtype == dtype


@pytest.mark.parametrize(
    ("hidden_size", "dim_kv", "dtype", "masked"),
    [(768, 64, jnp.float32, True), (1024, 128, jnp.float16, False)],
)
def test_t5_self_attention(hidden_size, dim_kv, dtype, masked):
    # small sequence length for testing
    seq_len = 128

    config = T5Config(hidden_size=hidden_size, dim_kv=dim_kv, dtype=dtype)
    self_attention = T5SelfAttention(config=config, rngs=nnx.Rngs(0))

    hidden_states = jnp.ones((1, seq_len, hidden_size))
    attention_mask = jnp.ones((1, seq_len)) if masked else None

    output = self_attention(hidden_states, attention_mask=attention_mask)

    assert output.shape == (1, seq_len, hidden_size)
    assert output.dtype == dtype


@pytest.mark.parametrize(
    ("hidden_size", "dim_kv", "dtype", "masked"),
    [(768, 64, jnp.float32, True), (1024, 128, jnp.float16, False)],
)
def test_t5_cross_attention(hidden_size, dim_kv, dtype, masked):
    # small sequence length for testing
    seq_len = 128

    config = T5Config(hidden_size=hidden_size, dim_kv=dim_kv, dtype=dtype)
    cross_attention = T5CrossAttention(config=config, rngs=nnx.Rngs(0))

    hidden_states = jnp.ones((1, seq_len, hidden_size))
    attention_mask = jnp.ones((1, seq_len)) if masked else None

    output = cross_attention(hidden_states, attention_mask=attention_mask)

    assert output.shape == (1, seq_len, hidden_size)
    assert output.dtype == dtype
