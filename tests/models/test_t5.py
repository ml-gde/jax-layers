import jax.numpy as jnp
import pytest
from flax import nnx

from jaxgarden.models.t5 import T5MLP, T5LayerNorm


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
    seq_len = 256

    mlp = T5MLP(dim=dim, intermediate_dim=intermediate_dim, dtype=dtype, rngs=nnx.Rngs(0))
    x = jnp.ones((1, seq_len, dim))
    output = mlp(x)

    assert output.shape == (1, seq_len, dim)
    assert output.dtype == dtype
