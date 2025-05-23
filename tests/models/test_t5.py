import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from jaxgarden.models.t5 import (
    T5MLP,
    T5Attention,
    T5Block,
    T5Config,
    T5CrossAttention,
    T5ForCausalLM,
    T5LayerNorm,
    T5SelfAttention,
    T5Stack,
)


@pytest.fixture(scope="module")
def tiny_config():
    return T5Config(
        hidden_size=128,
        dim_kv=64,
        num_layers=2,
        vocab_size=100,
        dtype=jnp.float32,
    )


@pytest.fixture(scope="module")
def dummy_rngs():
    return nnx.Rngs(params=jax.random.PRNGKey(0))


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


@pytest.mark.parametrize(
    ("hidden_size", "dim_kv", "dtype", "masked", "causal"),
    [
        (768, 64, jnp.float32, True, True),
        (768, 64, jnp.float32, False, False),
        (1024, 128, jnp.float16, True, False),
        (1024, 128, jnp.float16, False, True),
    ],
)
def test_t5_block(hidden_size, dim_kv, dtype, masked, causal):
    # small sequence length for testing
    seq_len = 128

    config = T5Config(hidden_size=hidden_size, dim_kv=dim_kv, dtype=dtype)
    block = T5Block(config=config, causal=causal, rngs=nnx.Rngs(0))

    hidden_states = jnp.ones((1, seq_len, hidden_size))
    attention_mask = jnp.ones((1, seq_len)) if masked else None

    output = block(hidden_states, attention_mask=attention_mask, deterministic=True)

    assert output.shape == (1, seq_len, hidden_size)
    assert output.dtype == dtype


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float16])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("masked", [True, False])
@pytest.mark.parametrize("with_encoder", [True, False])
def test_t5_stack(dtype, causal, masked, with_encoder):
    hidden_size = 768
    dim_kv = 64
    vocab_size = 100
    num_layers = 2
    seq_len = 8
    batch_size = 2

    config = T5Config(hidden_size=hidden_size, dim_kv=dim_kv, dtype=dtype)
    embed_tokens = nnx.Embed(
        num_embeddings=vocab_size,
        features=hidden_size,
        dtype=dtype,
        rngs=nnx.Rngs(0),
    )
    stack = T5Stack(
        config=config,
        embed_tokens=embed_tokens,
        num_layers=num_layers,
        causal=causal,
        rngs=nnx.Rngs(0),
    )

    input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    attention_mask = jnp.ones((batch_size, seq_len), dtype=dtype) if masked else None

    encoder_hidden_states = (
        jnp.ones((batch_size, seq_len, hidden_size), dtype=dtype) if with_encoder else None
    )
    encoder_attention_mask = (
        jnp.ones((batch_size, seq_len), dtype=dtype) if (with_encoder and masked) else None
    )

    output = stack(
        input_ids,
        attention_mask=attention_mask,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        deterministic=True,
    )

    assert output.shape == (batch_size, seq_len, hidden_size)
    assert output.dtype == dtype


# --- Test Full Model ---
def test_t5_for_causal_lm(tiny_config, dummy_rngs):
    # small values for testing
    seq_len = 8
    batch = 2

    model = T5ForCausalLM(config=tiny_config, rngs=dummy_rngs)

    input_ids = jnp.ones((batch, seq_len), dtype=jnp.int32)
    pos_ids = jnp.arange(seq_len)[None, :].repeat(batch, axis=0)
    attn_mask = None

    out = model(input_ids, pos_ids, attn_mask)

    assert out.shape == (batch, seq_len, tiny_config.vocab_size)
    assert out.dtype == tiny_config.dtype
