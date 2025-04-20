"""Tests for LLama model."""

import jax
import jax.numpy as jnp
from flax import nnx

from jaxgarden.models.llama import LlamaConfig, LlamaForCausalLM


def test_llama_initialization():
    """Test that the LLama model can be properly initialized."""
    # Set configuration for a tiny model for testing
    config = LlamaConfig(
        dim=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=8,
        intermediate_size=64,
        vocab_size=100,
    )

    # Initialize the model
    key = jax.random.PRNGKey(0)
    rngs = nnx.Rngs(params=key)

    model = LlamaForCausalLM(config, rngs=rngs)

    # Verify the model was created with expected structure
    assert len(model.layers) == 2
    assert isinstance(model, LlamaForCausalLM)


def test_llama_inference():
    """Test that the LLama model can run inference end-to-end."""
    # Set configuration for a tiny model for testing
    config = LlamaConfig(
        dim=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=8,
        intermediate_size=64,
        vocab_size=100,
    )

    # Initialize the model
    key = jax.random.PRNGKey(0)
    rngs = nnx.Rngs(params=key)

    model = LlamaForCausalLM(config, rngs=rngs)

    # Create sample input
    batch_size = 1
    seq_len = 4
    input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    position_ids = jnp.arange(seq_len)[None, :]

    # Run forward pass
    logits = model(input_ids, position_ids)

    # Check output shape
    assert logits.shape == (batch_size, seq_len, config.vocab_size)
