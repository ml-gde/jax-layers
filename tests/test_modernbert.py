"""Tests for ModernBERT components."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as nn

from jax_layers.models.modernbert import (
    ModernBertAttention,
    ModernBertEmbeddings,
    ModernBertMLP,
    RoPEPositionalEmbedding,
    apply_rotary_pos_emb,
    create_sinusoidal_positions,
    create_sliding_window_mask,
)


class RotaryPositionalEmbeddingsTorch(nn.Module):
    """PyTorch reference implementation from torchtun.

    See https://pytorch.org/torchtune/0.4/_modules/torchtune/modules/position_embeddings.html#RotaryPositionalEmbeddings
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10_000,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.rope_init()

    def rope_init(self) -> None:
        theta = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache()

    def build_rope_cache(self) -> None:
        seq_idx = torch.arange(self.max_seq_len, dtype=torch.float32)
        idx_theta = torch.einsum("i,j->ij", seq_idx, self.theta)
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x: torch.Tensor, *, input_pos: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = x.size(1)
        rope_cache = self.cache[:seq_len] if input_pos is None else self.cache[input_pos]  # type: ignore
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )
        x_out = x_out.flatten(3)
        return x_out.type_as(x)


def test_create_sinusoidal_positions():
    """Test creation of sinusoidal position embeddings."""
    max_length = 16
    dim = 64
    base = 10000.0

    # Create embeddings using our implementation
    cache_jax = create_sinusoidal_positions(max_length, dim, base)

    # Create embeddings using PyTorch implementation
    rope_torch = RotaryPositionalEmbeddingsTorch(dim=dim, max_seq_len=max_length, base=base)
    cache_torch = rope_torch.cache.numpy()

    # Compare results
    np.testing.assert_allclose(cache_jax, cache_torch, atol=1e-6)


def test_apply_rotary_pos_emb():
    """Test application of rotary position embeddings."""
    batch_size = 2
    seq_len = 16
    num_heads = 8
    head_dim = 64

    # Create random input
    key = jax.random.PRNGKey(0)
    x_jax = jax.random.normal(key, (batch_size, seq_len, num_heads, head_dim))
    x_np = np.array(x_jax)  # Convert to numpy first

    # Apply RoPE using our implementation
    cache_jax = create_sinusoidal_positions(seq_len, head_dim)
    out_jax = np.array(apply_rotary_pos_emb(x_jax, cache_jax))  # Convert to numpy

    # Apply RoPE using PyTorch implementation
    rope_torch = RotaryPositionalEmbeddingsTorch(dim=head_dim, max_seq_len=seq_len)
    x_torch = torch.from_numpy(x_np).float()  # Ensure float32
    out_torch = rope_torch(x_torch)

    # Compare results
    np.testing.assert_allclose(out_jax, out_torch.numpy(), atol=1e-6)


def test_rope_module():
    """Test the full RoPE module."""
    batch_size = 2
    seq_len = 16
    num_heads = 8
    head_dim = 64

    # Create random input
    key = jax.random.PRNGKey(0)
    x_jax = jax.random.normal(key, (batch_size, seq_len, num_heads, head_dim))
    x_np = np.array(x_jax)  # Convert to numpy first

    # Initialize and apply our RoPE
    rope_jax = RoPEPositionalEmbedding(rngs=nnx.Rngs(0), dim=head_dim)
    out_jax = np.array(rope_jax(x_jax))  # Convert to numpy

    # Apply PyTorch RoPE
    rope_torch = RotaryPositionalEmbeddingsTorch(dim=head_dim, max_seq_len=seq_len)
    x_torch = torch.from_numpy(x_np).float()  # Ensure float32
    out_torch = rope_torch(x_torch)

    # Compare results
    np.testing.assert_allclose(out_jax, out_torch.numpy(), atol=1e-6)


def test_rope_with_positions():
    """Test RoPE with custom position indices."""
    batch_size = 2
    seq_len = 16
    num_heads = 8
    head_dim = 64

    # Create random input and positions
    key = jax.random.PRNGKey(0)
    x_jax = jax.random.normal(key, (batch_size, seq_len, num_heads, head_dim))
    x_np = np.array(x_jax)  # Convert to numpy first

    positions_jax = jnp.array([[3, 7, 1, 4] + [0] * (seq_len - 4)] * batch_size)
    positions_np = np.array(positions_jax)  # Convert to numpy

    # Initialize and apply our RoPE
    rope_jax = RoPEPositionalEmbedding(rngs=nnx.Rngs(0), dim=head_dim)
    out_jax = np.array(rope_jax(x_jax, positions=positions_jax))  # Convert to numpy

    # Apply PyTorch RoPE
    max_pos = int(positions_np.max()) + 1  # Convert to int for PyTorch
    rope_torch = RotaryPositionalEmbeddingsTorch(dim=head_dim, max_seq_len=max_pos)
    x_torch = torch.from_numpy(x_np).float()  # Ensure float32
    positions_torch = torch.from_numpy(positions_np).long()  # Convert to long for indexing
    out_torch = rope_torch(x_torch, input_pos=positions_torch)

    # Compare results
    np.testing.assert_allclose(out_jax, out_torch.numpy(), atol=1e-6)


def test_embeddings():
    """Test token embeddings with normalization and dropout."""
    batch_size = 2
    seq_len = 16
    vocab_size = 1000
    hidden_size = 64

    # Create random input
    key = jax.random.PRNGKey(0)
    input_ids = jax.random.randint(key, (batch_size, seq_len), 0, vocab_size)

    # Initialize embeddings
    embeddings = ModernBertEmbeddings(
        rngs=nnx.Rngs(0),
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        embedding_dropout=0.1,
    )

    # Test forward pass with input_ids
    output = embeddings(input_ids, deterministic=True)
    assert output.shape == (batch_size, seq_len, hidden_size)

    # Test forward pass with pre-computed embeddings
    key_embeds = jax.random.PRNGKey(1)
    inputs_embeds = jax.random.normal(key_embeds, (batch_size, seq_len, hidden_size))
    output = embeddings(input_ids, inputs_embeds=inputs_embeds)
    assert output.shape == (batch_size, seq_len, hidden_size)

    # Test dropout is applied in training
    output_train = embeddings(input_ids, deterministic=False)
    assert not jnp.array_equal(output, output_train)  # Dropout should make them different

    # Test layer norm is working
    assert jnp.allclose(jnp.mean(output, axis=-1), 0.0, atol=1e-6)
    assert jnp.allclose(jnp.std(output, axis=-1), 1.0, atol=1e-1)


def test_mlp():
    """Test MLP with gated linear units."""
    batch_size = 2
    seq_len = 16
    hidden_size = 64
    intermediate_size = 256

    # Create random input
    key = jax.random.PRNGKey(0)
    hidden_states = jax.random.normal(key, (batch_size, seq_len, hidden_size))

    # Initialize MLP
    mlp = ModernBertMLP(
        rngs=nnx.Rngs(0),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        mlp_dropout=0.1,
    )

    # Test forward pass
    output = mlp(hidden_states, deterministic=True)
    assert output.shape == (batch_size, seq_len, hidden_size)

    # Test dropout is applied in training
    output_train = mlp(hidden_states, deterministic=False)
    assert not jnp.array_equal(output, output_train)  # Dropout should make them different

    # Test gating mechanism
    # Get intermediate activations
    combined = mlp.Wi(hidden_states)
    input_tensor, gate = jnp.split(combined, 2, axis=-1)

    # Verify shapes
    assert input_tensor.shape == (batch_size, seq_len, intermediate_size)
    assert gate.shape == (batch_size, seq_len, intermediate_size)

    # Test that gating is working (values should be between input and zero)
    activated = jax.nn.gelu(input_tensor)
    gated = activated * gate
    assert jnp.all(jnp.abs(gated) <= jnp.abs(activated))


def test_mlp_no_bias():
    """Test MLP without bias terms."""
    batch_size = 2
    seq_len = 16
    hidden_size = 64
    intermediate_size = 256

    # Create random input
    key = jax.random.PRNGKey(0)
    hidden_states = jax.random.normal(key, (batch_size, seq_len, hidden_size))

    # Initialize MLP without bias
    mlp = ModernBertMLP(
        rngs=nnx.Rngs(0),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        mlp_bias=False,
    )

    # Test forward pass
    output = mlp(hidden_states)
    assert output.shape == (batch_size, seq_len, hidden_size)

    # Verify no bias terms exist
    assert mlp.Wi.bias is None
    assert mlp.Wo.bias is None


def rotate_half_torch(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_torch(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """PyTorch reference implementation of RoPE."""
    # Ensure cos and sin have shape [seq_len, 1, head_dim]
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half_torch(q) * sin)
    k_embed = (k * cos) + (rotate_half_torch(k) * sin)
    return q_embed, k_embed


def test_rope_numerical_correctness():
    """Test that our JAX RoPE implementation matches the PyTorch reference."""
    # Test parameters
    batch_size = 2
    seq_len = 16
    num_heads = 4
    head_dim = 64
    max_position_embeddings = 512
    base = 10000.0

    # Create random input tensors
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)

    # Create random query and key tensors
    q_jax = jax.random.normal(key1, (batch_size, seq_len, num_heads, head_dim))
    k_jax = jax.random.normal(key2, (batch_size, seq_len, num_heads, head_dim))

    # Create position embeddings cache
    cache = create_sinusoidal_positions(max_position_embeddings, head_dim, base)

    # Apply RoPE using our JAX implementation
    q_rotated_jax = apply_rotary_pos_emb(q_jax, cache[:seq_len])
    k_rotated_jax = apply_rotary_pos_emb(k_jax, cache[:seq_len])

    # Convert to PyTorch for reference implementation
    q_torch = torch.from_numpy(np.array(q_jax))
    k_torch = torch.from_numpy(np.array(k_jax))

    # Apply RoPE using PyTorch implementation
    rope_torch = RotaryPositionalEmbeddingsTorch(
        dim=head_dim, max_seq_len=max_position_embeddings, base=base
    )
    q_rotated_torch = rope_torch(q_torch)
    k_rotated_torch = rope_torch(k_torch)

    # Convert results back to numpy for comparison
    q_rotated_torch = q_rotated_torch.numpy()
    k_rotated_torch = k_rotated_torch.numpy()
    q_rotated_jax = np.array(q_rotated_jax)
    k_rotated_jax = np.array(k_rotated_jax)

    # Compare results
    np.testing.assert_allclose(q_rotated_jax, q_rotated_torch, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(k_rotated_jax, k_rotated_torch, rtol=1e-5, atol=1e-5)


def test_attention_global():
    """Test global attention mechanism."""
    batch_size = 2
    seq_len = 16
    hidden_size = 64
    num_heads = 4

    # Create random input
    key = jax.random.PRNGKey(0)
    hidden_states = jax.random.normal(key, (batch_size, seq_len, hidden_size))

    # Initialize attention
    attention = ModernBertAttention(
        rngs=nnx.Rngs(0),
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        attention_dropout=0.1,
    )

    # Test forward pass
    output = attention(hidden_states, deterministic=True)[0]
    assert output.shape == (batch_size, seq_len, hidden_size)

    # Test dropout is applied in training
    output_train = attention(hidden_states, deterministic=False)[0]
    assert not jnp.array_equal(output, output_train)

    # Test attention mask
    attention_mask = jnp.zeros((batch_size, 1, seq_len, seq_len))
    attention_mask = attention_mask.at[:, :, :, seq_len // 2 :].set(-10000.0)
    output_masked = attention(hidden_states, attention_mask=attention_mask)[0]

    # Check that masked positions have less influence
    influence_unmasked = jnp.mean(jnp.abs(output[:, :, :] - output_masked[:, :, :]))
    assert influence_unmasked > 0.0


def test_attention_local():
    """Test local attention with sliding window."""
    batch_size = 2
    seq_len = 16
    hidden_size = 64
    num_heads = 4
    window_size = (4, 4)  # 4 tokens left and right

    # Create random input
    key = jax.random.PRNGKey(0)
    hidden_states = jax.random.normal(key, (batch_size, seq_len, hidden_size))

    # Initialize attention with local window
    attention = ModernBertAttention(
        rngs=nnx.Rngs(0),
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        attention_dropout=0.1,
        local_attention=window_size,
    )

    # Test forward pass
    output = attention(hidden_states, deterministic=True)[0]
    assert output.shape == (batch_size, seq_len, hidden_size)

    # Create attention mask that allows full attention
    full_mask = jnp.zeros((1, 1, seq_len, seq_len))
    output_full = attention(hidden_states, sliding_window_mask=full_mask)[0]

    # The outputs should be different due to windowing
    assert not jnp.allclose(output, output_full, atol=1e-5)

    # Test with custom position IDs
    position_ids = jnp.array([[3, 7, 1, 4] + [0] * (seq_len - 4)] * batch_size)
    output_pos = attention(hidden_states, position_ids=position_ids)[0]
    assert not jnp.array_equal(output, output_pos)


def test_attention_output_attentions():
    """Test attention with output_attentions=True."""
    batch_size = 2
    seq_len = 16
    hidden_size = 64
    num_heads = 4

    # Create random input
    key = jax.random.PRNGKey(0)
    hidden_states = jax.random.normal(key, (batch_size, seq_len, hidden_size))

    # Initialize attention
    attention = ModernBertAttention(
        rngs=nnx.Rngs(0),
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
    )

    # Test with output_attentions=True
    output, attention_probs = attention(hidden_states, output_attentions=True)
    assert output.shape == (batch_size, seq_len, hidden_size)
    assert attention_probs.shape == (batch_size, num_heads, seq_len, seq_len)

    # Verify attention probabilities sum to 1
    assert jnp.allclose(jnp.sum(attention_probs, axis=-1), 1.0, atol=1e-6)


def test_create_sliding_window_mask():
    """Test creation of sliding window attention mask."""
    seq_len = 8
    window_size = (2, 2)  # 2 tokens left and right

    # Create mask
    mask = create_sliding_window_mask(seq_len, window_size)
    assert mask.shape == (1, 1, seq_len, seq_len)

    # Check mask values
    for i in range(seq_len):
        for j in range(seq_len):
            if abs(i - j) <= window_size[0]:
                assert mask[0, 0, i, j] == 0.0
            else:
                assert mask[0, 0, i, j] == -jnp.inf
