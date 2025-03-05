"""Tests for ModernBERT components."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from jax_layers.models.modernbert import (
    Identity,
    ModernBertAttention,
    ModernBertEmbeddings,
    ModernBertLayer,
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


def apply_rotary_pos_emb_torch(
    query: torch.Tensor, key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference implementation of RoPE."""
    # cos/sin: [batch_size, seq_len, head_dim//2]
    # query/key: [batch_size, num_heads, seq_len, head_dim]
    cos = cos.unsqueeze(1)  # [batch_size, 1, seq_len, head_dim//2]
    sin = sin.unsqueeze(1)  # [batch_size, 1, seq_len, head_dim//2]

    # Reshape query and key to split last dimension
    query_split = query.unflatten(-1, (query.shape[-1] // 2, 2))  # [batch, heads, seq, dim//2, 2]
    key_split = key.unflatten(-1, (key.shape[-1] // 2, 2))

    # Apply rotation using complex multiplication
    query_rot = torch.stack(
        [
            query_split[..., 0] * cos - query_split[..., 1] * sin,
            query_split[..., 1] * cos + query_split[..., 0] * sin,
        ],
        dim=-1,
    )
    key_rot = torch.stack(
        [
            key_split[..., 0] * cos - key_split[..., 1] * sin,
            key_split[..., 1] * cos + key_split[..., 0] * sin,
        ],
        dim=-1,
    )

    # Reshape back
    return query_rot.flatten(-2), key_rot.flatten(-2)


class TorchRotaryEmbedding(nn.Module):
    """PyTorch reference implementation of RoPE."""

    def __init__(self, dim: int, max_position_embeddings: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Create position embeddings
        positions = torch.arange(max_position_embeddings, dtype=torch.float32)
        dim_indices = torch.arange(0, dim, 2, dtype=torch.float32)
        theta = 1.0 / (base ** (dim_indices / dim))
        angles = torch.einsum("i,j->ij", positions, theta)
        self.register_buffer("cos", torch.cos(angles))  # [max_seq, dim//2]
        self.register_buffer("sin", torch.sin(angles))  # [max_seq, dim//2]

    def forward(
        self,
        qkv: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get cos and sin embeddings based on position IDs."""
        seq_len = qkv.shape[1]
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=qkv.device)
            position_ids = position_ids.unsqueeze(0).expand(qkv.shape[0], -1)  # type: ignore

        # Add type ignore since we know these are valid Tensors
        cos = F.embedding(position_ids, self.cos)  # type: ignore[arg-type]
        sin = F.embedding(position_ids, self.sin)  # type: ignore[arg-type]
        return cos, sin


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


def test_attention_numerical_correctness():
    """Test that our JAX attention implementation matches PyTorch reference."""
    # Test parameters
    batch_size = 2
    seq_len = 16
    hidden_size = 256
    num_heads = 4
    head_dim = hidden_size // num_heads
    max_position_embeddings = 32
    base = 10000.0

    # Create random input
    rng = np.random.RandomState(0)
    hidden_states = rng.normal(0, 1, (batch_size, seq_len, hidden_size)).astype(np.float32)

    # Create JAX attention module
    jax_attention = ModernBertAttention(
        rngs=nnx.Rngs(0),
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        max_position_embeddings=max_position_embeddings,
        attention_dropout=0.0,  # Disable dropout for comparison
    )

    # Create PyTorch attention components
    torch_qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
    torch_wo = nn.Linear(hidden_size, hidden_size, bias=True)
    torch_rope = TorchRotaryEmbedding(head_dim, max_position_embeddings, base)

    # Copy weights from JAX to PyTorch
    with torch.no_grad():
        # Copy attention weights
        torch_qkv.weight.copy_(torch.tensor(jax_attention.Wqkv.kernel.T))
        torch_qkv.bias.copy_(torch.tensor(jax_attention.Wqkv.bias))
        torch_wo.weight.copy_(torch.tensor(jax_attention.Wo.kernel.T))
        torch_wo.bias.copy_(torch.tensor(jax_attention.Wo.bias))

    # Convert inputs to tensors
    hidden_states_torch = torch.tensor(hidden_states)
    hidden_states_jax = jnp.array(hidden_states)

    # Test both global and local attention
    for local_attention in [(-1, -1), (8, 8)]:
        # Create attention masks
        if local_attention != (-1, -1):
            sliding_window_mask = create_sliding_window_mask(seq_len, local_attention)
            sliding_window_mask_torch = torch.tensor(np.array(sliding_window_mask))
            attention_mask = sliding_window_mask_torch
        else:
            sliding_window_mask = None
            attention_mask = None

        # Forward pass in PyTorch
        qkv = torch_qkv(hidden_states_torch)
        qkv = qkv.view(batch_size, -1, 3, num_heads, head_dim)
        cos, sin = torch_rope(qkv)
        query, key, value = qkv.transpose(3, 1).unbind(dim=2)
        query, key = apply_rotary_pos_emb_torch(query, key, cos, sin)

        # Compute attention scores
        scale = head_dim**-0.5
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * scale

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_probs, value)
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, -1, hidden_size)
        torch_output = torch_wo(attention_output)

        # Forward pass in JAX
        jax_attention.local_attention = local_attention
        jax_output = jax_attention(
            hidden_states_jax, sliding_window_mask=sliding_window_mask, deterministic=True
        )[0]

        # Compare outputs
        np.testing.assert_allclose(
            jax_output,
            torch_output.detach().numpy(),
            rtol=1e-4,
            atol=1e-4,
            err_msg=f"Outputs don't match for local_attention={local_attention}",
        )


def test_layer_shapes():
    """Test that ModernBertLayer preserves expected shapes."""
    batch_size = 2
    seq_len = 16
    hidden_size = 64
    num_heads = 4
    intermediate_size = 256

    # Create random input
    key = jax.random.PRNGKey(0)
    hidden_states = jax.random.normal(key, (batch_size, seq_len, hidden_size))

    # Initialize layer
    layer = ModernBertLayer(
        rngs=nnx.Rngs(0),
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        intermediate_size=intermediate_size,
    )

    # Test forward pass
    output = layer(hidden_states, deterministic=True)[0]
    assert output.shape == (batch_size, seq_len, hidden_size)

    # Test with attention outputs
    output, attention = layer(hidden_states, output_attentions=True)
    assert output.shape == (batch_size, seq_len, hidden_size)
    assert attention.shape == (batch_size, num_heads, seq_len, seq_len)


def test_layer_first_layer_identity():
    """Test that first layer uses identity for attention norm."""
    batch_size = 2
    seq_len = 16
    hidden_size = 64
    num_heads = 4
    intermediate_size = 256

    # Create random input
    key = jax.random.PRNGKey(0)
    hidden_states = jax.random.normal(key, (batch_size, seq_len, hidden_size))

    # Initialize first layer (layer_id=0)
    first_layer = ModernBertLayer(
        rngs=nnx.Rngs(0),
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        intermediate_size=intermediate_size,
        layer_id=0,
    )

    # Initialize non-first layer
    other_layer = ModernBertLayer(
        rngs=nnx.Rngs(1),
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        intermediate_size=intermediate_size,
        layer_id=1,
    )

    # First layer should use identity for attention norm
    assert isinstance(first_layer.attn_norm, Identity), "First layer should use Identity"
    # Other layer should use LayerNorm
    assert isinstance(other_layer.attn_norm, nnx.LayerNorm), "Other layers should use LayerNorm"

    # Verify first layer's identity behavior
    normed_states = first_layer.attn_norm(hidden_states)
    np.testing.assert_array_equal(hidden_states, normed_states)


def test_layer_residual_connections():
    """Test that residual connections are properly applied."""
    batch_size = 2
    seq_len = 16
    hidden_size = 64
    num_heads = 4
    intermediate_size = 256

    # Create random input with small values
    key = jax.random.PRNGKey(0)
    hidden_states = jax.random.normal(key, (batch_size, seq_len, hidden_size)) * 0.1

    # Initialize layer with zero weights to test residual
    def zeros_init(key, shape, dtype=jnp.float32):
        return jnp.zeros(shape, dtype)

    class ZeroLayer(ModernBertLayer):
        """Layer that initializes all weights to zero."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Override weight initializations with zeros
            self.attn.Wqkv = nnx.Linear(
                rngs=nnx.Rngs(0),
                in_features=self.attn.Wqkv.in_features,
                out_features=self.attn.Wqkv.out_features,
                kernel_init=lambda *_: jnp.zeros_like(self.attn.Wqkv.kernel),
                bias_init=lambda *_: jnp.zeros_like(self.attn.Wqkv.bias),
            )
            self.attn.Wo = nnx.Linear(
                rngs=nnx.Rngs(0),
                in_features=self.attn.Wo.in_features,
                out_features=self.attn.Wo.out_features,
                kernel_init=lambda *_: jnp.zeros_like(self.attn.Wo.kernel),
                bias_init=lambda *_: jnp.zeros_like(self.attn.Wo.bias),
            )
            self.mlp.Wi = nnx.Linear(
                rngs=nnx.Rngs(0),
                in_features=self.mlp.Wi.in_features,
                out_features=self.mlp.Wi.out_features,
                kernel_init=lambda *_: jnp.zeros_like(self.mlp.Wi.kernel),
                bias_init=lambda *_: jnp.zeros_like(self.mlp.Wi.bias),
            )
            self.mlp.Wo = nnx.Linear(
                rngs=nnx.Rngs(0),
                in_features=self.mlp.Wo.in_features,
                out_features=self.mlp.Wo.out_features,
                kernel_init=lambda *_: jnp.zeros_like(self.mlp.Wo.kernel),
                bias_init=lambda *_: jnp.zeros_like(self.mlp.Wo.bias),
            )

    layer = ZeroLayer(
        rngs=nnx.Rngs(0),
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        intermediate_size=intermediate_size,
    )

    # With zero weights and no residual, output would be zero
    output = layer(hidden_states, deterministic=True)[0]

    # Due to residual connections, output should equal input
    np.testing.assert_allclose(hidden_states, output, rtol=1e-5)


def test_layer_dropout():
    """Test that dropout is properly applied in training mode."""
    batch_size = 2
    seq_len = 16
    hidden_size = 64
    num_heads = 4
    intermediate_size = 256
    dropout_rate = 0.5

    # Create random input
    key = jax.random.PRNGKey(0)
    hidden_states = jax.random.normal(key, (batch_size, seq_len, hidden_size))

    # Initialize layer with high dropout
    layer = ModernBertLayer(
        rngs=nnx.Rngs(0),
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        intermediate_size=intermediate_size,
        attention_dropout=dropout_rate,
        hidden_dropout=dropout_rate,
    )

    # Test in eval mode (deterministic=True)
    eval_output = layer(hidden_states, deterministic=True)[0]

    # Test in training mode (deterministic=False)
    train_output = layer(hidden_states, deterministic=False)[0]

    # Outputs should be different due to dropout
    assert not jnp.allclose(eval_output, train_output)


def test_layer_sliding_window():
    """Test layer with sliding window attention."""
    batch_size = 2
    seq_len = 16
    hidden_size = 64
    num_heads = 4
    intermediate_size = 256
    window_size = (4, 4)

    # Create random input
    key = jax.random.PRNGKey(0)
    hidden_states = jax.random.normal(key, (batch_size, seq_len, hidden_size))

    # Initialize layer with local attention
    layer = ModernBertLayer(
        rngs=nnx.Rngs(0),
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        intermediate_size=intermediate_size,
        local_attention=window_size,
    )

    # Get attention outputs
    output, attention = layer(hidden_states, output_attentions=True)

    # Check that attention outside window is zero
    mask = create_sliding_window_mask(seq_len, window_size)
    masked_positions = mask[0, 0] == -jnp.inf
    assert jnp.allclose(attention[..., masked_positions], 0.0)


def test_layer_numerical_correctness():
    """Test layer against PyTorch reference implementation."""
    # Test parameters
    batch_size = 2
    seq_len = 16
    hidden_size = 256
    num_heads = 4
    intermediate_size = 1024
    layer_id = 1

    # Create random input with fixed seed for reproducibility
    key = jax.random.PRNGKey(0)
    hidden_states = jax.random.normal(key, (batch_size, seq_len, hidden_size))
    hidden_states = hidden_states.astype(jnp.float32)  # Ensure float32

    # Create JAX layer with fixed seed
    jax_layer = ModernBertLayer(
        rngs=nnx.Rngs(0),
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        intermediate_size=intermediate_size,
        layer_id=layer_id,
        attention_dropout=0.0,  # Disable dropout for comparison
        hidden_dropout=0.0,
        norm_eps=1e-12,  # Match PyTorch default
    )

    # Create PyTorch layer
    config = type(
        "ModernBertConfig",
        (),
        {
            "hidden_size": hidden_size,
            "num_attention_heads": num_heads,
            "intermediate_size": intermediate_size,
            "attention_dropout": 0.0,
            "hidden_dropout": 0.0,
            "attention_bias": True,
            "norm_eps": 1e-12,
            "norm_bias": True,
            "global_rope_theta": 10000.0,
            "max_position_embeddings": 4096,
            "local_attention": (-1, -1),
            "local_rope_theta": None,
            "global_attn_every_n_layers": 4,
            "reference_compile": False,
        },
    )()

    torch_layer = TorchModernBertLayer(config, layer_id=layer_id)

    # Copy weights from JAX to PyTorch with exact precision
    with torch.no_grad():
        # Copy attention weights
        torch_layer.attn.qkv.weight.copy_(
            torch.tensor(np.array(jax_layer.attn.Wqkv.kernel.T), dtype=torch.float32)
        )
        torch_layer.attn.qkv.bias.copy_(
            torch.tensor(np.array(jax_layer.attn.Wqkv.bias), dtype=torch.float32)
        )
        torch_layer.attn.o.weight.copy_(
            torch.tensor(np.array(jax_layer.attn.Wo.kernel.T), dtype=torch.float32)
        )
        torch_layer.attn.o.bias.copy_(
            torch.tensor(np.array(jax_layer.attn.Wo.bias), dtype=torch.float32)
        )

        # Copy MLP weights
        torch_layer.mlp.wi.weight.copy_(
            torch.tensor(np.array(jax_layer.mlp.Wi.kernel.T), dtype=torch.float32)
        )
        torch_layer.mlp.wi.bias.copy_(
            torch.tensor(np.array(jax_layer.mlp.Wi.bias), dtype=torch.float32)
        )
        torch_layer.mlp.wo.weight.copy_(
            torch.tensor(np.array(jax_layer.mlp.Wo.kernel.T), dtype=torch.float32)
        )
        torch_layer.mlp.wo.bias.copy_(
            torch.tensor(np.array(jax_layer.mlp.Wo.bias), dtype=torch.float32)
        )

        # Copy LayerNorm weights
        if not isinstance(torch_layer.attn_norm, nn.Identity):
            torch_layer.attn_norm.weight.copy_(
                torch.tensor(np.array(jax_layer.attn_norm.scale), dtype=torch.float32)
            )
            torch_layer.attn_norm.bias.copy_(
                torch.tensor(np.array(jax_layer.attn_norm.bias), dtype=torch.float32)
            )
        torch_layer.mlp_norm.weight.copy_(
            torch.tensor(np.array(jax_layer.mlp_norm.scale), dtype=torch.float32)
        )
        torch_layer.mlp_norm.bias.copy_(
            torch.tensor(np.array(jax_layer.mlp_norm.bias), dtype=torch.float32)
        )

    # Forward pass in both frameworks with same input
    hidden_states_torch = torch.tensor(np.array(hidden_states), dtype=torch.float32)
    hidden_states_jax = hidden_states

    # Run forward passes
    torch_output = torch_layer(hidden_states_torch)[0]
    jax_output = jax_layer(hidden_states_jax, deterministic=True)[0]

    # Convert outputs to numpy for comparison
    torch_output_np = torch_output.detach().cpu().numpy()
    jax_output_np = np.array(jax_output)

    # Compare outputs with increased tolerance
    np.testing.assert_allclose(
        jax_output_np,
        torch_output_np,
        rtol=1e-4,
        atol=1e-4,
        err_msg="Layer outputs don't match within tolerance",
    )


class TorchModernBertLayer(nn.Module):
    """PyTorch reference implementation of ModernBertLayer."""

    def __init__(self, config, layer_id: int = 0):
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.intermediate_size = config.intermediate_size

        # Pre-norm architecture
        # First layer uses identity for attention norm
        self.attn_norm = (
            nn.Identity
            if layer_id == 0
            else nn.LayerNorm(config.hidden_size, eps=config.norm_eps, elementwise_affine=True)
        )  # type: ignore

        # Attention
        self.attn = TorchModernBertAttention(config)
        self.mlp_norm = nn.LayerNorm(
            config.hidden_size, eps=config.norm_eps, elementwise_affine=True
        )
        self.mlp = TorchModernBertMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        sliding_window_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        # Pre-norm for attention
        attn_input = self.attn_norm(hidden_states)
        attn_outputs = self.attn(
            attn_input,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
        )
        attention_output = attn_outputs[0]
        outputs = attn_outputs[1:]

        # Residual connection
        hidden_states = hidden_states + attention_output

        # Pre-norm for MLP
        mlp_input = self.mlp_norm(hidden_states)
        mlp_output = self.mlp(mlp_input)

        # Residual connection
        hidden_states = hidden_states + mlp_output

        if output_attentions:
            return (hidden_states,) + outputs  # noqa:RUF005
        return (hidden_states,)


class TorchModernBertAttention(nn.Module):
    """PyTorch reference implementation of ModernBertAttention."""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.local_attention = config.local_attention

        # QKV projection
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=config.attention_bias)
        self.o = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)

        # RoPE
        self.rope = TorchRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.global_rope_theta,
        )

        # Dropout
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.hidden_dropout = nn.Dropout(config.hidden_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        sliding_window_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        batch_size, seq_len, _ = hidden_states.size()

        # Project to QKV
        qkv = self.qkv(hidden_states)
        qkv = qkv.view(batch_size, -1, 3, self.num_attention_heads, self.head_dim)

        # Apply RoPE to QK
        cos, sin = self.rope(qkv, position_ids)
        query, key, value = qkv.transpose(3, 1).unbind(dim=2)
        query, key = apply_rotary_pos_emb_torch(query, key, cos, sin)

        # Compute attention scores
        scale = self.head_dim**-0.5
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * scale

        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        if sliding_window_mask is not None:
            attention_scores = attention_scores + sliding_window_mask

        # Normalize scores to probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)

        # Apply attention dropout
        attention_probs = self.attn_dropout(attention_probs)

        # Compute attention output
        attention_output = torch.matmul(attention_probs, value)
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, -1, self.hidden_size)

        # Project to output size
        attention_output = self.o(attention_output)

        # Apply hidden dropout
        attention_output = self.hidden_dropout(attention_output)

        outputs = (attention_output,)
        if output_attentions:
            outputs = outputs + (attention_probs,)  # noqa: RUF005
        return outputs


class TorchModernBertMLP(nn.Module):
    """PyTorch reference implementation of ModernBertMLP."""

    def __init__(self, config):
        super().__init__()
        self.wi = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=True)
        self.wo = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.wi(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.wo(hidden_states[:, :, ::2] * hidden_states[:, :, 1::2])
        hidden_states = self.dropout(hidden_states)
        return hidden_states
