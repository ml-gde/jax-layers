"""ModernBERT implementation in JAX using Flax NNX.

This module implements the ModernBERT architecture as described in the paper
"Bringing BERT into Modernity" by Answer.AI. The implementation includes modern
improvements such as RoPE, SwiGLU, and global/local attention mechanisms.
"""


import flax.nnx as nnx
import jax.numpy as jnp


def create_sinusoidal_positions(
    max_length: int, dim: int, base: float = 10000.0
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Create sinusoidal position embeddings.

    Args:
        max_length: Maximum sequence length
        dim: Dimension of the embeddings (must be even)
        base: Base for the sinusoidal functions

    Returns:
        Tuple of (cos, sin) arrays of shape (max_length, dim//2, 2)
    """
    # Create position indices [0, 1, ..., max_length-1]
    positions = jnp.arange(max_length, dtype=jnp.float32)

    # Create dimension indices [0, 2, ..., dim-2] for half the dimensions
    dim_indices = jnp.arange(0, dim, 2, dtype=jnp.float32)

    # Calculate theta: base^(-2i/dim)
    theta = 1.0 / (base ** (dim_indices / dim))

    # Calculate angles: pos * theta using einsum
    # Shape: (max_length, dim//2)
    angles = jnp.einsum("i,j->ij", positions, theta)

    # Return stacked (cos, sin) tuple of shape (max_length, dim//2, 2)
    return jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)


def apply_rotary_pos_emb(
    x: jnp.ndarray, cache: jnp.ndarray, positions: jnp.ndarray | None = None
) -> jnp.ndarray:
    """Apply rotary position embeddings to input tensor.

    Args:
        x: Input tensor of shape (batch_size, seq_len, num_heads, head_dim)
        cache: Cache tensor of shape (max_seq_len, head_dim//2, 2) containing (cos, sin)
        positions: Optional position indices of shape (batch_size, seq_len)

    Returns:
        Tensor with rotary embeddings applied
    """
    # Get sequence length from input
    seq_len = x.shape[1]

    # Get cache values based on positions
    if positions is not None:
        rope_cache = cache[positions]  # [batch, seq, dim//2, 2]
    else:
        rope_cache = cache[:seq_len]  # [seq, dim//2, 2]
        # Add batch dimension for broadcasting
        rope_cache = jnp.expand_dims(rope_cache, 0)  # [1, seq, dim//2, 2]

    # Add head dimension for broadcasting
    rope_cache = jnp.expand_dims(rope_cache, 2)  # [batch/1, seq, 1, dim//2, 2]

    # Reshape input for rotation
    x_reshaped = x.reshape(*x.shape[:-1], -1, 2)  # [batch, seq, heads, dim//2, 2]

    # Apply rotation using complex multiplication
    x_out = jnp.stack(
        [
            x_reshaped[..., 0] * rope_cache[..., 0] - x_reshaped[..., 1] * rope_cache[..., 1],
            x_reshaped[..., 1] * rope_cache[..., 0] + x_reshaped[..., 0] * rope_cache[..., 1],
        ],
        axis=-1,
    )

    # Reshape back to original shape
    return x_out.reshape(x.shape)


class RoPEPositionalEmbedding(nnx.Module):
    """Rotary Position Embedding (RoPE) implementation.

    Based on https://arxiv.org/abs/2104.09864
    """

    def __init__(
        self, rngs: nnx.Rngs, dim: int, max_position_embeddings: int = 4096, base: float = 10000.0
    ):
        """Initialize RoPE module.

        Args:
            rngs: PRNG key collection
            dim: Dimension of the embeddings (must be even)
            max_position_embeddings: Maximum sequence length to cache
            base: Base for the sinusoidal functions
        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Create and store the cos/sin cache
        self.cache = create_sinusoidal_positions(self.max_position_embeddings, self.dim, self.base)

    def __call__(
        self,
        x: jnp.ndarray,
        positions: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Apply rotary position embeddings to input tensor.

        Args:
            x: Input tensor of shape [batch_size, seq_len, num_heads, head_dim]
            positions: Optional position ids of shape [batch_size, seq_len]

        Returns:
            Tensor with position embeddings applied
        """
        return apply_rotary_pos_emb(x, self.cache, positions)


class SwiGLU(nnx.Module):
    """SwiGLU activation function module."""

    hidden_dim: int

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply SwiGLU activation.

        Args:
            x: Input tensor

        Returns:
            Activated tensor
        """
        # TODO: Implement SwiGLU
        return x


class ModernBERTAttention(nnx.Module):
    """ModernBERT attention module with support for global and local attention."""

    num_heads: int
    head_dim: int
    dropout_rate: float = 0.0

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray | None = None,
        deterministic: bool = True,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Apply attention mechanism.

        Args:
            hidden_states: Input tensor
            attention_mask: Optional attention mask
            deterministic: Whether to apply dropout

        Returns:
            Output tensor and attention weights
        """
        # TODO: Implement ModernBERT attention
        return hidden_states, {}


class ModernBERTLayer(nnx.Module):
    """ModernBERT transformer layer."""

    num_heads: int
    head_dim: int
    intermediate_size: int
    dropout_rate: float = 0.0

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray | None = None,
        deterministic: bool = True,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Apply transformer layer.

        Args:
            hidden_states: Input tensor
            attention_mask: Optional attention mask
            deterministic: Whether to apply dropout

        Returns:
            Output tensor and attention weights
        """
        # TODO: Implement ModernBERT layer
        return hidden_states, {}


class ModernBERTEncoder(nnx.Module):
    """ModernBERT encoder consisting of multiple transformer layers."""

    num_layers: int
    num_heads: int
    head_dim: int
    intermediate_size: int
    dropout_rate: float = 0.0

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray | None = None,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> tuple[jnp.ndarray, ...]:
        """Apply transformer encoder.

        Args:
            hidden_states: Input tensor
            attention_mask: Optional attention mask
            deterministic: Whether to apply dropout
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states

        Returns:
            Tuple of:
                - Output tensor
                - All hidden states (optional)
                - All attention weights (optional)
        """
        # TODO: Implement ModernBERT encoder
        return (hidden_states,)


class ModernBERTMLMHead(nnx.Module):
    """ModernBERT masked language modeling head."""

    vocab_size: int
    hidden_size: int

    def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        """Apply MLM head.

        Args:
            hidden_states: Input tensor

        Returns:
            Logits for vocabulary
        """
        # TODO: Implement MLM head
        return hidden_states


class ModernBERTForMaskedLM(nnx.Module):
    """ModernBERT model with masked language modeling head."""

    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    head_dim: int
    intermediate_size: int
    max_position_embeddings: int = 2048
    dropout_rate: float = 0.0

    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: jnp.ndarray | None = None,
        token_type_ids: jnp.ndarray | None = None,
        position_ids: jnp.ndarray | None = None,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> dict[str, jnp.ndarray]:
        """Apply ModernBERT model.

        Args:
            input_ids: Input token ids
            attention_mask: Optional attention mask
            token_type_ids: Optional token type ids
            position_ids: Optional position ids
            deterministic: Whether to apply dropout
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states

        Returns:
            Dictionary containing:
                - logits
                - hidden states (optional)
                - attentions (optional)
        """
        # TODO: Implement full ModernBERT model
        return {"logits": jnp.zeros((1,))}
