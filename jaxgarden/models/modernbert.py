"""ModernBERT implementation in JAX using Flax NNX.

This module implements the ModernBERT architecture as described in the paper
"Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder
for Fast, Memory Efficient, and Long Context Finetuning and Inference" by Answer.AI.
The implementation includes modern improvements such as RoPE and global/local attention mechanisms.

See: https://arxiv.org/abs/2412.13663
"""

from dataclasses import dataclass

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from jaxgarden.models.base import BaseConfig, BaseModel


class Identity(nnx.Module):
    """Identity layer that simply returns its input."""

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x


def create_sinusoidal_positions(max_length: int, dim: int, base: float = 10000.0) -> jnp.ndarray:
    """Create sinusoidal position embeddings.

    Args:
        max_length: Maximum sequence length
        dim: Dimension of the embeddings (must be even)
        base: Base for the sinusoidal functions

    Returns:
        Array of shape (max_length, dim//2, 2) containing (cos, sin) values
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
        # Take exactly seq_len positions and repeat if needed
        positions = jnp.arange(seq_len) % cache.shape[0]
        rope_cache = cache[positions]  # [seq, dim//2, 2]
        # Add batch dimension for broadcasting
        rope_cache = jnp.expand_dims(rope_cache, 0)  # [1, seq, dim//2, 2]

    # Reshape input for rotation
    x_reshaped = x.reshape(*x.shape[:-1], -1, 2)  # [batch, seq, heads, dim//2, 2]

    # Add head dimension to rope_cache for broadcasting
    rope_cache = jnp.expand_dims(rope_cache, 2)  # [batch/1, seq, 1, dim//2, 2]

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


def create_sliding_window_mask(
    seq_len: int, window_size: tuple[int, int], dtype: jnp.dtype = jnp.float32
) -> jnp.ndarray:
    """Create a sliding window attention mask.

    Args:
        seq_len: Length of the sequence
        window_size: Tuple of (left, right) window sizes
        dtype: Data type of the mask

    Returns:
        Mask tensor of shape [1, 1, seq_len, seq_len] with -inf outside window
    """
    # Create position indices
    positions = jnp.arange(seq_len)

    # Create relative position matrix [seq_len, seq_len]
    relative_positions = positions[:, None] - positions[None, :]

    # Create window mask
    left_window, right_window = window_size
    mask = (relative_positions >= -left_window) & (relative_positions <= right_window)

    # Convert to float and replace False with -inf
    mask = mask.astype(dtype)
    mask = jnp.where(mask, 0.0, -jnp.inf)

    # Add batch and head dimensions [1, 1, seq_len, seq_len]
    return mask[None, None, :, :]


class ModernBertAttention(nnx.Module):
    """Multi-headed self attention implementation.

    This implements the standard attention mechanism with RoPE (Rotary Position Embeddings).
    Supports both global attention and sliding window attention.
    """

    def __init__(
        self,
        rngs: nnx.Rngs,
        hidden_size: int,
        num_attention_heads: int,
        attention_dropout: float = 0.0,
        attention_bias: bool = True,
        global_rope_theta: float = 10000.0,
        max_position_embeddings: int = 4096,
        local_attention: tuple[int, int] = (-1, -1),  # (-1, -1) means global attention
        local_rope_theta: float | None = None,
        layer_id: int | None = None,
        global_attn_every_n_layers: int = 4,
    ):
        """Initialize attention module.

        Args:
            rngs: PRNG key collection
            hidden_size: Size of hidden states
            num_attention_heads: Number of attention heads
            attention_dropout: Dropout probability for attention weights
            attention_bias: Whether to use bias in linear layers
            global_rope_theta: Base for global RoPE
            max_position_embeddings: Maximum sequence length
            local_attention: Tuple of (left, right) window sizes for local attention
            local_rope_theta: Base for local RoPE (optional)
            layer_id: Layer index for determining attention type
            global_attn_every_n_layers: Apply global attention every N layers
        """
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number "
                f"of attention heads ({num_attention_heads})"
            )

        # Store configuration
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.all_head_size = self.head_dim * num_attention_heads
        self.attention_dropout = attention_dropout

        # Compute local attention window if needed
        if layer_id is not None and layer_id % global_attn_every_n_layers != 0:
            local_attention = (local_attention[0] // 2, local_attention[1] // 2)

        # Select RoPE parameters based on attention type
        rope_theta = global_rope_theta
        max_pos = max_position_embeddings
        if local_attention != (-1, -1):
            if local_rope_theta is not None:
                rope_theta = local_rope_theta
            max_pos = local_attention[0] + local_attention[1]

        # Initialize layers
        self.Wqkv = nnx.Linear(
            rngs=rngs,
            in_features=hidden_size,
            out_features=3 * self.all_head_size,
            use_bias=attention_bias,
        )

        self.rotary_emb = RoPEPositionalEmbedding(
            rngs=rngs,
            dim=self.head_dim,
            max_position_embeddings=max_pos,
            base=rope_theta,
        )

        self.Wo = nnx.Linear(
            rngs=rngs,
            in_features=hidden_size,
            out_features=hidden_size,
            use_bias=attention_bias,
        )

        # Store configuration
        self.local_attention = local_attention
        self._rngs = rngs  # Store rngs for dropout

    def __call__(
        self,
        hidden_states: jax.Array,
        attention_mask: jax.Array | None = None,
        sliding_window_mask: jax.Array | None = None,
        position_ids: jax.Array | None = None,
        deterministic: bool = True,
        output_attentions: bool = False,
    ) -> tuple[jax.Array] | tuple[jax.Array, jax.Array]:
        """Apply attention module.

        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask
            sliding_window_mask: Optional sliding window mask for local attention
            position_ids: Optional position ids for RoPE
            deterministic: Whether to apply dropout
            output_attentions: Whether to return attention probabilities

        Returns:
            Tuple of:
                - Output tensor of shape [batch_size, seq_len, hidden_size]
                - Attention probabilities (optional) of shape [b_size, n_heads, seq_len, seq_len]
        """
        # Project to Q, K, V
        qkv = self.Wqkv(hidden_states)  # [batch_size, seq_len, 3 * hidden_size]

        # Reshape to [batch_size, seq_len, 3, num_heads, head_dim]
        batch_size = hidden_states.shape[0]
        qkv = jnp.reshape(qkv, (batch_size, -1, 3, self.num_attention_heads, self.head_dim))

        # Split into query, key, value and apply RoPE
        # First transpose to get [batch_size, num_heads, seq_len, head_dim]
        qkv = jnp.transpose(qkv, (0, 3, 1, 2, 4))  # [batch_size, num_heads, seq_len, 3, head_dim]
        query, key, value = jnp.split(
            qkv, 3, axis=3
        )  # Each: [batch_size, num_heads, seq_len, 1, head_dim]
        query = jnp.squeeze(query, axis=3)  # [batch_size, num_heads, seq_len, head_dim]
        key = jnp.squeeze(key, axis=3)
        value = jnp.squeeze(value, axis=3)

        # Apply rotary embeddings
        # First transpose to [batch_size, seq_len, num_heads, head_dim] for RoPE
        query = jnp.transpose(query, (0, 2, 1, 3))
        key = jnp.transpose(key, (0, 2, 1, 3))
        query = apply_rotary_pos_emb(query, self.rotary_emb.cache, position_ids)
        key = apply_rotary_pos_emb(key, self.rotary_emb.cache, position_ids)
        # Transpose back to [batch_size, num_heads, seq_len, head_dim]
        query = jnp.transpose(query, (0, 2, 1, 3))
        key = jnp.transpose(key, (0, 2, 1, 3))

        # Compute attention scores
        scale = jnp.sqrt(self.head_dim)
        attention_scores = jnp.matmul(query, jnp.swapaxes(key, -2, -1)) / scale

        # Apply attention masks
        if self.local_attention != (-1, -1):
            # Create sliding window mask if not provided
            if sliding_window_mask is None:
                sliding_window_mask = create_sliding_window_mask(
                    query.shape[2], self.local_attention, query.dtype
                )
            # Ensure the mask is properly broadcasted to [batch_size, num_heads, seq_len, seq_len]
            if len(sliding_window_mask.shape) == 4 and sliding_window_mask.shape[:2] == (1, 1):
                sliding_window_mask = jnp.broadcast_to(
                    sliding_window_mask,
                    (batch_size, self.num_attention_heads, query.shape[2], query.shape[2]),
                )
            # Apply sliding window mask first
            attention_scores = attention_scores + sliding_window_mask

        # Apply additional attention mask if provided
        if attention_mask is not None and attention_mask is not sliding_window_mask:
            # Ensure proper broadcasting for attention mask
            if len(attention_mask.shape) == 4 and attention_mask.shape[:2] == (1, 1):
                attention_mask = jnp.broadcast_to(
                    attention_mask,
                    (batch_size, self.num_attention_heads, query.shape[2], query.shape[2]),
                )
            attention_scores = attention_scores + attention_mask

        # Apply softmax and dropout
        attention_probs = jax.nn.softmax(attention_scores, axis=-1)

        # Explicitly zero out probabilities where sliding window mask was -inf
        if self.local_attention != (-1, -1):
            # Create a binary mask from the sliding window mask
            binary_mask = jnp.where(sliding_window_mask == -jnp.inf, 0.0, 1.0)
            attention_probs = attention_probs * binary_mask

        if not deterministic and self.attention_dropout > 0:
            attention_probs = nnx.Dropout(
                self.attention_dropout,
                rngs=self._rngs,
                deterministic=deterministic,
            )(attention_probs)

        # Compute attention output
        attention_output = jnp.matmul(attention_probs, value)

        # Reshape and apply output projection
        attention_output = jnp.transpose(attention_output, (0, 2, 1, 3))
        attention_output = jnp.reshape(attention_output, (batch_size, -1, self.hidden_size))
        attention_output = self.Wo(attention_output)

        # Apply output dropout
        if not deterministic and self.attention_dropout > 0:
            attention_output = nnx.Dropout(
                self.attention_dropout,
                rngs=self._rngs,
                deterministic=deterministic,
            )(attention_output)

        if output_attentions:
            return (attention_output, attention_probs)
        return (attention_output,)


class ModernBertLayer(nnx.Module):
    """ModernBERT transformer layer with pre-LayerNorm architecture.

    This implements a transformer layer with:
    1. Pre-LayerNorm for attention and MLP
    2. Residual connections
    3. Optional identity for first layer's attention norm
    """

    def __init__(
        self,
        rngs: nnx.Rngs,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        layer_id: int | None = None,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        attention_bias: bool = True,
        norm_eps: float = 1e-12,
        norm_bias: bool = True,
        global_rope_theta: float = 10000.0,
        max_position_embeddings: int = 4096,
        local_attention: tuple[int, int] = (-1, -1),
        local_rope_theta: float | None = None,
        global_attn_every_n_layers: int = 4,
    ):
        """Initialize transformer layer.

        Args:
            rngs: PRNG key collection
            hidden_size: Size of hidden states
            num_attention_heads: Number of attention heads
            intermediate_size: Size of MLP intermediate layer
            layer_id: Layer index (first layer uses identity for attn norm)
            attention_dropout: Dropout probability for attention
            hidden_dropout: Dropout probability for hidden states
            attention_bias: Whether to use bias in attention
            norm_eps: Epsilon for layer normalization
            norm_bias: Whether to use bias in layer normalization
            global_rope_theta: Base for global RoPE
            max_position_embeddings: Maximum sequence length
            local_attention: Tuple of (left, right) window sizes
            local_rope_theta: Base for local RoPE (optional)
            global_attn_every_n_layers: Apply global attention every N layers
        """
        super().__init__()

        # Initialize attention normalization
        self.attn_norm = (
            Identity()
            if layer_id == 0
            else nnx.LayerNorm(
                rngs=rngs,
                num_features=hidden_size,
                epsilon=norm_eps,
                use_bias=norm_bias,
                reduction_axes=(-1,),
                feature_axes=(-1,),
            )
        )  # type: ignore

        # Initialize attention
        self.attn = ModernBertAttention(
            rngs=rngs,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            attention_bias=attention_bias,
            global_rope_theta=global_rope_theta,
            max_position_embeddings=max_position_embeddings,
            local_attention=local_attention,
            local_rope_theta=local_rope_theta,
            layer_id=layer_id,
            global_attn_every_n_layers=global_attn_every_n_layers,
        )

        # Initialize MLP normalization
        self.mlp_norm = nnx.LayerNorm(
            rngs=rngs,
            num_features=hidden_size,
            epsilon=norm_eps,
            use_bias=norm_bias,
            reduction_axes=(-1,),
            feature_axes=(-1,),
        )

        # Initialize MLP
        self.mlp = ModernBertMLP(
            rngs=rngs,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            mlp_dropout=hidden_dropout,
        )

    def __call__(
        self,
        hidden_states: jax.Array,
        attention_mask: jax.Array | None = None,
        sliding_window_mask: jax.Array | None = None,
        position_ids: jax.Array | None = None,
        deterministic: bool = True,
        output_attentions: bool = False,
    ) -> tuple[jax.Array] | tuple[jax.Array, jax.Array]:
        """Apply transformer layer.

        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask
            sliding_window_mask: Optional sliding window mask
            position_ids: Optional position ids for RoPE
            deterministic: Whether to apply dropout
            output_attentions: Whether to return attention probabilities

        Returns:
            Tuple of:
                - Output tensor of shape [batch_size, seq_len, hidden_size]
                - Attention probabilities (optional) of shape [b_size, n_heads, seq_len, seq_len]
        """
        # Apply attention with pre-norm and residual
        attn_outputs = self.attn(
            self.attn_norm(hidden_states),
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            output_attentions=output_attentions,
        )
        attention_output = attn_outputs[0]
        attention_weights = attn_outputs[1] if output_attentions else None  # type: ignore

        # Residual connection
        hidden_states = hidden_states + attention_output

        # Apply MLP with pre-norm and residual
        mlp_output = self.mlp(self.mlp_norm(hidden_states), deterministic=deterministic)
        hidden_states = hidden_states + mlp_output

        if output_attentions:
            return (hidden_states, attention_weights)  # type: ignore
        return (hidden_states,)


class ModernBERTEncoder(nnx.Module):
    """ModernBERT encoder consisting of multiple transformer layers."""

    def __init__(
        self,
        rngs: nnx.Rngs,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        num_hidden_layers: int,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        attention_bias: bool = True,
        norm_eps: float = 1e-12,
        norm_bias: bool = True,
        global_rope_theta: float = 10000.0,
        max_position_embeddings: int = 4096,
        local_attention: tuple[int, int] = (-1, -1),
        local_rope_theta: float | None = None,
        global_attn_every_n_layers: int = 4,
    ):
        """Initialize encoder.

        Args:
            rngs: PRNG key collection
            hidden_size: Size of hidden states
            num_attention_heads: Number of attention heads
            intermediate_size: Size of MLP intermediate layer
            num_hidden_layers: Number of transformer layers
            attention_dropout: Dropout probability for attention
            hidden_dropout: Dropout probability for hidden states
            attention_bias: Whether to use bias in attention
            norm_eps: Epsilon for layer normalization
            norm_bias: Whether to use bias in layer normalization
            global_rope_theta: Base for global RoPE
            max_position_embeddings: Maximum sequence length
            local_attention: Tuple of (left, right) window sizes
            local_rope_theta: Base for local RoPE (optional)
            global_attn_every_n_layers: Apply global attention every N layers
        """
        super().__init__()

        # Initialize transformer layers
        self.layers = [
            ModernBertLayer(
                rngs=rngs,
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                layer_id=layer_id,
                attention_dropout=attention_dropout,
                hidden_dropout=hidden_dropout,
                attention_bias=attention_bias,
                norm_eps=norm_eps,
                norm_bias=norm_bias,
                global_rope_theta=global_rope_theta,
                max_position_embeddings=max_position_embeddings,
                local_attention=local_attention,
                local_rope_theta=local_rope_theta,
                global_attn_every_n_layers=global_attn_every_n_layers,
            )
            for layer_id in range(num_hidden_layers)
        ]

        # Initialize final layer normalization
        self.final_norm = nnx.LayerNorm(
            rngs=rngs,
            num_features=hidden_size,
            epsilon=norm_eps,
            use_bias=norm_bias,
            reduction_axes=(-1,),
            feature_axes=(-1,),
        )

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray | None = None,
        sliding_window_mask: jnp.ndarray | None = None,
        position_ids: jnp.ndarray | None = None,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> (
        tuple[jnp.ndarray]
        | tuple[jnp.ndarray, list[jnp.ndarray]]
        | tuple[jnp.ndarray, list[jnp.ndarray], list[jnp.ndarray]]
    ):
        """Apply transformer encoder.

        Args:
            hidden_states: Input tensor
            attention_mask: Optional attention mask
            sliding_window_mask: Optional sliding window mask
            position_ids: Optional position ids
            deterministic: Whether to apply dropout
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states

        Returns:
            Tuple of:
                - Output tensor
                - All hidden states (optional, if output_hidden_states=True)
                - All attention weights (optional, if output_attentions=True)
        """
        all_hidden_states: tuple | None = () if output_hidden_states else None  # type: ignore
        all_self_attentions: tuple | None = () if output_attentions else None

        # Process through each layer
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)  # type: ignore #noqa: RUF005

            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                sliding_window_mask=sliding_window_mask,
                position_ids=position_ids,
                deterministic=deterministic,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]
            if output_attentions and len(layer_outputs) > 1 and all_self_attentions is not None:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)  #  noqa: RUF005

        # Add final hidden state if requested
        if output_hidden_states and all_hidden_states is not None:
            all_hidden_states = all_hidden_states + (hidden_states,)  # noqa: RUF005

        # Apply final layer normalization
        hidden_states = self.final_norm(hidden_states)

        if not output_hidden_states and not output_attentions:
            return (hidden_states,)
        elif output_hidden_states and not output_attentions:
            return (hidden_states, all_hidden_states)  # type: ignore
        elif output_attentions and not output_hidden_states:
            return (hidden_states, all_self_attentions)  # type: ignore
        else:  # both output_hidden_states and output_attentions
            return (hidden_states, all_hidden_states, all_self_attentions)  # type: ignore


class ModernBERTMLMHead(nnx.Module):
    """ModernBERT masked language modeling head."""

    def __init__(
        self,
        rngs: nnx.Rngs,
        hidden_size: int,
        vocab_size: int,
        norm_eps: float = 1e-12,
        norm_bias: bool = True,
    ):
        """Initialize MLM head.

        Args:
            rngs: PRNG key collection
            hidden_size: Size of hidden states
            vocab_size: Size of vocabulary
            norm_eps: Epsilon for layer normalization
            norm_bias: Whether to use bias in layer normalization
        """
        super().__init__()

        # Layer norm
        self.norm = nnx.LayerNorm(
            rngs=rngs,
            num_features=hidden_size,
            epsilon=norm_eps,
            use_bias=norm_bias,
            reduction_axes=(-1,),
            feature_axes=(-1,),
        )

        # Dense projection
        self.dense = nnx.Linear(
            rngs=rngs,
            in_features=hidden_size,
            out_features=hidden_size,
            use_bias=True,
        )

        # Output projection (tied with embeddings)
        self.decoder = nnx.Linear(
            rngs=rngs,
            in_features=hidden_size,
            out_features=vocab_size,
            use_bias=True,
        )

    def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        """Apply MLM head.

        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]

        Returns:
            Logits of shape [batch_size, seq_len, vocab_size]
        """
        hidden_states = self.norm(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = jax.nn.gelu(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


@dataclass
class ModernBERTConfig(BaseConfig):
    """
    Configuration for ModernBERT model.

    This configuration class extends BaseConfig and contains all the parameters
    required to initialize a ModernBERT model. It includes settings for model architecture,
    attention mechanisms, dropout rates, and other hyperparameters.

    Attributes:
        vocab_size: Size of vocabulary
        hidden_size: Size of hidden states
        num_hidden_layers: Number of transformer layers
        num_attention_heads: Number of attention heads
        intermediate_size: Size of MLP intermediate layer
        max_position_embeddings: Maximum sequence length
        attention_dropout: Dropout probability for attention
        hidden_dropout: Dropout probability for hidden states
        attention_bias: Whether to use bias in attention
        norm_eps: Epsilon for layer normalization
        norm_bias: Whether to use bias in layer normalization
        global_rope_theta: Base for global RoPE
        local_attention: Tuple of (left, right) window sizes
        local_rope_theta: Base for local RoPE (optional)
        global_attn_every_n_layers: Apply global attention every N layers
        pad_token_id: Token ID to use for padding
    """

    # Default ModernBERT configuration
    # https://huggingface.co/docs/transformers/main//model_doc/modernbert#transformers.ModernBertConfig
    vocab_size: int = 50368
    hidden_size: int = 768
    intermediate_size: int = 1152
    num_hidden_layers: int = 22
    num_attention_heads: int = 12
    max_position_embeddings: int = 8192
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    attention_bias: bool = False
    norm_eps: float = 1e-05
    norm_bias: bool = False
    global_rope_theta: float = 160000.0
    local_attention: tuple[int, int] = (-1, -1)
    local_rope_theta: float | None = None
    global_attn_every_n_layers: int = 3
    pad_token_id: int = 50283


class ModernBERTForMaskedLM(BaseModel):
    """ModernBERT model with masked language modeling head.

    This implements the ModernBERT architecture as described in the paper
    "Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient,
    and Long Context Finetuning and Inference" by Answer.AI.

    The implementation includes modern improvements such as:
    - Rotary Position Embeddings (RoPE)
    - Mixed global/local attention mechanism
    - Pre-LayerNorm architecture
    - Efficient parameter sharing
    """

    def __init__(
        self,
        config: ModernBERTConfig,
        *,
        dtype: jnp.dtype | None = None,
        param_dtype: jnp.dtype = jnp.float32,
        precision: jax.lax.Precision | str | None = None,
        rngs: nnx.Rngs,
    ):
        """Initialize ModernBERT model.

        Args:
            config: Configuration for ModernBERT model
            dtype: Data type in which computation is performed
            param_dtype: Data type in which params are stored
            precision: Numerical precision
            rngs: Random number generators for param initialization
        """
        super().__init__(
            config, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs
        )

        # Initialize embeddings
        self.embeddings = ModernBertEmbeddings(
            rngs=rngs,
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            pad_token_id=config.pad_token_id,
            norm_eps=config.norm_eps,
            norm_bias=config.norm_bias,
            embedding_dropout=config.hidden_dropout,
        )

        # Initialize encoder
        self.encoder = ModernBERTEncoder(
            rngs=rngs,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            attention_dropout=config.attention_dropout,
            hidden_dropout=config.hidden_dropout,
            attention_bias=config.attention_bias,
            norm_eps=config.norm_eps,
            norm_bias=config.norm_bias,
            global_rope_theta=config.global_rope_theta,
            max_position_embeddings=config.max_position_embeddings,
            local_attention=config.local_attention,
            local_rope_theta=config.local_rope_theta,
            global_attn_every_n_layers=config.global_attn_every_n_layers,
        )

        # Initialize MLM head
        self.mlm_head = ModernBERTMLMHead(
            rngs=rngs,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            norm_eps=config.norm_eps,
            norm_bias=config.norm_bias,
        )

    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: jnp.ndarray | None = None,
        sliding_window_mask: jnp.ndarray | None = None,
        position_ids: jnp.ndarray | None = None,
        inputs_embeds: jnp.ndarray | None = None,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> dict[str, jnp.ndarray]:
        """Apply ModernBERT model.

        Args:
            input_ids: Input token ids of shape [batch_size, seq_len]
            attention_mask: Optional attention mask
            sliding_window_mask: Optional sliding window mask
            position_ids: Optional position ids
            inputs_embeds: Optional pre-computed embeddings
            deterministic: Whether to apply dropout
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states

        Returns:
            Dictionary containing:
                - logits: Output logits of shape [batch_size, seq_len, vocab_size]
                - hidden_states: All hidden states (optional)
                - attentions: All attention weights (optional)
        """
        # Get embeddings
        hidden_states = self.embeddings(
            input_ids=input_ids,
            deterministic=deterministic,
            inputs_embeds=inputs_embeds,
        )

        # Apply encoder
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        # Get sequence output and optional states
        sequence_output = encoder_outputs[0]
        hidden_states = None  # type: ignore
        attentions = None

        if len(encoder_outputs) > 1:
            if output_hidden_states:
                hidden_states = encoder_outputs[1]  # type: ignore
            if output_attentions:
                attentions = encoder_outputs[1] if not output_hidden_states else encoder_outputs[2]  # type: ignore[misc]

        # Apply MLM head
        logits = self.mlm_head(sequence_output)

        # Build output dictionary
        outputs = {"logits": logits}
        if output_hidden_states:
            outputs["hidden_states"] = hidden_states  # type: ignore
        if output_attentions:
            outputs["attentions"] = attentions  # type: ignore

        return outputs


class ModernBertEmbeddings(nnx.Module):
    """Token embeddings with normalization and dropout.

    Similar to BERT embeddings but without position embeddings since we use RoPE.
    """

    def __init__(
        self,
        rngs: nnx.Rngs,
        vocab_size: int,
        hidden_size: int,
        pad_token_id: int = 0,
        norm_eps: float = 1e-12,
        norm_bias: bool = True,
        embedding_dropout: float = 0.0,
    ):
        """Initialize embeddings module.

        Args:
            rngs: PRNG key collection
            vocab_size: Size of the vocabulary
            hidden_size: Size of the embeddings
            pad_token_id: Token ID to use for padding
            norm_eps: Epsilon for layer normalization
            norm_bias: Whether to use bias in layer normalization
            embedding_dropout: Dropout probability for embeddings
        """
        super().__init__()

        # Create embeddings table with non-zero initialization
        self.token_embeddings = nnx.Embed(
            rngs=rngs,
            num_embeddings=vocab_size,
            features=hidden_size,
            embedding_init=nnx.initializers.normal(stddev=1.0),  # Increased stddev
        )

        # Layer norm with explicit feature axes
        self.norm = nnx.LayerNorm(
            rngs=rngs,
            num_features=hidden_size,
            epsilon=norm_eps,
            use_bias=norm_bias,
            use_scale=True,
            scale_init=nnx.initializers.ones,
            bias_init=nnx.initializers.zeros,
            reduction_axes=(-1,),  # Explicit tuple
            feature_axes=(-1,),  # Explicit tuple
        )

        self.dropout = embedding_dropout
        self.deterministic = True  # Will be overridden in __call__
        self._rngs = rngs  # Store rngs for dropout

    def __call__(
        self,
        input_ids: jnp.ndarray,
        deterministic: bool = True,
        inputs_embeds: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Apply embeddings module.

        Args:
            input_ids: Integer tokens of shape [batch_size, seq_len]
            deterministic: Whether to apply dropout
            inputs_embeds: Optional pre-computed embeddings

        Returns:
            Embedded tokens with shape [batch_size, seq_len, hidden_size]
        """
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.token_embeddings(input_ids)

            # Scale embeddings
            hidden_states = hidden_states * (hidden_states.shape[-1] ** 0.5)

        # Apply layer norm
        hidden_states = self.norm(hidden_states)

        # Apply dropout if in training
        if not deterministic and self.dropout > 0.0:
            hidden_states = nnx.Dropout(
                rngs=self._rngs,
                rate=self.dropout,
            )(hidden_states, deterministic=deterministic)

        return hidden_states


class ModernBertMLP(nnx.Module):
    """MLP with gated linear units.

    Replaces the traditional intermediate + output layers with a single gated MLP.
    """

    def __init__(
        self,
        rngs: nnx.Rngs,
        hidden_size: int,
        intermediate_size: int,
        mlp_bias: bool = True,
        mlp_dropout: float = 0.0,
    ):
        """Initialize MLP module.

        Args:
            rngs: PRNG key collection
            hidden_size: Size of input and output
            intermediate_size: Size of intermediate layer
            mlp_bias: Whether to use bias in linear layers
            mlp_dropout: Dropout probability
        """
        super().__init__()

        # Input projection (creates both input and gate values)
        self.Wi = nnx.Linear(
            rngs=rngs,
            in_features=hidden_size,
            out_features=intermediate_size * 2,
            use_bias=mlp_bias,
            kernel_init=nnx.initializers.normal(stddev=0.02),
        )

        # Output projection
        self.Wo = nnx.Linear(
            rngs=rngs,
            in_features=intermediate_size,
            out_features=hidden_size,
            use_bias=mlp_bias,
            kernel_init=nnx.initializers.normal(stddev=0.02),
        )

        self.dropout = mlp_dropout
        self.deterministic = True  # Will be overridden in __call__
        self._rngs = rngs  # Store rngs for dropout

    def __call__(self, hidden_states: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """Apply MLP module.

        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            deterministic: Whether to apply dropout

        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        # Project to intermediate size
        hidden_states = self.Wi(hidden_states)

        # Apply GELU activation
        hidden_states = jax.nn.gelu(hidden_states)

        # Split into input and gate by taking every other feature
        hidden_states = self.Wo(hidden_states[..., ::2] * hidden_states[..., 1::2])

        # Apply dropout if in training
        if not deterministic and self.dropout > 0.0:
            hidden_states = nnx.Dropout(
                rngs=self._rngs,
                rate=self.dropout,
            )(hidden_states, deterministic=deterministic)

        return hidden_states
