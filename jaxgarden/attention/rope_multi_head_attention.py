"""
JAX/Flax implementation of Multi-Head Attention with Rotary Positional Embedding (RoPE).

This code implements the RoPE technique within a standard Multi-Head Attention
framework. RoPE injects relative positional information by rotating pairs of
features in the Query and Key vectors based on their absolute position before
the attention calculation.

The method was introduced in the paper:
"RoFormer: Enhanced Transformer with Rotary Position Embedding"
by Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu.
arXiv:2104.09864v5 [cs.CL] (Submitted on 20 Apr 2021)
"""

import flax.linen as nn
import jax
import jax.numpy as jnp


def rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    """Rotates half the hidden dims of the input tensor."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    # Builds the rotated tensor by concatenating the negated second half
    # and the first half along the last dimension.
    return jnp.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb(x: jnp.ndarray, cos_emb: jnp.ndarray, sin_emb: jnp.ndarray) -> jnp.ndarray:
    """Applies Rotary Positional Embedding to the input tensor.

    Args:
      x: Input tensor, e.g., query or key (batch, seq_len, num_heads, head_dim)
      cos_emb: Cosine component of the positional embedding.
               Shape: (1, seq_len, 1, head_dim) or compatible via broadcasting.
      sin_emb: Sine component of the positional embedding.
               Shape: (1, seq_len, 1, head_dim) or compatible via broadcasting.
    Returns:
      Tensor with RoPE applied.
    """
    # Applying the rotation formula:
    # x_rotated = x * cos(theta) + rotate_half(x) * sin(theta)
    # Ensure shapes are broadcastable: cos_emb and sin_emb should have dimensions
    # for sequence length and features, matching the corresponding dimensions in x.
    # Typically, precomputed embeddings have shape (seq_len, head_dim)
    # or (1, seq_len, 1, head_dim) for easy broadcasting.
    return (x * cos_emb) + (rotate_half(x) * sin_emb)


def precompute_rotary_embeddings(
    seq_len: int, head_dim: int, base: float = 10000.0
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Precomputes the RoPE cosine and sine embeddings.

    Args:
      seq_len: The maximum sequence length.
      head_dim: The dimension of each attention head (must be even).
      base: The base value for the inverse frequency calculation.

    Returns:
      cos_emb: Cosine embeddings (1, seq_len, 1, head_dim)
      sin_emb: Sine embeddings (1, seq_len, 1, head_dim)
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")

    # Calculate inverse frequencies (theta_i)
    # theta_i = 1 / (base^(2*i / head_dim)) for i in [0, 1, ..., head_dim/2 - 1]
    inv_freq = 1.0 / (base ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))

    # Calculate position indices (m)
    pos = jnp.arange(seq_len, dtype=jnp.float32)

    # Calculate angles (m * theta_i)
    freqs = jnp.outer(pos, inv_freq)  # Shape: (seq_len, head_dim / 2)

    # Duplicate frequencies for the full head dimension (for both elements in pairs)
    emb = jnp.concatenate((freqs, freqs), axis=-1)  # Shape: (seq_len, head_dim)

    # Calculate cosine and sine embeddings
    cos_emb = jnp.cos(emb)[None, :, None, :]  # Shape: (1, seq_len, 1, head_dim)
    sin_emb = jnp.sin(emb)[None, :, None, :]  # Shape: (1, seq_len, 1, head_dim)

    return cos_emb, sin_emb


class RoPEMultiHeadAttention(nn.Module):
    """Multi-Head Attention with Rotary Positional Embeddings."""

    num_heads: int
    head_dim: int
    rope_base: float = 10000.0
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:  # Added -> None return type
        """Initializes the attention projections."""
        # Check head_dim validity early during setup
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim ({self.head_dim}) must be even for RoPE.")

        # Define layers here - they will be initialized when the module is first called
        total_head_dim = self.num_heads * self.head_dim
        self.query_proj = nn.Dense(
            features=total_head_dim, use_bias=False, dtype=self.dtype, name="query_proj"
        )
        self.key_proj = nn.Dense(
            features=total_head_dim, use_bias=False, dtype=self.dtype, name="key_proj"
        )
        self.value_proj = nn.Dense(
            features=total_head_dim, use_bias=False, dtype=self.dtype, name="value_proj"
        )
        self.output_proj = nn.Dense(
            features=self.num_heads * self.head_dim,  # Output should match embed_dim
            use_bias=False,
            dtype=self.dtype,
            name="output_proj",
        )

    @nn.compact
    # Also using Optional for the mask type hint for clarity with None default
    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray | None = None) -> jnp.ndarray:
        """Forward pass for RoPE MHA.

        Args:
          x: Input tensor (batch_size, seq_len, embed_dim).
          mask: Optional attention mask (batch_size, 1, seq_len, seq_len)
                or (batch_size, 1, 1, seq_len) for causal masking.
                Mask values should be 0 where attention is allowed, -inf otherwise.
                Flax convention often uses boolean masks (True=masked). We'll handle both.

        Returns:
          Output tensor (batch_size, seq_len, embed_dim).
        """
        batch_size, seq_len, embed_dim = x.shape
        total_head_dim = self.num_heads * self.head_dim

        if embed_dim != total_head_dim:
            raise ValueError(
                f"embed_dim ({embed_dim}) must equal num_heads*head_dim ({total_head_dim})"
            )
        # Note: head_dim even check moved to setup for earlier failure

        # 1. Linear projections for Q, K, V
        query = self.query_proj(x)
        key = self.key_proj(x)
        value = self.value_proj(x)

        # 2. Reshape for multi-head processing
        # (batch, seq_len, embed_dim) -> (batch, seq_len, num_heads, head_dim)
        query = query.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        value = value.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # 3. Precompute RoPE embeddings (cosine and sine)
        # We compute them dynamically based on the input sequence length
        cos_emb, sin_emb = precompute_rotary_embeddings(seq_len, self.head_dim, base=self.rope_base)
        # Ensure RoPE embeddings have correct dtype
        cos_emb = cos_emb.astype(self.dtype)
        sin_emb = sin_emb.astype(self.dtype)

        # 4. Apply RoPE to Query and Key
        query = apply_rotary_pos_emb(query, cos_emb, sin_emb)
        key = apply_rotary_pos_emb(key, cos_emb, sin_emb)

        # 5. Transpose for attention calculation: (batch, num_heads, seq_len, head_dim)
        query = query.transpose((0, 2, 1, 3))
        key = key.transpose((0, 2, 1, 3))
        value = value.transpose((0, 2, 1, 3))

        # 6. Scaled Dot-Product Attention
        # Attention scores: (batch, num_heads, seq_len, seq_len)
        attn_scores = jnp.matmul(query, key.transpose((0, 1, 3, 2))) / jnp.sqrt(
            self.head_dim).astype(self.dtype) # Ensure sqrt is correct dtype

        # Apply mask (if provided)
        if mask is not None:
            # Standard Flax causal mask is boolean (True means mask)
            # nn.make_causal_mask returns (1, seq_len, seq_len) or (batch, 1, seq_len, seq_len)
            # Check if mask needs broadcasting or conversion
            if mask.ndim == 2:  # Likely (seq_len, seq_len)
                mask = mask[None, None, :, :]  # -> (1, 1, seq_len, seq_len)
            elif mask.ndim == 3 and mask.shape[1] != self.num_heads:
                # Likely (batch, seq_len, seq_len) or causal (1, sl, sl)
                mask = mask[:, None, :, :]
                # Assume (batch, seq_len, seq_len) -> (batch, 1, seq_len, seq_len)

            # Ensure mask is broadcastable to attn_scores shape
            mask_shape_expected = (batch_size, self.num_heads, seq_len, seq_len)
            if mask.shape != mask_shape_expected:
                 # Attempt broadcasting common causal mask shapes
                 if mask.shape == (1, 1, seq_len, seq_len) or mask.shape == (batch_size, 1,
                        seq_len, seq_len): # Causal mask for all batches/heads
                     mask = jnp.broadcast_to(mask, mask_shape_expected)
                 # Add other broadcasting cases if needed
                 else:
                     raise ValueError(f"Mask shape {mask.shape} != exp shape {mask_shape_expected}")

            # Apply mask: Use large negative number where mask is True
            # (or where mask value is 0 if using 0/-inf convention)
            # Assuming boolean mask convention (True = mask) common in Flax examples
            # If using 0/-inf mask, the logic would be: attn_scores = attn_scores + mask
            attn_scores = jnp.where(mask, jnp.finfo(self.dtype).min, attn_scores)

        # Softmax to get attention weights
        # Shape: (batch, num_heads, seq_len, seq_len)
        attn_weights = jax.nn.softmax(
            attn_scores, axis=-1
        ).astype(self.dtype)  

        # Apply attention weights to Value
        # Output per head: (batch, num_heads, seq_len, head_dim)
        attn_output = jnp.matmul(attn_weights, value)

        # 7. Concatenate heads and final projection
        # Transpose back: (batch, seq_len, num_heads, head_dim)
        attn_output = attn_output.transpose((0, 2, 1, 3))
        # Reshape to (batch, seq_len, embed_dim)
        attn_output = attn_output.reshape(batch_size, seq_len, total_head_dim)

        # Final linear projection
        output = self.output_proj(attn_output)  # Use self.output_proj defined in setup


        return output
