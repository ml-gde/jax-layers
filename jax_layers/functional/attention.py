"""Attention functions for JAX Layers.

This module provides attention functions that are compatible with both JAX and Flax NNX.
"""

from typing import Literal

import jax
import jax.numpy as jnp


def dot_product_attention(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    bias: jnp.ndarray | None = None,
    mask: jnp.ndarray | None = None,
    broadcast_dropout: bool = True,
    dropout_rng: jnp.ndarray | None = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    dtype: jnp.dtype | None = None,
    precision: jax.lax.Precision | str | None = None,
    implementation: Literal["xla", "cudnn", "flash"] | None = None,
    module: object | None = None,
) -> jnp.ndarray:
    """Computes dot-product attention with optional Flash Attention support.

    This function provides a wrapper around JAX's dot_product_attention with the option
    to use Flash Attention when available. It follows the Flax NNX interface while
    allowing the use of different implementations through the implementation parameter.

    Args:
        query: queries for calculating attention with shape of
          `[batch..., q_length, num_heads, qk_depth_per_head]`.
        key: keys for calculating attention with shape of
          `[batch..., kv_length, num_heads, qk_depth_per_head]`.
        value: values to be used in attention with shape of
          `[batch..., kv_length, num_heads, v_depth_per_head]`.
        bias: bias for the attention weights. This should be broadcastable to
          the shape [batch..., num_heads, q_length, kv_length].
        mask: mask for the attention weights. This should be broadcastable to
          the shape [batch..., num_heads, q_length, kv_length].
        broadcast_dropout: bool: use a broadcasted dropout along batch dims.
        dropout_rng: JAX PRNGKey: to be used for dropout.
        dropout_rate: dropout rate.
        deterministic: bool, deterministic or not (to apply dropout).
        dtype: the dtype of the computation (default: infer from inputs).
        precision: numerical precision of the computation.
        implementation: which implementation to use. Options are:
          - "xla": Use XLA's default implementation
          - "cudnn": Use cuDNN's Flash Attention implementation (if available)
          - "flash": Alias for "cudnn"
          - None: Automatically select the best available implementation
        module: the Module that will sow the attention weights.

    Returns:
        Output of shape `[batch..., q_length, num_heads, v_depth_per_head]`.
    """
    # Map "flash" to "cudnn" for clarity
    if implementation == "flash":
        implementation = "cudnn"

    # Convert mask to bias if needed
    if mask is not None:
        # In JAX, mask=True means keep the value, while in Flax mask=False means mask out
        # So we need to invert the mask and convert to a bias
        if bias is None:
            bias = jnp.where(mask, 0.0, -1e10)
        else:
            bias = jnp.where(mask, bias, -1e10)

    # Call JAX's dot_product_attention with the implementation parameter
    return jax.nn.dot_product_attention(
        query=query,
        key=key,
        value=value,
        bias=bias,
        # JAX-specific parameters
        scale=None,  # Use default scaling
        is_causal=False,  # We handle causal masking through the bias/mask
        query_seq_lengths=None,
        key_value_seq_lengths=None,
        local_window_size=None,
        implementation=implementation,
    )
