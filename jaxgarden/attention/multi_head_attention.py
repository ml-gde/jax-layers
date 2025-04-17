"""MultiHeadAttention implementation with Flash Attention support for JAX Layers."""

from collections.abc import Callable
from typing import Any, Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from flax.nnx.nn.linear import default_kernel_init

from jaxgarden.functional.attention import dot_product_attention


class MultiHeadAttention(nnx.MultiHeadAttention):
    """Multi-head attention with support for Flash Attention.

    This class extends Flax NNX's MultiHeadAttention to support Flash Attention
    through JAX's dot_product_attention implementation parameter.

    Example usage:

    ```python
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    from jax_layers.attention import MultiHeadAttention

    # Create a MultiHeadAttention module with Flash Attention support
    attention = MultiHeadAttention(
        num_heads=8,
        in_features=512,
        implementation="cudnn",  # Use cuDNN's Flash Attention if available
        rngs=nnx.Rngs(0),
    )

    # Initialize parameters
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (2, 128, 512))  # (batch, seq_length, hidden_dim)

    # Create a causal attention mask
    mask = jnp.tril(jnp.ones((2, 1, 128, 128)))  # (batch, 1, q_len, kv_len)

    # Apply the model
    output = attention(x, mask=mask)
    ```
    """

    def __init__(
        self,
        num_heads: int,
        in_features: int,
        qkv_features: int | None = None,
        out_features: int | None = None,
        *,
        dtype: jnp.dtype | None = None,
        param_dtype: jnp.dtype = jnp.float32,
        broadcast_dropout: bool = True,
        dropout_rate: float = 0.0,
        deterministic: bool | None = None,
        precision: jax.lax.Precision | str | None = None,
        kernel_init: Callable = default_kernel_init,
        out_kernel_init: Callable | None = None,
        bias_init: Callable = nnx.initializers.zeros,
        out_bias_init: Callable | None = None,
        use_bias: bool = True,
        attention_fn: Callable | None = None,
        decode: bool | None = None,
        normalize_qk: bool = False,
        qkv_dot_general: Callable | None = None,
        out_dot_general: Callable | None = None,
        qkv_dot_general_cls: type | None = None,
        out_dot_general_cls: type | None = None,
        implementation: Literal["xla", "cudnn", "flash"] | None = None,
        rngs: nnx.Rngs,
    ):
        """Initialize the MultiHeadAttention module.

        Args:
            num_heads: number of attention heads.
            in_features: int or tuple with number of input features.
            qkv_features: dimension of the key, query, and value.
            out_features: dimension of the last projection.
            dtype: the dtype of the computation.
            param_dtype: the dtype passed to parameter initializers.
            broadcast_dropout: bool: use a broadcasted dropout along batch dims.
            dropout_rate: dropout rate.
            deterministic: if false, the attention weight is masked randomly using dropout.
            precision: numerical precision of the computation.
            kernel_init: initializer for the kernel of the Dense layers.
            out_kernel_init: initializer for the kernel of the output Dense layer.
            bias_init: initializer for the bias of the Dense layers.
            out_bias_init: initializer for the bias of the output Dense layer.
            use_bias: bool: whether pointwise QKVO dense transforms use bias.
            attention_fn: dot_product_attention or compatible function.
            decode: whether to prepare and use an autoregressive cache.
            normalize_qk: should QK normalization be applied.
            qkv_dot_general: dot_general function for QKV projection.
            out_dot_general: dot_general function for output projection.
            qkv_dot_general_cls: dot_general class for QKV projection.
            out_dot_general_cls: dot_general class for output projection.
            implementation: which implementation to use for attention. Options are:
              - "xla": Use XLA's default implementation
              - "cudnn": Use cuDNN's Flash Attention implementation (if available)
              - "flash": Alias for "cudnn"
              - None: Automatically select the best available implementation
            rngs: random number generator keys.
        """
        # Create a custom attention function that uses our dot_product_attention
        # with the specified implementation
        if attention_fn is None:

            def custom_attention_fn(
                query: jnp.ndarray,
                key: jnp.ndarray,
                value: jnp.ndarray,
                bias: jnp.ndarray | None = None,
                mask: jnp.ndarray | None = None,
                **kwargs: Any,
            ) -> jnp.ndarray:
                return dot_product_attention(
                    query=query,
                    key=key,
                    value=value,
                    bias=bias,
                    mask=mask,
                    implementation=implementation,
                    **kwargs,
                )

            attention_fn = custom_attention_fn

        # Initialize the parent class with our custom attention function
        super().__init__(
            num_heads=num_heads,
            in_features=in_features,
            qkv_features=qkv_features,
            out_features=out_features,
            dtype=dtype,
            param_dtype=param_dtype,
            broadcast_dropout=broadcast_dropout,
            dropout_rate=dropout_rate,
            deterministic=deterministic,
            precision=precision,
            kernel_init=kernel_init,
            out_kernel_init=out_kernel_init,
            bias_init=bias_init,
            out_bias_init=out_bias_init,
            use_bias=use_bias,
            attention_fn=attention_fn,
            decode=decode,
            normalize_qk=normalize_qk,
            qkv_dot_general=qkv_dot_general,
            out_dot_general=out_dot_general,
            qkv_dot_general_cls=qkv_dot_general_cls,
            out_dot_general_cls=out_dot_general_cls,
            rngs=rngs,
        )
