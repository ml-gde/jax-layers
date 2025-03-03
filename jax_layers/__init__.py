"""JAX Layers - High-performance neural network layers for JAX."""

from jax_layers.attention.multi_head_attention import MultiHeadAttention
from jax_layers.functional.attention import dot_product_attention

__all__ = [
    # Attention modules
    "MultiHeadAttention",
    # Functional interfaces
    "dot_product_attention",
]

__version__ = "0.1.0"
