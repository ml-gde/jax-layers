"""JAX Layers - High-performance neural network layers for JAX."""

from jaxgarden.attention.multi_head_attention import MultiHeadAttention
from jaxgarden.functional.attention import dot_product_attention
from jaxgarden.models.base import BaseConfig, BaseModel
from jaxgarden.models.generation_utils import GenerationMixin
from jaxgarden.models.modernbert import (
    ModernBertAttention,
    ModernBertEmbeddings,
    ModernBERTEncoder,
    ModernBERTForMaskedLM,
    ModernBertLayer,
    ModernBertMLP,
)

__all__ = [
    # Base classes
    "BaseConfig",
    "BaseModel",
    # Mixins
    "GenerationMixin",
    # Models
    "ModernBERTEncoder",
    "ModernBERTForMaskedLM",
    "ModernBertAttention",
    "ModernBertEmbeddings",
    "ModernBertLayer",
    "ModernBertMLP",
    # Attention modules
    "MultiHeadAttention",
    # Functional interfaces
    "dot_product_attention",
]

__version__ = "0.1.0"
