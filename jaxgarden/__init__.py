"""JAXgarden - High-performance neural network layers for JAX."""

from jaxgarden.attention.multi_head_attention import MultiHeadAttention
from jaxgarden.functional.attention import dot_product_attention
from jaxgarden.models.base import BaseConfig, BaseModel
from jaxgarden.models.generation_utils import GenerationMixin
from jaxgarden.models.llama import (
    LlamaAttention,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    LlamaTransformerBlock,
)
from jaxgarden.models.modernbert import (
    ModernBertAttention,
    ModernBertEmbeddings,
    ModernBERTEncoder,
    ModernBERTForMaskedLM,
    ModernBertLayer,
    ModernBertMLP,
)
from jaxgarden.tokenization import Tokenizer

__all__ = [
    # Base classes
    "BaseConfig",
    "BaseModel",
    # Mixins
    "GenerationMixin",
    # Models
    "LlamaAttention",
    "LlamaConfig",
    "LlamaForCausalLM",
    "LlamaMLP",
    "LlamaRMSNorm",
    "LlamaRotaryEmbedding",
    "LlamaTransformerBlock",
    "ModernBERTEncoder",
    "ModernBERTForMaskedLM",
    "ModernBertAttention",
    "ModernBertEmbeddings",
    "ModernBertLayer",
    "ModernBertMLP",
    # Attention modules
    "MultiHeadAttention",
    # tokenization
    "Tokenizer",
    # Functional interfaces
    "dot_product_attention",
]

__version__ = "0.2.0"
