"""JAXgarden - High-performance neural network layers for JAX."""

from jaxgarden.attention.multi_head_attention import MultiHeadAttention
from jaxgarden.functional.attention import dot_product_attention
from jaxgarden.models.base import BaseConfig, BaseModel
from jaxgarden.models.gemma2 import (
    Gemma2Attention,
    Gemma2Config,
    Gemma2ForCausalLM,
    Gemma2MLP,
    Gemma2RMSNorm,
    Gemma2RotaryEmbedding,
)
from jaxgarden.models.gemma3 import (
    Gemma3Attention,
    Gemma3Config,
    Gemma3ForCausalLM,
    Gemma3MLP,
    Gemma3RMSNorm,
    Gemma3RotaryEmbedding,
)
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
from jaxgarden.models.t5 import (
    T5MLP,
    T5Attention,
    T5Block,
    T5Config,
    T5CrossAttention,
    T5ForCausalLM,
    T5LayerNorm,
    T5SelfAttention,
    T5Stack,
)
from jaxgarden.tokenization import Tokenizer  # type: ignore

__all__ = [
    "T5MLP",
    # Base classes
    "BaseConfig",
    "BaseModel",
    # Gemma Models
    "Gemma2Attention",
    "Gemma2Config",
    "Gemma2ForCausalLM",
    "Gemma2MLP",
    "Gemma2RMSNorm",
    "Gemma2RotaryEmbedding",
    # Gemma3 Models
    "Gemma3Attention",
    "Gemma3Config",
    "Gemma3ForCausalLM",
    "Gemma3MLP",
    "Gemma3RMSNorm",
    "Gemma3RotaryEmbedding",
    # Mixins
    "GenerationMixin",
    # Llama Models
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
    # T5 Models
    "T5Attention",
    "T5Block",
    "T5Config",
    "T5CrossAttention",
    "T5ForCausalLM",
    "T5LayerNorm",
    "T5SelfAttention",
    "T5Stack",
    # tokenization
    "Tokenizer",
    # Functional interfaces
    "dot_product_attention",
]

__version__ = "0.2.0"
