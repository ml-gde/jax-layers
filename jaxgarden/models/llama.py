"""LLama model implementation in JAX using Flax NNX.

This module implements the LLama architecture as described in Meta's LLama series of models.
The implementation includes modern transformer architecture with rotary position embeddings (RoPE),
group query attention (GQA), and SwiGLU activation function in the feed-forward network.

See: https://arxiv.org/abs/2302.13971
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx

from jaxgarden.models.base import BaseConfig, BaseModel
from jaxgarden.models.generation_utils import GenerationMixin


@dataclass
class LlamaConfig(BaseConfig):
    """
    Configuration for LLama model.

    This configuration class extends BaseConfig and contains all the parameters
    required to initialize a LLama model. It includes settings for model architecture,
    attention mechanisms, and other hyperparameters.

    Attributes:
        dim: Size of hidden states
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        n_kv_heads: Number of key/value heads (for group query attention)
        head_dim: Dimension of each attention head
        intermediate_size: Size of MLP intermediate layer
        vocab_size: Size of vocabulary
        multiple_of: Ensure dimensions are multiples of this value
        norm_eps: Epsilon for layer normalization
        rope_theta: Base for rotary position embeddings
    """
    dim: int = 2048
    n_layers: int = 16
    n_heads: int = 32
    n_kv_heads: int = 8
    head_dim: int = 64
    intermediate_size: int = 14336
    vocab_size: int = 128256
    multiple_of: int = 256
    norm_eps: float = 1e-05
    rope_theta: float = 500000.0


class LlamaRMSNorm(nnx.Module):
    """Root Mean Square Layer Normalization.
    
    This implementation follows the RMSNorm paper: https://arxiv.org/abs/1910.07467
    Instead of using mean and variance like traditional LayerNorm, RMSNorm only uses
    the root mean square of the inputs for normalization.
    
    Attributes:
        norm_weights: The learned scale parameters
        norm_eps: Small constant for numerical stability
    """
    def __init__(self, dim: int, norm_eps: float = 1e-05, rngs: nnx.Rngs | None = None):
        """Initialize RMSNorm module.
        
        Args:
            dim: Dimension of the input tensor
            norm_eps: Small constant for numerical stability
            rngs: PRNG key collection
        """
        super().__init__(rngs=rngs)
        self.norm_weights = nnx.Param(jnp.zeros((dim,), dtype=jnp.bfloat16))
        self.norm_eps = norm_eps

    @nnx.jit()
    def __call__(self, hidden_states):
        """Apply RMS normalization to input tensor.
        
        Args:
            hidden_states: Input tensor of shape [..., dim]
            
        Returns:
            Normalized tensor with same shape as input
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(jnp.float32)
        squared_mean = jnp.mean(jnp.square(hidden_states), axis=-1, keepdims=True)
        hidden_states = hidden_states * jnp.reciprocal(jnp.sqrt(squared_mean + self.norm_eps))
        return self.norm_weights * hidden_states.astype(input_dtype)


class LlamaRotaryEmbedding(nnx.Module):
    """Rotary Position Embedding (RoPE) implementation for LLama.
    
    Based on: https://arxiv.org/abs/2104.09864
    
    Attributes:
        dim: Dimension of the embeddings
        base: Base for the sinusoidal functions
    """
    def __init__(self, dim: int, base: int = 10000, rngs: nnx.Rngs | None = None):
        """Initialize RoPE module.
        
        Args:
            dim: Dimension of the embeddings (must be even)
            base: Base for the sinusoidal functions
            rngs: PRNG key collection
        """
        super().__init__(rngs=rngs)
        self.dim = dim
        self.base = base

    @nnx.jit()
    def __call__(self, position_ids):
        """Generate rotary embeddings from position ids.
        
        Args:
            position_ids: Position indices of shape [batch_size, seq_len]
            
        Returns:
            Tuple of (cos, sin) tensors for rotary embeddings
        """
        inv_freq = 1.0 / (self.base ** (jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim))
        inv_freq_expanded = jnp.expand_dims(inv_freq, axis=(0, 1))
        position_ids_expanded = jnp.expand_dims(position_ids, axis=(0, 2)).astype(jnp.float32)
        freqs = jnp.einsum("bij,bjk->bijk", position_ids_expanded, inv_freq_expanded)
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        cos = jnp.cos(emb).squeeze(2).astype(jnp.bfloat16)
        sin = jnp.sin(emb).squeeze(2).astype(jnp.bfloat16)
        return cos, sin


class LlamaAttention(nnx.Module):
    """Multi-headed attention with support for Group Query Attention (GQA).
    
    This implements the LLama attention mechanism with rotary position embeddings (RoPE)
    and support for fewer key-value heads than query heads (GQA).
    
    Attributes:
        q_proj: Linear projection for queries
        k_proj: Linear projection for keys
        v_proj: Linear projection for values
        o_proj: Linear projection for output
        rotary_emb: Rotary position embeddings
        head_dim: Dimension of each attention head
        n_heads: Number of attention heads
        n_kv_heads: Number of key/value heads
    """
    def __init__(
        self,
        layer_idx: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        rope_theta: float,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize attention module.
        
        Args:
            layer_idx: Index of the layer
            dim: Size of hidden states
            n_heads: Number of attention heads
            n_kv_heads: Number of key/value heads
            head_dim: Dimension of each attention head
            rope_theta: Base for rotary position embeddings
            rngs: PRNG key collection
        """
        self.q_proj = nnx.Linear(
            dim, n_heads * head_dim, use_bias=False, rngs=rngs, param_dtype=jnp.bfloat16
        )
        self.k_proj = nnx.Linear(
            dim, n_kv_heads * head_dim, use_bias=False, rngs=rngs, param_dtype=jnp.bfloat16
        )
        self.v_proj = nnx.Linear(
            dim, n_kv_heads * head_dim, use_bias=False, rngs=rngs, param_dtype=jnp.bfloat16
        )
        self.o_proj = nnx.Linear(
            n_heads * head_dim, dim, use_bias=False, rngs=rngs, param_dtype=jnp.bfloat16
        )
        self.rotary_emb = LlamaRotaryEmbedding(head_dim, base=rope_theta, rngs=rngs)

        self.head_dim = head_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads

    def apply_rotary_pos_emb(self, q, k, cos, sin, unsqueeze_dim=1):
        """Apply rotary position embeddings to query and key tensors.
        
        Args:
            q: Query tensor
            k: Key tensor
            cos: Cosine component of rotary embeddings
            sin: Sine component of rotary embeddings
            unsqueeze_dim: Dimension to unsqueeze cosine and sine components
            
        Returns:
            Tuple of (rotated_q, rotated_k)
        """
        cos = jnp.expand_dims(cos, axis=unsqueeze_dim)
        sin = jnp.expand_dims(sin, axis=unsqueeze_dim)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed

    def rotate_half(self, x):
        """Rotate half the hidden dims of the input.
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor with half the dimensions rotated
        """
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return jnp.concatenate([-x2, x1], axis=-1)

    def repeat_kv(self, hidden_states, n_repeat):
        """Repeat key/value heads to match the number of query heads.
        
        When using GQA, we need to repeat each key/value head to match
        the number of query heads.
        
        Args:
            hidden_states: Key or value tensor of shape [batch, n_kv_heads, seq_len, head_dim]
            n_repeat: Number of times to repeat each key/value head
            
        Returns:
            Tensor with repeated key/value heads
        """
        batch, n_kv_heads, seq_len, head_dim = hidden_states.shape
        if n_repeat == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].repeat(n_repeat, axis=2)
        return hidden_states.reshape(batch, n_kv_heads * n_repeat, seq_len, head_dim)

    @nnx.jit()
    def __call__(self, x, position_ids):
        """Apply self-attention using queries, keys, and values derived from input x.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            position_ids: Position indices of shape [batch_size, seq_len]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, dim]
        """
        batch_size, seq_len, _ = x.shape
        query = (
            self.q_proj(x)
            .reshape(batch_size, seq_len, self.n_heads, self.head_dim)
            .transpose((0, 2, 1, 3))
        )
        key = (
            self.k_proj(x)
            .reshape(batch_size, seq_len, self.n_kv_heads, self.head_dim)
            .transpose((0, 2, 1, 3))
        )
        value = (
            self.v_proj(x)
            .reshape(batch_size, seq_len, self.n_kv_heads, self.head_dim)
            .transpose((0, 2, 1, 3))
        )
        # Assuming batch_size=1
        cos, sin = self.rotary_emb(position_ids[0])
        query, key = self.apply_rotary_pos_emb(query, key, cos, sin)

        key = self.repeat_kv(key, self.n_heads // self.n_kv_heads)
        value = self.repeat_kv(value, self.n_heads // self.n_kv_heads)

        attn_weights = jnp.matmul(query, jnp.transpose(key, (0, 1, 3, 2)))
        attn_weights = (attn_weights.astype(jnp.float32) / jnp.sqrt(self.head_dim)).astype(
            jnp.bfloat16
        )
        attn_weights = jax.nn.softmax(attn_weights.astype(jnp.float32), axis=-1).astype(
            jnp.bfloat16
        )
        attn_output = (
            jnp.matmul(attn_weights, value).transpose((0, 2, 1, 3)).reshape(batch_size, seq_len, -1)
        )
        output = self.o_proj(attn_output)
        return output


class LlamaMLP(nnx.Module):
    """LLama's MLP implementation with SwiGLU activation.
    
    This implements the SwiGLU MLP used in LLama models:
    down_proj(silu(gate_proj(x)) * up_proj(x))
    
    Attributes:
        gate_proj: Linear projection for gate
        up_proj: Linear projection for up-projection
        down_proj: Linear projection for down-projection
    """
    def __init__(
        self, layer_idx: int, dim: int, intermediate_size: int, rngs: nnx.Rngs | None = None
    ):
        """Initialize MLP module.
        
        Args:
            layer_idx: Index of the layer
            dim: Size of hidden states
            intermediate_size: Size of intermediate layer
            rngs: PRNG key collection
        """
        self.gate_proj = nnx.Linear(
            dim, intermediate_size, use_bias=False, rngs=rngs, param_dtype=jnp.bfloat16
        )
        self.up_proj = nnx.Linear(
            dim, intermediate_size, use_bias=False, rngs=rngs, param_dtype=jnp.bfloat16
        )
        self.down_proj = nnx.Linear(
            intermediate_size, dim, use_bias=False, rngs=rngs, param_dtype=jnp.bfloat16
        )

    @nnx.jit()
    def __call__(self, x):
        """Apply SwiGLU MLP to input tensor.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, dim]
        """
        return self.down_proj(jax.nn.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaTransformerBlock(nnx.Module):
    """LLama transformer block implementation.
    
    This implements a single layer of the LLama transformer, consisting of
    attention, layer normalization, and MLP components.
    
    Attributes:
        input_layernorm: Layer normalization before attention
        attention: Multi-headed attention
        post_attention_layernorm: Layer normalization after attention
        mlp: MLP block
    """
    def __init__(
        self,
        layer_idx: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        rope_theta: float,
        intermediate_size: int,
        norm_eps: float = 1e-05,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize transformer block.
        
        Args:
            layer_idx: Index of the layer
            dim: Size of hidden states
            n_heads: Number of attention heads
            n_kv_heads: Number of key/value heads
            head_dim: Dimension of each attention head
            rope_theta: Base for rotary position embeddings
            intermediate_size: Size of MLP intermediate layer
            norm_eps: Epsilon for layer normalization
            rngs: PRNG key collection
        """
        self.input_layernorm = LlamaRMSNorm(dim=dim, norm_eps=norm_eps, rngs=rngs)
        self.attention = LlamaAttention(
            layer_idx=layer_idx,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            rope_theta=rope_theta,
            rngs=rngs,
        )
        self.post_attention_layernorm = LlamaRMSNorm(dim=dim, norm_eps=norm_eps, rngs=rngs)
        self.mlp = LlamaMLP(
            layer_idx=layer_idx, dim=dim, intermediate_size=intermediate_size, rngs=rngs
        )

    @nnx.jit()
    def __call__(self, x, position_ids):
        """Apply transformer block to input tensor.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            position_ids: Position indices of shape [batch_size, seq_len]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, dim]
        """
        residual = x
        x = self.input_layernorm(x)
        x = self.attention(x, position_ids)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x


class LlamaForCausalLM(BaseModel, GenerationMixin):
    """LLama model for causal language modeling.
    
    This implements the full LLama model for generating text.
    It consists of token embeddings, transformer layers, and language modeling head.
    
    Attributes:
        token_embed: Token embedding layer
        layers: List of transformer blocks
        lm_head: Linear layer for language modeling
        norm: Final layer normalization
    """
    def __init__(
        self,
        config: LlamaConfig,
        *,
        dtype: jnp.dtype | None = None,
        param_dtype: jnp.dtype = jnp.float32,
        precision: jax.lax.Precision | str | None = None,
        rngs: nnx.Rngs,
    ):
        """Initialize LlamaForCausalLM.
        
        Args:
            config: Model configuration
            dtype: Data type for computation
            param_dtype: Data type for parameters
            precision: Precision for matrix multiplication
            rngs: PRNG key collection
        """
        super().__init__(
            config, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=rngs
        )

        self.token_embed = nnx.Embed(
            num_embeddings=config.vocab_size,
            features=config.dim,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.layers = [
            LlamaTransformerBlock(
                layer_idx=idx,
                dim=config.dim,
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads,
                head_dim=config.head_dim,
                rope_theta=config.rope_theta,
                intermediate_size=config.intermediate_size,
                norm_eps=config.norm_eps,
                rngs=rngs,
            )
            for idx in range(config.n_layers)
        ]
        self.lm_head = nnx.Linear(
            config.dim, config.vocab_size, use_bias=False, rngs=rngs, param_dtype=param_dtype
        )
        self.norm = LlamaRMSNorm(dim=config.head_dim, rngs=rngs)

    @nnx.jit()
    def __call__(self, input_ids, position_ids):
        """Forward pass of the LLama model.
        
        Args:
            input_ids: Input token ids of shape [batch_size, seq_len]
            position_ids: Position indices of shape [batch_size, seq_len]
            
        Returns:
            Logits for next token prediction of shape [batch_size, seq_len, vocab_size]
        """
        assert input_ids.shape[0] == 1, "Only batch size 1 is supported"
        x = self.token_embed(input_ids)
        for layer in self.layers:
            x = layer(x, position_ids)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits
