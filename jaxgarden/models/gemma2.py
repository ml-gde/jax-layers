"""
Gemma 2 model implementation for JAX-Layers, based on:
https://arxiv.org/pdf/2408.00118

This implementation is heavily influenced by the original Gemma 2
implementation from DeepMind. https://github.com/google-deepmind/gemma
"""

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from jaxgarden.models.base import BaseConfig, BaseModel
from jaxgarden.models.generation_utils import GenerationMixin


# 1. Configuration
@dataclass
class Gemma2Config(BaseConfig):
    """
    Configuration for Gemma 2. Variant defaults below (from technical report):
    # 2B:  vocab_size=256128, hidden_size=2304, intermediate_size=18432,
    # num_hidden_layers=26, num_attention_heads=8, num_key_value_heads=4, head_dim=256

    # 9B:  vocab_size=256128, hidden_size=3584, intermediate_size=28672,
    # num_hidden_layers=42, num_attention_heads=16, num_key_value_heads=8, head_dim=256

    # 27B: vocab_size=256128, hidden_size=4608, intermediate_size=73728,
    # num_hidden_layers=46, num_attention_heads=32, num_key_value_heads=16, head_dim=128

    This default configuration is based on the original Gemma 2 9B variant.
    """

    vocab_size: int = 256128
    hidden_size: int = 3584
    intermediate_size: int = 28672
    num_hidden_layers: int = 42
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 256
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    attn_logits_soft_cap: float | None = 50.0
    final_logit_soft_cap: float | None = 30.0
    sliding_window_size: int | None = 4096
    pad_token_id: int = 0
    eos_token_id: int = 1
    context_length: int = 8192
    param_dtype: Any = field(default=jnp.bfloat16, metadata={"dtype": True})
    dtype: Any = field(default=jnp.bfloat16, metadata={"dtype": True})
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads // 2
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        # assert self.vocab_size == 256128, "Gemma2 uses 256128 vocab size"
        # assert self.context_length == 8192, "Gemma2 context length is 8192"
        assert self.num_attention_heads % self.num_key_value_heads == 0, (
            "GQA: num_attention_heads must be divisible by num_key_value_heads"
        )
        assert self.hidden_size == self.num_attention_heads * self.head_dim, (
            "hidden_size must equal num_attention_heads * head_dim"
        )


# 2. Core Modules
class Gemma2RMSNorm(nnx.Module):
    """Root Mean Square Layer Normalization.

    This implementation follows the RMSNorm implementation in DeepMind's GEMMA repository:

    https://github.com/google-deepmind/gemma/blob/main/gemma/layers.py#L41
    """

    def __init__(self, dim: int, *, eps: float = 1e-6, rngs: nnx.Rngs):
        super().__init__()
        # Initialize weight/scale parameter to zeros, following the (1 + scale) pattern
        self.weight = nnx.Param(jnp.zeros((dim,), dtype=jnp.bfloat16))
        self.eps = eps

    @nnx.jit()
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        orig_dtype = x.dtype
        x_f32 = x.astype(jnp.float32)
        variance = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
        x_norm = x_f32 * jax.lax.rsqrt(variance + self.eps)

        # Reshape weight to match rank of x_norm for explicit broadcasting
        weight_f32 = self.weight.astype(jnp.float32)
        reshaped_weight = jnp.expand_dims(weight_f32, axis=range(x_norm.ndim - 1))

        # Apply scale as (1 + weight)
        scaled_x = x_norm * (1 + reshaped_weight)
        return scaled_x.astype(orig_dtype)


class Gemma2RotaryEmbedding(nnx.Module):
    """Applies Rotary Position Embedding (RoPE) to input tensors."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,  # Default from original code
        theta: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.theta = theta

        # Precompute inverse frequency
        inv_freq = 1.0 / (self.theta ** (jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim))
        # Store as a direct attribute (JAX array), not nnx.Buffer/State
        self.inv_freq = inv_freq

        # Build cos and sin cached up to max_position_embeddings
        t = jnp.arange(self.max_position_embeddings, dtype=jnp.float32)
        # Use the direct attribute value
        freqs = jnp.einsum("i,j->ij", t, self.inv_freq)
        cos_cached = jnp.cos(freqs)
        sin_cached = jnp.sin(freqs)
        # Store as direct attributes (JAX arrays), not nnx.Buffer/State
        self._cos_cached = cos_cached
        self._sin_cached = sin_cached

    @nnx.jit()
    def __call__(self, x: jnp.ndarray, position_ids: jnp.ndarray) -> jnp.ndarray:
        """Applies RoPE to the input tensor using cached sin/cos values.

        Args:
            x: Input tensor of shape [B, L, N, H].
            position_ids: Position indices of shape [B, L].

        Returns:
            Rotated tensor of the same shape as x.
        """
        orig_dtype = x.dtype
        # x shape: [B, L, N, H]
        # position_ids shape: [B, L]

        # --- Fetch cos and sin from cache ---
        # Access direct attributes
        # self._cos_cached shape: [max_pos, H/2]
        # self._sin_cached shape: [max_pos, H/2]

        # Gather from cache: shapes become [B, L, H/2]
        # Access direct attributes
        cos_gathered = self._cos_cached[position_ids]
        sin_gathered = self._sin_cached[position_ids]

        # Expand dims for broadcasting over heads: [B, L, 1, H/2]
        cos = cos_gathered[:, :, None, :].astype(orig_dtype)
        sin = sin_gathered[:, :, None, :].astype(orig_dtype)

        # --- Apply rotation ---
        x1, x2 = jnp.split(x, 2, axis=-1)  # x1, x2 shape: [B, L, N, H/2]
        # cos, sin shape: [B, L, 1, H/2] - Broadcasts over N dimension
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos
        x_embed = jnp.concatenate((rotated_x1, rotated_x2), axis=-1)

        return x_embed.astype(orig_dtype)


# Separate static helper function for rotation if needed elsewhere
def _rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)


class Gemma2Attention(nnx.Module):
    """Gemma 2 Attention mechanism with GQA and RoPE."""

    def __init__(
        self, layer_idx: int, config: Gemma2Config, *, attention_type: str, rngs: nnx.Rngs
    ):
        super().__init__()
        self.config = config  # Store config
        self.layer_idx = layer_idx
        self.attention_type = attention_type
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim
        self.window = 4096 if attention_type == "local" else 8192
        self.attn_logits_soft_cap = config.attn_logits_soft_cap  # Added
        self.is_local_attn = (layer_idx % 2 != 0) and (config.sliding_window_size is not None)

        self.rope = Gemma2RotaryEmbedding(self.head_dim, theta=config.rope_theta)
        self.q_proj = nnx.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            use_bias=False,
            param_dtype=config.param_dtype,
            rngs=rngs,
        )
        self.k_proj = nnx.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            use_bias=False,
            param_dtype=config.param_dtype,
            rngs=rngs,
        )
        self.v_proj = nnx.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            use_bias=False,
            param_dtype=config.param_dtype,
            rngs=rngs,
        )
        self.o_proj = nnx.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            use_bias=False,
            param_dtype=config.param_dtype,
            rngs=rngs,
        )

    @nnx.jit()
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        position_ids: jnp.ndarray,
        attention_mask: jnp.ndarray | None = None,  # Boolean Input Padding Mask [B, kv_len]
        cache: tuple[jnp.ndarray, jnp.ndarray] | None = None,  # (k_cache, v_cache)
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
        batch_size, q_len, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states).reshape(
            batch_size, q_len, self.num_heads, self.head_dim
        )
        key_states = self.k_proj(hidden_states).reshape(
            batch_size, q_len, self.num_kv_heads, self.head_dim
        )
        value_states = self.v_proj(hidden_states).reshape(
            batch_size, q_len, self.num_kv_heads, self.head_dim
        )

        # Apply RoPE to queries and keys
        query_states = self.rope(query_states, position_ids)  # [B, q_len, num_heads, head_dim]
        key_states = self.rope(key_states, position_ids)  # [B, q_len, num_kv_heads, head_dim]
        # No RoPE for values
        # Transpose to [B, heads, seq, head_dim]
        query_states = query_states.transpose(0, 2, 1, 3)  # [B, num_heads, q_len, head_dim]
        key_states = key_states.transpose(0, 2, 1, 3)  # [B, num_kv_heads, q_len, head_dim]
        value_states = value_states.transpose(0, 2, 1, 3)  # [B, num_kv_heads, q_len, head_dim]
        # Repeat K/V heads for GQA
        key_states = self._repeat_kv(
            key_states, self.num_key_value_groups
        )  # [B, num_heads, q_len, head_dim]
        value_states = self._repeat_kv(value_states, self.num_key_value_groups)
        # KV caching: concatenate along sequence axis (axis=2)
        if cache is not None:
            k_cache, v_cache = cache  # both [B, num_heads, cache_len, head_dim]
            key_states = jnp.concatenate([k_cache, key_states], axis=2)
            value_states = jnp.concatenate([v_cache, value_states], axis=2)
        updated_cache = (key_states, value_states)
        kv_seq_len = key_states.shape[2]  # Total sequence length including cache
        # Scaled Dot-Product Attention
        # Q: [B, num_heads, q_len, head_dim]
        # K: [B, num_heads, kv_seq_len, head_dim]
        # Compute attention weights:
        # [B, num_heads, q_len, head_dim] @ [B, num_heads, head_dim, kv_seq_len]
        attn_weights = jnp.matmul(query_states, key_states.transpose(0, 1, 3, 2))

        # Apply attention scaling factor
        # Apply attention logits soft cap (BEFORE masking)
        if self.attn_logits_soft_cap is not None:
            attn_weights = self.apply_soft_cap(attn_weights, self.attn_logits_soft_cap)

        # --- Apply Mask ---
        if attention_mask is not None and attention_mask.ndim == 4:
            # Direct additive mask: shape [B, 1, q_len, kv_seq_len]
            attn_weights = attn_weights + attention_mask.astype(self.config.dtype)
        else:
            # 1. Causal/Sliding Window Mask [1, 1, q_len, kv_seq_len]
            attn_internal_mask = self._make_sliding_window_mask(q_len, kv_seq_len, dtype=jnp.bool_)
            # 2. Combine with padding mask if provided (boolean 2D mask [B, kv_seq_len])
            if attention_mask is not None:
                # Ensure shape [B, kv_seq_len]
                assert attention_mask.shape == (batch_size, kv_seq_len), (
                    f"Input attention_mask shape {attention_mask.shape} does not match "
                    f"expected ({batch_size}, {kv_seq_len})"
                )
                padding_mask = attention_mask[:, None, None, :].astype(jnp.bool_)
                final_mask = jnp.logical_and(attn_internal_mask, padding_mask)
            else:
                # Broadcast internal mask to batch
                final_mask = jnp.broadcast_to(
                    attn_internal_mask, (batch_size, 1, q_len, kv_seq_len)
                )
            # Apply additive mask bias: 0 for keep, large negative for mask
            neg_inf = jnp.finfo(self.config.dtype).min
            attention_bias = jnp.where(final_mask, 0.0, neg_inf).astype(self.config.dtype)
            attn_weights = attn_weights + attention_bias

        # --- Softmax & Output ---
        attn_weights = jax.nn.softmax(attn_weights, axis=-1).astype(self.config.dtype)

        # Apply attention weights to value states
        # attn_weights: [B, num_heads, seq, kv_len]
        # value_states: [B, kv_len, num_heads, head_dim]
        # attn_output:  [B, num_heads, seq, head_dim]
        attn_output = jnp.matmul(attn_weights, value_states)

        # --- Reshape and Output --- #
        # Transpose back to [B, q_len, num_heads, head_dim]
        attn_output = attn_output.transpose(0, 2, 1, 3)
        # Reshape to [B, q_len, hidden_size]
        attn_output = attn_output.reshape(batch_size, q_len, self.hidden_size)

        # Apply output projection
        attn_output = self.o_proj(attn_output)
        # Always return output and the computed K/V states for this pass
        return attn_output, updated_cache

    def _make_sliding_window_mask(self, q_len: int, kv_len: int, dtype: jnp.dtype) -> jnp.ndarray:
        """Creates a combined causal and sliding window mask. True allows attention."""
        # Creates the lower triangular part for causality
        causal_mask = jnp.tril(jnp.ones((q_len, kv_len), dtype=jnp.bool_))

        # If global attention or no sliding window, causal mask is sufficient
        if not self.is_local_attn or self.config.sliding_window_size is None:
            return causal_mask[None, None, :, :]  # Add batch and head dims

        window = self.config.sliding_window_size

        # Position indices for query and key/value sequences
        # Query positions are relative to the end of the kv sequence
        # Key positions range from 0 to kv_len - 1
        q_pos = jnp.arange(kv_len - q_len, kv_len)[:, None]  # Shape [q_len, 1]
        kv_pos = jnp.arange(kv_len)[None, :]  # Shape [1, kv_len]

        # Sliding window constraint: key_pos > query_pos - window
        window_mask = kv_pos > (q_pos - window)  # Shape [q_len, kv_len]

        # Combine causal and sliding window
        final_mask = jnp.logical_and(causal_mask, window_mask)

        # Expand dims: [q_len, kv_len] -> [1, 1, q_len, kv_len]
        return final_mask[None, None, :, :].astype(dtype)

    def _repeat_kv(self, x: jnp.ndarray, n_rep: int) -> jnp.ndarray:
        """Repeats the key/value heads along the head dimension (axis=1) for GQA."""
        # x shape: [B, N_kv, S, H]
        if n_rep == 1:
            return x
        # Repeat along the N_kv dimension (axis=1)
        # Output shape: [B, N_kv * n_rep, S, H] = [B, N_q, S, H]
        return jnp.repeat(x, n_rep, axis=1)

    @staticmethod
    def apply_soft_cap(x: jnp.ndarray, cap: float) -> jnp.ndarray:
        return cap * jnp.tanh(x / cap)

    @staticmethod
    def rotate_half(x):
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate([-x2, x1], axis=-1)


class Gemma2MLP(nnx.Module):
    def __init__(self, config: Gemma2Config, *, rngs: nnx.Rngs):
        super().__init__()
        # GeGLU: intermediate_size must be doubled for gating
        self.fc1 = nnx.Linear(
            config.hidden_size,
            config.intermediate_size * 2,
            use_bias=False,
            param_dtype=config.param_dtype,
            rngs=rngs,
        )
        self.fc2 = nnx.Linear(
            config.intermediate_size,
            config.hidden_size,
            use_bias=False,
            param_dtype=config.param_dtype,
            rngs=rngs,
        )

    @staticmethod
    def geglu(x: jnp.ndarray) -> jnp.ndarray:
        x1, x2 = jnp.split(x, 2, axis=-1)
        return x1 * jax.nn.gelu(x2)

    @nnx.jit()
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.fc1(x)
        x = Gemma2MLP.geglu(x)
        x = self.fc2(x)
        return x


class Gemma2DecoderLayer(nnx.Module):
    def __init__(self, layer_idx: int, config: Gemma2Config, *, rngs: nnx.Rngs):
        super().__init__()
        # Alternate global/local per layer
        attention_type = "global" if layer_idx % 2 == 0 else "local"
        self.pre_attn_norm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps, rngs=rngs)
        self.attn = Gemma2Attention(layer_idx, config, attention_type=attention_type, rngs=rngs)
        self.post_attn_norm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps, rngs=rngs)
        self.pre_mlp_norm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps, rngs=rngs)
        self.mlp = Gemma2MLP(config, rngs=rngs)
        self.post_mlp_norm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps, rngs=rngs)

    @nnx.jit()
    def __call__(
        self,
        x: jnp.ndarray,
        position_ids: jnp.ndarray,
        attention_mask: jnp.ndarray | None = None,  # Boolean Input Padding Mask [B, kv_len]
        cache: tuple[jnp.ndarray, jnp.ndarray] | None = None,  # (k_cache, v_cache)
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
        # 1. Pre-Attention Norm and Attention
        residual = x
        hidden_states = self.pre_attn_norm(x)
        attn_output, updated_cache = self.attn(hidden_states, position_ids, attention_mask, cache)

        # 2. Post-Attention Norm and Residual 1
        residual = residual + self.post_attn_norm(attn_output)

        # 3. Pre-MLP Norm and MLP
        hidden_states = self.pre_mlp_norm(residual)
        mlp_output = self.mlp(hidden_states)

        # 4. Post-MLP Norm and Residual 2
        output = residual + self.post_mlp_norm(mlp_output)

        return output, updated_cache


# 3. Main Model
class Gemma2ForCausalLM(BaseModel, GenerationMixin):
    def __init__(self, config: Gemma2Config, *, rngs: nnx.Rngs):
        super().__init__(config, dtype=config.dtype, param_dtype=config.param_dtype, rngs=rngs)
        self.embed_tokens = nnx.Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=config.dtype,
            rngs=rngs,
        )
        self.layers = [
            Gemma2DecoderLayer(idx, config, rngs=rngs) for idx in range(config.num_hidden_layers)
        ]
        self.norm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps, rngs=rngs)

    @nnx.jit()
    def __call__(
        self,
        input_ids: jnp.ndarray,  # [B, S]
        position_ids: jnp.ndarray,
        attention_mask: jnp.ndarray
        | None = None,  # [B, S], True for valid tokens, used for *padding*
        cache: list[tuple[jnp.ndarray, jnp.ndarray]]
        | None = None,  # List of (k_cache, v_cache) per layer
    ) -> tuple[jnp.ndarray, list[tuple[jnp.ndarray, jnp.ndarray]] | None]:
        batch_size, seq_length = input_ids.shape

        # --- Input Embeddings ---
        hidden_states = self.embed_tokens(input_ids)
        # Gemma normalizes the embedding output
        hidden_states = hidden_states * jnp.sqrt(float(self.config.hidden_size))
        hidden_states = hidden_states.astype(self.config.dtype)

        # --- Prepare Inputs for Layers ---
        # Compute cache and kv sequence lengths
        cache_len = cache[0][0].shape[2] if cache is not None else 0
        kv_seq_len = cache_len + seq_length

        # Prepare position_ids
        if position_ids is None:
            # If no cache, positions are [0, ..., S-1]
            # If cache, positions are [cache_len, ..., cache_len + S - 1]
            position_ids = jnp.arange(cache_len, kv_seq_len, dtype=jnp.int32)[None, :]
            # position_ids should match the query sequence length (seq_length)
            position_ids = position_ids[:, -seq_length:]  # Ensure shape [1, S]
            position_ids = jnp.broadcast_to(position_ids, (batch_size, seq_length))
        elif position_ids.shape[-1] != seq_length:
            raise ValueError(
                "position_ids shape does not match input_ids shape "
                f"(position_ids: {position_ids.shape}, input_ids: {input_ids.shape})"
            )

        # Prepare attention_mask (padding mask)
        # This mask should cover the entire kv_seq_len for keys/values
        if attention_mask is not None:
            # The input attention_mask corresponds to input_ids [B, S]
            # We need to extend it for the cached keys/values.
            # Assume cached tokens were valid.
            if cache is not None:
                # Create mask for cached part (all True)
                cache_mask = jnp.ones((batch_size, cache_len), dtype=jnp.bool_)
                # Concatenate with the input mask
                padding_mask_2d = jnp.concatenate([cache_mask, attention_mask], axis=1)
            else:
                padding_mask_2d = attention_mask  # No cache, use input mask directly

            # Final shape check for the 2D padding mask
            if padding_mask_2d.shape != (batch_size, kv_seq_len):
                raise ValueError(
                    f"Constructed 2D padding mask shape {padding_mask_2d.shape} "
                    f"does not match expected ({batch_size}, {kv_seq_len})"
                )
        else:
            # If no mask provided, assume all tokens are valid
            padding_mask_2d = jnp.ones((batch_size, kv_seq_len), dtype=jnp.bool_)

        # Reshape the 2D boolean padding mask to 4D log-mask for attention calculation
        # Shape: [B, 1, 1, kv_len]. Log-mask: 0.0 for attend, -inf for ignore.
        # Use the validated `padding_mask_2d` here
        attn_mask_4d = jnp.where(
            padding_mask_2d[:, None, None, :], 0.0, jnp.finfo(self.config.dtype).min
        )

        # --- Pass through Decoder Layers ---
        next_cache_list = []
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            hidden_states, updated_layer_cache = layer(
                hidden_states,
                position_ids,
                attention_mask=attn_mask_4d,  # Pass the 4D log-mask [B, 1, 1, kv_len]
                cache=layer_cache,
            )
            # Always append the updated cache from the layer
            next_cache_list.append(updated_layer_cache)

        hidden_states = self.norm(hidden_states)

        # --- Logits Calculation --- #
        # Final projection using embedding weights (weight tying)
        if self.config.tie_word_embeddings:
            logits = hidden_states @ self.embed_tokens.embedding.T
        else:
            # TODO: Add final LM head linear layer if not tied
            raise NotImplementedError("Separate LM head not implemented yet.")

        # Apply final logit soft capping if specified
        if self.config.final_logit_soft_cap is not None:
            logits = logits / self.config.final_logit_soft_cap

        return logits, next_cache_list
