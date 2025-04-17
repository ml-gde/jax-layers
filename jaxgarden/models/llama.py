# Llama 3.2-1B, using NNX
#
# TODO: Add Chex assertions to check shapes, types, etc.

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx

from jaxgarden.models.base import BaseConfig, BaseModel
from jaxgarden.models.generation_utils import GenerationMixin


@dataclass
class LlamaConfig(BaseConfig):
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
    def __init__(self, dim: int, norm_eps: float = 1e-05, rngs: nnx.Rngs | None = None):
        super().__init__(rngs=rngs)
        self.norm_weights = nnx.Param(jnp.zeros((dim,), dtype=jnp.bfloat16))
        self.norm_eps = norm_eps

    @nnx.jit()
    def __call__(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(jnp.float32)
        squared_mean = jnp.mean(jnp.square(hidden_states), axis=-1, keepdims=True)
        hidden_states = hidden_states * jnp.reciprocal(jnp.sqrt(squared_mean + self.norm_eps))
        return self.norm_weights * hidden_states.astype(input_dtype)


class LlamaRotaryEmbedding(nnx.Module):
    def __init__(self, dim: int, base: int = 10000, rngs: nnx.Rngs | None = None):
        super().__init__(rngs=rngs)
        self.dim = dim
        self.base = base

    @nnx.jit()
    def __call__(self, position_ids):
        inv_freq = 1.0 / (self.base ** (jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim))
        inv_freq_expanded = jnp.expand_dims(inv_freq, axis=(0, 1))
        position_ids_expanded = jnp.expand_dims(position_ids, axis=(0, 2)).astype(jnp.float32)
        freqs = jnp.einsum("bij,bjk->bijk", position_ids_expanded, inv_freq_expanded)
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        cos = jnp.cos(emb).squeeze(2).astype(jnp.bfloat16)
        sin = jnp.sin(emb).squeeze(2).astype(jnp.bfloat16)
        return cos, sin


class LlamaAttention(nnx.Module):
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

    # Alternative implementation:
    # https://github.com/google/flax/blob/5d896bc1a2c68e2099d147cd2bc18ebb6a46a0bd/examples/gemma/positional_embeddings.py#L45
    def apply_rotary_pos_emb(self, q, k, cos, sin, unsqueeze_dim=1):
        cos = jnp.expand_dims(cos, axis=unsqueeze_dim)
        sin = jnp.expand_dims(sin, axis=unsqueeze_dim)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed

    def rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return jnp.concatenate([-x2, x1], axis=-1)

    def repeat_kv(self, hidden_states, n_repeat):
        batch, n_kv_heads, seq_len, head_dim = hidden_states.shape
        if n_repeat == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].repeat(n_repeat, axis=2)
        return hidden_states.reshape(batch, n_kv_heads * n_repeat, seq_len, head_dim)

    @nnx.jit()
    def __call__(self, x, position_ids):
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
    def __init__(
        self, layer_idx: int, dim: int, intermediate_size: int, rngs: nnx.Rngs | None = None
    ):
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
        return self.down_proj(jax.nn.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaTransformerBlock(nnx.Module):
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
    def __init__(
        self,
        config: LlamaConfig,
        *,
        dtype: jnp.dtype | None = None,
        param_dtype: jnp.dtype = jnp.float32,
        precision: jax.lax.Precision | str | None = None,
        rngs: nnx.Rngs,
    ):
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
        assert input_ids.shape[0] == 1, "Only batch size 1 is supported"
        x = self.token_embed(input_ids)
        for layer in self.layers:
            x = layer(x, position_ids)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits
