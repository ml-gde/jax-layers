"""
T5 model implementation in JAX using Flax NNX.

This module implements the T5 architecture as described in Google's T5 series of models.

See: https://jmlr.org/papers/volume21/20-074/20-074.pdf
"""

import typing
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from flax.nnx import combine_masks, dot_product_attention, make_causal_mask
from jax.typing import ArrayLike

from jaxgarden.models.base import BaseConfig, BaseModel
from jaxgarden.models.generation_utils import GenerationMixin


@dataclass
class T5Config(BaseConfig):
    """
    Configuration for T5 model.

    This configuration class extends BaseConfig and contains all the parameters
    required to initialize a T5 model. It includes settings for model architecture,
    attention mechanisms, and other hyperparameters.

    This default configuration is based on the original t5-base variant.

    Attributes:
        vocab_size: Size of vocabulary
        hidden_size: Size of hidden states
        dim_ff: Size of feed-forward layer
        dim_kv: Size of key/value states
        intermediate_dim: Size of intermediate layer
        num_heads: Number of attention heads
        num_layers: Number of layers
        relative_attention_num_buckets: Number of buckets for relative attention
        relative_attention_max_distance: Maximum distance for relative attention
        initializer_factor: Factor for initializing weights
        dropout_rate: Dropout rate
        layer_norm_epsilon: Epsilon for layer normalization
        dtype: Data type for the model
    """

    vocab_size: int = 32128
    hidden_size: int = 768
    dim_ff: int = 3072
    dim_kv: int = 64
    intermediate_dim: int = 4096
    num_heads: int = 12
    num_layers: int = 12
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    initializer_factor: float = 1.0
    dropout_rate: float = 0.1
    layer_norm_epsilon: float = 1e-6
    dtype: jnp.dtype = jnp.float32


class T5LayerNorm(nnx.Module):
    """
    LayerNorm module in the T5 style; No bias and no subtraction of mean.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        *,
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.eps = eps
        self.dtype = dtype
        self.weight = nnx.Param(nnx.initializers.ones(rngs.params(), (dim,), dtype=dtype))

    def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        variance = jnp.power(hidden_states.astype(jnp.float32), 2).mean(axis=-1, keepdims=True)
        hidden_states = hidden_states / jnp.sqrt(variance + self.eps)

        hidden_states = hidden_states.astype(self.dtype)
        return self.weight * hidden_states


class T5MLP(nnx.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        initializer_factor: float = 1.0,
        dropout_rate: float = 0.1,
        *,
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.dim = dim
        self.dtype = dtype
        self.intermediate_dim = intermediate_dim
        self.initializer_factor = initializer_factor

        input_init_std = self.initializer_factor * (self.dim**-0.5)
        output_init_std = self.initializer_factor * (self.intermediate_dim**-0.5)

        self.layer_norm = T5LayerNorm(dim=dim, dtype=dtype, rngs=rngs)
        self.input_dense = nnx.Linear(
            in_features=dim,
            out_features=intermediate_dim,
            use_bias=False,
            kernel_init=nnx.initializers.normal(input_init_std, dtype=dtype),
            dtype=dtype,
            rngs=rngs,
        )
        self.act = nnx.relu
        self.dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.output_dense = nnx.Linear(
            in_features=intermediate_dim,
            out_features=dim,
            use_bias=False,
            kernel_init=nnx.initializers.normal(output_init_std, dtype=dtype),
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        hidden_states = self.layer_norm(x)
        hidden_states = self.input_dense(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = self.output_dense(hidden_states)
        x = x + self.dropout(hidden_states, deterministic=deterministic)
        return x.astype(self.dtype)


class T5Attention(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        dim_kv: int,
        num_heads: int,
        relative_attention_num_buckets: int,
        relative_attention_max_distance: int,
        causal: bool | None = False,
        has_relative_attention_bias: bool | None = False,
        initializer_factor: float = 1.0,
        dropout_rate: float = 0.1,
        *,
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.hidden_size = hidden_size
        self.dim_kv = dim_kv
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.causal = causal
        self.has_relative_attention_bias = has_relative_attention_bias
        self.initializer_factor = initializer_factor
        self.dropout_rate = dropout_rate

        self.dtype = dtype
        self.rngs = rngs

        self.inner_dim = self.num_heads * self.dim_kv

        q_init_std = self.initializer_factor * ((self.inner_dim * self.dim_kv) ** -0.5)
        kv_init_std = self.initializer_factor * (self.inner_dim**-0.5)
        o_init_std = self.initializer_factor * (self.inner_dim**-0.5)

        self.q = nnx.Linear(
            in_features=self.hidden_size,
            out_features=self.inner_dim,
            kernel_init=nnx.initializers.normal(q_init_std, dtype=dtype),
            dtype=dtype,
            rngs=rngs,
            use_bias=False,
        )
        self.k = nnx.Linear(
            in_features=self.hidden_size,
            out_features=self.inner_dim,
            kernel_init=nnx.initializers.normal(kv_init_std, dtype=dtype),
            dtype=dtype,
            rngs=rngs,
            use_bias=False,
        )
        self.v = nnx.Linear(
            in_features=self.hidden_size,
            out_features=self.inner_dim,
            kernel_init=nnx.initializers.normal(kv_init_std, dtype=dtype),
            dtype=dtype,
            rngs=rngs,
            use_bias=False,
        )
        self.o = nnx.Linear(
            in_features=self.inner_dim,
            out_features=self.hidden_size,
            kernel_init=nnx.initializers.normal(o_init_std, dtype=dtype),
            dtype=dtype,
            rngs=rngs,
            use_bias=False,
        )

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nnx.Embed(
                num_embeddings=self.relative_attention_num_buckets,
                features=self.num_heads,
                embedding_init=nnx.initializers.normal(kv_init_std, dtype=dtype),
                dtype=dtype,
                rngs=rngs,
            )

    @typing.no_type_check
    @staticmethod
    def _relative_position_bucket(
        relative_position: ArrayLike,
        bidirectional: bool = True,
        num_buckets: int = 32,
        max_distance: int = 128,
    ) -> jnp.ndarray:
        """
        Borrowed from Hugging Face T5 implementation:
        https://github.com/huggingface/transformers/blob/b59386dc0a44eef0b6c671ed6ed09a76b75235e8/src/transformers/models/t5/modeling_flax_t5.py#L232

        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on
        """  # noqa: E501

        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0) * num_buckets
            relative_position = jnp.abs(relative_position)
        else:
            relative_position = -jnp.clip(relative_position, max=0)
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically
        # bigger bins in positions upto max_distance
        relative_position_if_large = max_exact + (
            jnp.log(relative_position / max_exact)
            / jnp.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        )
        relative_position_if_large = jnp.clip(relative_position_if_large, max=num_buckets - 1)

        relative_buckets += jnp.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets.astype("i4")

    def compute_bias(self, query_length: int, key_length: int) -> jnp.ndarray:
        """
        Compute binned relative position bias

        Borrowed from Hugging Face T5 implementation:
        https://github.com/huggingface/transformers/blob/b59386dc0a44eef0b6c671ed6ed09a76b75235e8/src/transformers/models/t5/modeling_flax_t5.py#L267
        """
        context_position = jnp.arange(query_length, dtype="i4")[:, None]
        memory_position = jnp.arange(key_length, dtype="i4")[None, :]

        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=(not self.causal),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )

        values = self.relative_attention_bias(relative_position_bucket)
        values = values.transpose((2, 0, 1))[None, :, :, :]
        return values

    def _split_heads(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.dim_kv))

    def _merge_heads(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        return hidden_states.reshape(hidden_states.shape[:2] + (self.inner_dim,))

    def _create_position_bias(
        self,
        key_states: jnp.ndarray,
        query_states: jnp.ndarray,
        attention_mask: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        key_length = key_states.shape[1]
        query_length = query_states.shape[1]

        if self.has_relative_attention_bias:
            position_bias = self.compute_bias(query_length, key_length)
        elif attention_mask is not None:
            position_bias = jnp.zeros_like(attention_mask)
        else:
            position_bias = jnp.zeros(
                (1, self.num_heads, query_length, key_length), dtype=self.dtype
            )

        return position_bias

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray | None = None,
        key_value_states: jnp.ndarray | None = None,
        position_bias: jnp.ndarray | None = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        batch_size, seq_length = hidden_states.shape[:2]

        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, seq_length), dtype="bool")

        # ----- Q, K, V -----
        query_states = self.q(hidden_states)
        key_states = self.k(hidden_states) if key_value_states is None else self.k(key_value_states)
        value_states = (
            self.v(hidden_states) if key_value_states is None else self.v(key_value_states)
        )
        query_states = self._split_heads(query_states)
        key_states = self._split_heads(key_states)
        value_states = self._split_heads(value_states)

        # ----- Attention Mask -----
        if self.causal:
            causal_attention_mask = make_causal_mask(attention_mask, dtype="bool")
            causal_attention_mask = jnp.broadcast_to(
                causal_attention_mask, (batch_size,) + causal_attention_mask.shape[1:]
            )
            attention_mask_broadcast = jnp.broadcast_to(
                jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_attention_mask.shape
            )
            attention_mask = combine_masks(
                attention_mask_broadcast, causal_attention_mask, dtype=self.dtype
            )
        else:
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))

        if attention_mask is not None:
            mask_value = jnp.finfo(self.dtype).min
            attention_mask = jax.lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, mask_value).astype(self.dtype),
            )

        if position_bias is None:
            # compute position bias (only for first layer)
            position_bias = self._create_position_bias(
                key_states,
                query_states,
                attention_mask,
            )

            if attention_mask is not None:
                position_bias = position_bias + attention_mask

        # create dropout rng
        dropout_rng = None
        if not deterministic and self.dropout_rate > 0.0 and self.rngs is not None:
            dropout_rng = self.rngs.dropout()

        # --- Attention ---
        attn_weights = dot_product_attention(
            query_states,
            key_states,
            value_states,
            bias=position_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout_rate,
            broadcast_dropout=True,
            deterministic=deterministic,
            dtype=self.dtype,
        )

        attn_output = self._merge_heads(attn_weights)
        attn_output = self.o(attn_output)
        return attn_output


class T5SelfAttention(nnx.Module):
    def __init__(
        self,
        config: T5Config,
        causal: bool = True,
        has_relative_attention_bias: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.attention = T5Attention(
            hidden_size=config.hidden_size,
            dim_kv=config.dim_kv,
            num_heads=config.num_heads,
            relative_attention_num_buckets=config.relative_attention_num_buckets,
            relative_attention_max_distance=config.relative_attention_max_distance,
            causal=causal,
            has_relative_attention_bias=has_relative_attention_bias,
            rngs=rngs,
        )

        self.layer_norm = T5LayerNorm(dim=config.hidden_size, dtype=config.dtype, rngs=rngs)
        self.dropout = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray | None = None,
        key_value_states: jnp.ndarray | None = None,
        position_bias: jnp.ndarray | None = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        hidden_states = self.layer_norm(hidden_states)
        attn_output = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            deterministic=deterministic,
        )
        hidden_states = hidden_states + self.dropout(attn_output, deterministic=deterministic)
        return hidden_states.astype(self.config.dtype)


class T5CrossAttention(nnx.Module):
    def __init__(
        self,
        config: T5Config,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.attention = T5Attention(
            hidden_size=config.hidden_size,
            dim_kv=config.dim_kv,
            num_heads=config.num_heads,
            relative_attention_num_buckets=config.relative_attention_num_buckets,
            relative_attention_max_distance=config.relative_attention_max_distance,
            causal=False,
            has_relative_attention_bias=False,
            rngs=rngs,
        )

        self.layer_norm = T5LayerNorm(dim=config.hidden_size, dtype=config.dtype, rngs=rngs)
        self.dropout = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray | None = None,
        key_value_states: jnp.ndarray | None = None,
        position_bias: jnp.ndarray | None = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        hidden_states = self.layer_norm(hidden_states)
        attn_output = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            deterministic=deterministic,
        )
        hidden_states = hidden_states + self.dropout(attn_output, deterministic=deterministic)
        return hidden_states.astype(self.config.dtype)


class T5Block(nnx.Module):
    def __init__(
        self,
        config: T5Config,
        causal: bool = True,
        has_relative_attention_bias: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.causal = causal

        self.layer: list[Callable] = [
            T5SelfAttention(
                config=config,
                causal=causal,
                has_relative_attention_bias=has_relative_attention_bias,
                rngs=rngs,
            )
        ]
        if self.causal:
            self.layer.append(T5CrossAttention(config=config, rngs=rngs))

        self.layer.append(
            T5MLP(
                dim=config.hidden_size,
                intermediate_dim=config.intermediate_dim,
                initializer_factor=config.initializer_factor,
                dropout_rate=config.dropout_rate,
                rngs=rngs,
            )
        )

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray | None = None,
        position_bias: jnp.ndarray | None = None,
        encoder_hidden_states: jnp.ndarray | None = None,
        encoder_attention_mask: jnp.ndarray | None = None,
        encoder_decoder_position_bias: jnp.ndarray | None = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        hidden_states = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            deterministic=deterministic,
        )

        do_cross_attn = self.causal and encoder_hidden_states is not None
        if do_cross_attn:
            hidden_states = self.layer[1](
                hidden_states,
                attention_mask=encoder_attention_mask,
                key_value_states=encoder_hidden_states,
                position_bias=encoder_decoder_position_bias,
                deterministic=deterministic,
            )

        hidden_states = self.layer[-1](
            hidden_states,
            deterministic=deterministic,
        )

        return hidden_states.astype(self.config.dtype)


class T5LayerCollection(nnx.Module):
    def __init__(
        self,
        config: T5Config,
        causal: bool = True,
        has_relative_attention_bias: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.block = T5Block(
            config=config,
            causal=causal,
            has_relative_attention_bias=has_relative_attention_bias,
            rngs=rngs,
        )

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray | None = None,
        position_bias: jnp.ndarray | None = None,
        encoder_hidden_states: jnp.ndarray | None = None,
        encoder_attention_mask: jnp.ndarray | None = None,
        encoder_decoder_position_bias: jnp.ndarray | None = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        return self.block(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            encoder_decoder_position_bias=encoder_decoder_position_bias,
            deterministic=deterministic,
        )


class T5BlockCollection(nnx.Module):
    def __init__(
        self,
        config: T5Config,
        num_layers: int,
        causal: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.causal = causal
        self.blocks = [
            T5LayerCollection(
                config=config,
                causal=causal,
                has_relative_attention_bias=(i == 0),
                rngs=rngs,
            )
            for i in range(num_layers)
        ]

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: jnp.ndarray | None = None,
        encoder_hidden_states: jnp.ndarray | None = None,
        encoder_attention_mask: jnp.ndarray | None = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        position_bias = None
        encoder_decoder_position_bias = None

        for layer_module in self.blocks:
            hidden_states = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                deterministic=deterministic,
            )
        return hidden_states


class T5Stack(nnx.Module):
    def __init__(
        self,
        config: T5Config,
        embed_tokens: nnx.Embed,
        num_layers: int,
        causal: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.embed_tokens = embed_tokens
        self.dropout = nnx.Dropout(rate=config.dropout_rate, rngs=rngs)
        self.block = T5BlockCollection(
            config=config,
            num_layers=num_layers,
            causal=causal,
            rngs=rngs,
        )
        self.final_layer_norm = T5LayerNorm(
            dim=config.hidden_size, eps=config.layer_norm_epsilon, dtype=config.dtype, rngs=rngs
        )

    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: jnp.ndarray | None = None,
        encoder_hidden_states: jnp.ndarray | None = None,
        encoder_attention_mask: jnp.ndarray | None = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)

        hidden_states = self.block(
            hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            deterministic=deterministic,
        )

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states


class T5ForCausalLM(GenerationMixin, BaseModel):
    config: T5Config

    def __init__(self, config: T5Config, *, rngs: nnx.Rngs) -> None:
        super().__init__(config, dtype=config.dtype, param_dtype=config.dtype, rngs=rngs)

        self.shared = nnx.Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            embedding_init=nnx.initializers.normal(stddev=config.initializer_factor),
            dtype=config.dtype,
            rngs=rngs,
        )

        self.encoder = T5Stack(
            config=config,
            embed_tokens=self.shared,
            num_layers=config.num_layers,
            causal=False,
            rngs=rngs,
        )

        self.decoder = T5Stack(
            config=config,
            embed_tokens=self.shared,
            num_layers=config.num_layers,
            causal=True,
            rngs=rngs,
        )

        self.lm_head = nnx.Linear(
            in_features=config.hidden_size,
            out_features=config.vocab_size,
            kernel_init=nnx.initializers.normal(stddev=config.initializer_factor),
            use_bias=False,
            dtype=config.dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: jnp.ndarray | None = None,
        encoder_hidden_states: jnp.ndarray | None = None,
        encoder_attention_mask: jnp.ndarray | None = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        encoder_hidden_states = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            deterministic=deterministic,
        )

        decoder_hidden_states = self.decoder(
            input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            deterministic=deterministic,
        )

        logits = self.lm_head(decoder_hidden_states)
        return logits

    @typing.no_type_check
    def convert_weights_from_hf(
        self, state: nnx.State | dict[str, jnp.ndarray], weights: Iterator[tuple[Any, Any]]
    ) -> None:
        for wholekey, tensor in weights:
            keys = wholekey.split(".")

            # shared embedding
            if keys[0] == "shared" and keys[1] == "embedding":
                state["shared"].embedding.value = tensor
                # Tie lm_head if needed
                if "lm_head" in state:
                    state["lm_head"].kernel.value = tensor.T
                continue

            # lm_head
            if keys[0] == "lm_head" and keys[1] == "kernel":
                state["lm_head"].kernel.value = tensor
                continue

            # Encoder/Decoder blocks
            if keys[0] in ("encoder", "decoder") and keys[1] == "block":
                block_type = keys[0]  # encoder or decoder
                block_idx = int(keys[2])
                layer_type = keys[
                    4
                ]  # 'SelfAttention', 'EncDecAttention', 'DenseReluDense', 'layer_norm'

                if layer_type in ("SelfAttention", "EncDecAttention"):
                    if keys[5] in ("q", "k", "v", "o"):
                        proj = keys[5]
                        state[block_type]["block"].blocks[block_idx].block.layer[
                            0
                        ].attention.__dict__[proj].kernel.value = tensor.T
                    elif keys[5] == "relative_attention_bias":
                        if hasattr(
                            state[block_type]["block"].blocks[block_idx].block.layer[0].attention,
                            "relative_attention_bias",
                        ):
                            state[block_type]["block"].blocks[block_idx].block.layer[
                                0
                            ].attention.relative_attention_bias.embedding.value = tensor
                    continue

                # CrossAttention (decoder only, layer[1])
                if layer_type == "EncDecAttention":
                    # CrossAttention is layer[1] in decoder block
                    if keys[5] in ("q", "k", "v", "o"):
                        proj = keys[5]
                        state[block_type]["block"].blocks[block_idx].block.layer[
                            1
                        ].attention.__dict__[proj].kernel.value = tensor.T
                    continue

                # DenseReluDense (MLP)
                if layer_type == "DenseReluDense":
                    if keys[5] == "wi":
                        state[block_type]["block"].blocks[block_idx].block.layer[
                            -1
                        ].input_dense.kernel.value = tensor.T
                    elif keys[5] == "wo":
                        state[block_type]["block"].blocks[block_idx].block.layer[
                            -1
                        ].output_dense.kernel.value = tensor.T
                    continue

                if layer_type == "layer_norm":
                    ln_idx = int(keys[3])  # 0 for attn, 1 for mlp
                    state[block_type]["block"].blocks[block_idx].block.layer[
                        ln_idx
                    ].layer_norm.weight.value = tensor
                    continue

            if keys[0] == "encoder" and keys[1] == "final_layer_norm":
                state["encoder"]["final_layer_norm"].weight.value = tensor
                continue
            if keys[0] == "decoder" and keys[1] == "final_layer_norm":
                state["decoder"]["final_layer_norm"].weight.value = tensor
                continue
