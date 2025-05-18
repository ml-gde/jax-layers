from dataclasses import dataclass

import flax.nnx as nnx
import jax.numpy as jnp

from jaxgarden.models.base import BaseConfig


@dataclass
class T5Config(BaseConfig):
    dim: int = 1024
    intermediate_dim: int = 4096
    initializer_factor: float = 1.0
    dropout_rate: float = 0.1
    layer_norm_epsilon: float = 1e-6
    dtype: jnp.dtype = jnp.float32


class T5LayerNorm(nnx.Module):
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
        self.weight = nnx.initializers.ones(rngs.params(), (dim,), dtype=dtype)

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
