"""Training utilities for NanoGPT."""

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from typing import Any, Dict, Optional, Tuple

from .model import GPT

class TrainState(train_state.TrainState):
    """Training state for the GPT model."""
    dropout_rng: jax.random.PRNGKey
    key: jax.random.PRNGKey

def create_train_state(
    model: GPT,
    learning_rate: float,
    dropout_rng: jax.random.PRNGKey,
    key: jax.random.PRNGKey,
) -> TrainState:
    """Create initial training state."""
    params = model.init(key, jnp.ones((1, 1), dtype=jnp.int32))['params']
    tx = optax.adamw(learning_rate=learning_rate)
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        dropout_rng=dropout_rng,
        key=key,
    )

def train_step(
    state: TrainState,
    batch: Tuple[jnp.ndarray, jnp.ndarray],
) -> Tuple[TrainState, Dict[str, float]]:
    """Perform a single training step."""
    inputs, targets = batch
    
    def loss_fn(params):
        logits = state.apply_fn(
            {'params': params},
            inputs,
            rngs={'dropout': state.dropout_rng},
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
        return loss, logits
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    
    # Update state
    state = state.apply_gradients(grads=grads)
    
    # Generate new dropout rng
    dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)
    state = state.replace(dropout_rng=new_dropout_rng)
    
    # Compute metrics
    accuracy = (jnp.argmax(logits, axis=-1) == targets).mean()
    
    return state, {
        'loss': loss,
        'accuracy': accuracy,
    }

def eval_step(
    state: TrainState,
    batch: Tuple[jnp.ndarray, jnp.ndarray],
) -> Dict[str, float]:
    """Perform a single evaluation step."""
    inputs, targets = batch
    
    logits = state.apply_fn(
        {'params': state.params},
        inputs,
        rngs={'dropout': jax.random.PRNGKey(0)},  # Use fixed rng for eval
    )
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
    accuracy = (jnp.argmax(logits, axis=-1) == targets).mean()
    
    return {
        'loss': loss,
        'accuracy': accuracy,
    }

def create_learning_rate_schedule(
    learning_rate: float,
    warmup_steps: int,
    total_steps: int,
) -> optax.Schedule:
    """Create a learning rate schedule with warmup."""
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps - warmup_steps,
        end_value=0.0,
    ) 