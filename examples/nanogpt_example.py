"""Example script demonstrating NanoGPT usage."""

import os
import time
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from tqdm import tqdm

from jax_layers.nanogpt.config import Config
from jax_layers.nanogpt.data import create_dataset, get_shakespeare
from jax_layers.nanogpt.model import GPT
from jax_layers.nanogpt.train import (
    TrainState,
    create_train_state,
    create_learning_rate_schedule,
    eval_step,
    train_step,
)

def estimate_loss(
    state: TrainState,
    eval_data: Tuple[jnp.ndarray, jnp.ndarray],
    eval_iters: int,
) -> Dict[str, float]:
    """Estimate loss on evaluation data."""
    out = {}
    for k in range(eval_iters):
        batch = eval_data[0][k], eval_data[1][k]
        metrics = eval_step(state, batch)
        for k, v in metrics.items():
            out[k] = out.get(k, 0.0) + v
    for k in out:
        out[k] /= eval_iters
    return out

def main():
    # Initialize configuration
    config = Config()
    
    # Set random seed
    jax.random.PRNGKey(config.train.seed)
    
    # Create output directory
    os.makedirs(config.out_dir, exist_ok=True)
    
    # Load and prepare data
    print('Loading Shakespeare dataset...')
    text = get_shakespeare()
    train_data, val_data = create_dataset(
        text=text,
        block_size=config.model.block_size,
        batch_size=config.train.batch_size,
    )
    
    # Initialize model and training state
    print('Initializing model...')
    model = GPT(
        vocab_size=config.model.vocab_size,
        block_size=config.model.block_size,
        n_layer=config.model.n_layer,
        n_head=config.model.n_head,
        n_embd=config.model.n_embd,
        dropout=config.model.dropout,
        dtype=getattr(jnp, config.model.dtype),
    )
    
    # Create learning rate schedule
    lr_schedule = create_learning_rate_schedule(
        learning_rate=config.train.learning_rate,
        warmup_steps=config.train.warmup_iters,
        total_steps=config.train.max_iters,
    )
    
    # Initialize training state
    key = jax.random.PRNGKey(config.train.seed)
    dropout_rng, key = jax.random.split(key)
    state = create_train_state(
        model=model,
        learning_rate=lr_schedule,
        dropout_rng=dropout_rng,
        key=key,
    )
    
    # Training loop
    print('Starting training...')
    best_val_loss = float('inf')
    t0 = time.time()
    
    for iter_num in tqdm(range(config.train.max_iters)):
        # Determine and set the learning rate for this iteration
        lr = lr_schedule(iter_num) if config.train.decay_lr else config.train.learning_rate
        state = state.replace(opt_state=state.opt_state.replace(learning_rate=lr))
        
        # Sample a batch of data
        batch = train_data[0][iter_num % len(train_data[0])], train_data[1][iter_num % len(train_data[1])]
        
        # Evaluate the loss on train/val sets
        if iter_num % config.train.eval_interval == 0:
            train_losses = estimate_loss(state, train_data, config.train.eval_iters)
            val_losses = estimate_loss(state, val_data, config.train.eval_iters)
            print(f'iter {iter_num}: train loss {train_losses["loss"]:.4f}, val loss {val_losses["loss"]:.4f}')
            
            # Save best model
            if val_losses['loss'] < best_val_loss:
                best_val_loss = val_losses['loss']
                if iter_num > 0:
                    checkpoint = {'model': state.params}
                    with open(os.path.join(config.out_dir, 'best.ckpt'), 'wb') as f:
                        jax.serialization.save(f, checkpoint)
        
        # Forward backward update
        state, metrics = train_step(state, batch)
        
        # Timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % config.train.eval_interval == 0:
            print(f'iter {iter_num}: loss {metrics["loss"]:.4f}, time {dt*1000:.2f}ms')
    
    print('Training completed!')

if __name__ == '__main__':
    main() 