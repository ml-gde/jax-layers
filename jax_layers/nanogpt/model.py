"""NanoGPT model implementation using JAX and Flax."""

from typing import Optional, Tuple, Any, Dict

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import lax
from jax_layers.attention import MultiHeadAttention

class CausalSelfAttention(nn.Module):
    """Causal self-attention layer."""
    
    n_head: int
    n_embd: int
    dropout: float = 0.1
    dtype: jnp.dtype = jnp.float32
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        # Create causal mask if not provided
        if mask is None:
            B, T = x.shape[:2]
            mask = jnp.tril(jnp.ones((B, 1, T, T)))
        
        # Apply attention
        attn = MultiHeadAttention(
            num_heads=self.n_head,
            in_features=self.n_embd,
            dropout_rate=self.dropout,
            dtype=self.dtype,
            implementation="flash",  # Use Flash Attention if available
        )
        return attn(x, mask=mask)

class Block(nn.Module):
    """Transformer block."""
    
    n_head: int
    n_embd: int
    dropout: float = 0.1
    dtype: jnp.dtype = jnp.float32
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        # attention
        y = nn.LayerNorm(dtype=self.dtype)(x)
        y = CausalSelfAttention(self.n_head, self.n_embd, self.dropout, self.dtype)(y, mask)
        x = x + y
        
        # mlp
        y = nn.LayerNorm(dtype=self.dtype)(x)
        y = nn.Dense(4 * self.n_embd, dtype=self.dtype)(y)
        y = jax.nn.gelu(y)
        y = nn.Dense(self.n_embd, dtype=self.dtype)(y)
        x = x + y
        return x

class GPT(nn.Module):
    """GPT model."""
    
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.1
    dtype: jnp.dtype = jnp.float32
    
    @nn.compact
    def __call__(self, idx: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        B, T = idx.shape
        
        # token and position embeddings
        tok_emb = nn.Embed(self.vocab_size, self.n_embd, dtype=self.dtype)(idx)
        pos = jnp.arange(0, T, dtype=jnp.int32)[None, :]  # shape (1, T)
        pos_emb = nn.Embed(self.block_size, self.n_embd, dtype=self.dtype)(pos)
        x = nn.Dropout(rate=self.dropout)(tok_emb + pos_emb, deterministic=True)
        
        # transformer blocks
        for _ in range(self.n_layer):
            x = Block(self.n_head, self.n_embd, self.dropout, self.dtype)(x, mask)
        
        # final layer norm
        x = nn.LayerNorm(dtype=self.dtype)(x)
        
        # language modeling head
        logits = nn.Dense(self.vocab_size, dtype=self.dtype)(x)
        return logits
    
    def generate(self, idx: jnp.ndarray, max_new_tokens: int, temperature: float = 1.0) -> jnp.ndarray:
        """Generate new tokens given a sequence."""
        for _ in range(max_new_tokens):
            # crop context if needed
            idx_cond = idx if idx.shape[1] <= self.block_size else idx[:, -self.block_size:]
            
            # get predictions
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # sample from the distribution
            probs = jax.nn.softmax(logits, axis=-1)
            idx_next = jax.random.categorical(jax.random.PRNGKey(0), probs)
            
            # append sampled index to the sequence
            idx = jnp.concatenate((idx, idx_next[:, None]), axis=1)
        
        return idx 