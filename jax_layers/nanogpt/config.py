"""Configuration for NanoGPT."""

from dataclasses import dataclass
from typing import Optional

@dataclass
class GPTConfig:
    """GPT model configuration."""
    
    vocab_size: int = 256
    block_size: int = 128
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    dtype: str = 'float32'

@dataclass
class TrainConfig:
    """Training configuration."""
    
    batch_size: int = 64
    learning_rate: float = 3e-4
    max_iters: int = 5000
    eval_interval: int = 500
    eval_iters: int = 200
    warmup_iters: int = 2000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    decay_lr: bool = True
    min_lr: float = 3e-5
    device: str = 'cpu'
    seed: int = 42

@dataclass
class Config:
    """Main configuration."""
    
    model: GPTConfig = GPTConfig()
    train: TrainConfig = TrainConfig()
    out_dir: str = 'out'
    resume: Optional[str] = None 