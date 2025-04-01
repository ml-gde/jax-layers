"""Data processing utilities for NanoGPT."""

import os
from typing import Dict, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np
import regex as re
from tqdm import tqdm

class Tokenizer:
    """Simple character-level tokenizer."""
    
    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self.chars = [chr(i) for i in range(vocab_size)]
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token ids."""
        return [self.char_to_idx.get(ch, 0) for ch in text]
    
    def decode(self, ids: List[int]) -> str:
        """Decode token ids to text."""
        return ''.join(self.idx_to_char.get(i, '') for i in ids)

def get_shakespeare() -> str:
    """Download and load Shakespeare dataset."""
    input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
    
    if not os.path.exists(input_file_path):
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        import urllib.request
        print(f'Downloading Shakespeare dataset to {input_file_path}...')
        urllib.request.urlretrieve(data_url, input_file_path)
    
    with open(input_file_path, 'r', encoding='utf-8') as f:
        return f.read()

def create_dataset(
    text: str,
    block_size: int,
    batch_size: int,
    split: float = 0.9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create train and validation datasets from text."""
    # Create tokenizer and encode text
    tokenizer = Tokenizer()
    data = np.array(tokenizer.encode(text))
    
    # Split into train and validation sets
    n = int(split * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    def get_batches(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Create input and target sequences
        x = []
        y = []
        for i in range(0, len(data) - block_size):
            x.append(data[i:i + block_size])
            y.append(data[i + 1:i + block_size + 1])
        
        # Stack into batches
        x = np.stack(x)
        y = np.stack(y)
        
        # Shuffle and create batches
        indices = np.random.permutation(len(x))
        x = x[indices]
        y = y[indices]
        
        n_batches = len(x) // batch_size
        x = x[:n_batches * batch_size].reshape(n_batches, batch_size, -1)
        y = y[:n_batches * batch_size].reshape(n_batches, batch_size, -1)
        
        return x, y
    
    train_x, train_y = get_batches(train_data)
    val_x, val_y = get_batches(val_data)
    
    return train_x, train_y, val_x, val_y

def get_batch(
    x: np.ndarray,
    y: np.ndarray,
    batch_idx: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Get a single batch from the dataset."""
    return jnp.array(x[batch_idx]), jnp.array(y[batch_idx]) 