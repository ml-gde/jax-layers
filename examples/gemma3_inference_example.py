"""Example script demonstrating Gemma3 model usage."""

import jax.numpy as jnp
from flax import nnx
from transformers import AutoTokenizer  # type: ignore

from jaxgarden.models.gemma3 import Gemma3Config, Gemma3ForCausalLM


def main():
    """Run Gemma3 inference example."""
    # Initialize model with correct configuration
    print("Initializing model...")
    config = Gemma3Config(
        vocab_size=262_208,
        hidden_size=2304,
        intermediate_size=9216,
        num_hidden_layers=26,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=256,
        rope_theta=1_000_000.0,
        rope_local_base_freq=10_000.0,
        max_position_embeddings=131_072,
        sliding_window=4096,
        sliding_window_pattern=6,
        hidden_activation="gelu_pytorch_tanh",
        # Optional RoPE scaling for longer sequences
        rope_scaling={
            "rope_type": "linear",
            "factor": 2.0,  # Enables processing sequences twice the original length
        },
    )
    rng = nnx.Rngs(0)
    model = Gemma3ForCausalLM(config, rngs=rng)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")

    # Prepare input
    prompt = "Write a short story about a robot learning to paint:"
    print(f"\nPrompt: {prompt}")

    # Tokenize input
    input_ids = tokenizer(prompt, return_tensors="np").input_ids
    input_ids = jnp.array(input_ids)

    # Generate text
    print("\nGenerating response...")
    max_new_tokens = 100
    eos_token_id = config.eos_token_id

    # Initialize cache for faster generation
    cache = None
    generated = input_ids

    for _ in range(max_new_tokens):
        # Get logits and updated cache
        logits, cache = model(
            generated,
            use_cache=True,  # Enable KV caching
            deterministic=True,  # No dropout during inference
        )
        # Get next token (use argmax for simplicity)
        next_token = jnp.argmax(logits[:, -1, :], axis=-1)
        # Check if we hit the end of sequence
        if next_token[0] == eos_token_id:
            break
        # Append next token
        generated = jnp.concatenate([generated, next_token[:, None]], axis=1)

    # Decode generated text
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"\nGenerated text:\n{generated_text}")


if __name__ == "__main__":
    main()
