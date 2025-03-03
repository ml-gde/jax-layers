"""Example demonstrating the use of MultiHeadAttention with different implementations."""

import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from jax_layers.attention import MultiHeadAttention


def benchmark_attention(implementation=None, batch_size=2, seq_len=1024, num_heads=8, head_dim=64):
    """Benchmark MultiHeadAttention with different implementations."""
    print(f"\nBenchmarking MultiHeadAttention with implementation={implementation}")
    print(f"Input shape (b, s, h, d) = ({batch_size}, {seq_len}, {num_heads}, {head_dim})")

    # Create random input data
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    x = jax.random.normal(key1, (batch_size, seq_len, num_heads * head_dim))

    # Create a causal attention mask
    mask = jnp.tril(jnp.ones((batch_size, 1, seq_len, seq_len)))

    # Create the MultiHeadAttention module
    attention = MultiHeadAttention(
        num_heads=num_heads,
        in_features=num_heads * head_dim,
        implementation=implementation,
        rngs=nnx.Rngs(key2),
    )

    # Compile the forward pass
    @nnx.jit
    def forward(x, mask):
        return attention(x, mask=mask)

    # Warm-up
    _ = forward(x, mask)

    # Benchmark
    start_time = time.time()
    num_runs = 10
    for _ in range(num_runs):
        output = forward(x, mask)
        output.block_until_ready()  # Ensure computation is complete
    end_time = time.time()

    avg_time = (end_time - start_time) / num_runs
    print(f"Average time per run: {avg_time:.6f} seconds")

    return output, avg_time


def compare_implementations():
    """Compare different implementations of MultiHeadAttention."""
    print("Comparing different implementations of MultiHeadAttention")

    # Parameters for the comparison
    batch_size = 2
    seq_len = 1024
    num_heads = 8
    head_dim = 64

    # Benchmark each implementation
    implementations = [None, "xla", "cudnn"]
    outputs = {}
    times = {}

    for impl in implementations:
        try:
            output, avg_time = benchmark_attention(
                implementation=impl,
                batch_size=batch_size,
                seq_len=seq_len,
                num_heads=num_heads,
                head_dim=head_dim,
            )
            outputs[impl] = output
            times[impl] = avg_time
        except Exception as e:
            print(f"Error with implementation {impl}: {e}")

    # Compare outputs for correctness
    print("\nComparing outputs for correctness:")
    for impl1 in outputs:
        for impl2 in outputs:
            if impl1 != impl2:
                max_diff = jnp.max(jnp.abs(outputs[impl1] - outputs[impl2]))
                print(f"Max difference between {impl1} and {impl2}: {max_diff}")

    # Compare performance
    if len(times) > 1:
        print("\nPerformance comparison:")
        baseline = times[None]  # Default implementation as baseline
        for impl, avg_time in times.items():
            if impl is not None:
                speedup = baseline / avg_time
                print(f"{impl} implementation: {speedup:.2f}x speedup over default")


def main():
    """Run the example."""
    # Enable JAX logging to see which implementation is being used
    jax.config.update("jax_log_compiles", True)

    # Compare different implementations
    compare_implementations()

    # Show an example of using MultiHeadAttention with a specific implementation
    print("\nExample of using MultiHeadAttention with Flash Attention:")
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)

    # Create input data
    batch_size = 2
    seq_len = 128
    num_heads = 8
    head_dim = 64
    hidden_dim = num_heads * head_dim

    x = jax.random.normal(key1, (batch_size, seq_len, hidden_dim))

    # Create a causal attention mask
    mask = jnp.tril(jnp.ones((batch_size, 1, seq_len, seq_len)))

    # Create the MultiHeadAttention module with Flash Attention
    attention = MultiHeadAttention(
        num_heads=num_heads,
        in_features=hidden_dim,
        implementation="flash",  # Use Flash Attention (alias for "cudnn")
        rngs=nnx.Rngs(key2),
    )

    # Apply the attention
    output = attention(x, mask=mask)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output mean: {jnp.mean(output)}")
    print(f"Output std: {jnp.std(output)}")


if __name__ == "__main__":
    main()
