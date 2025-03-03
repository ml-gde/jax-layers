#!/usr/bin/env python3
"""
Script to verify that JAX with CUDA is working correctly.
Run this script after the container is built to confirm GPU access.
"""

import re
import subprocess
import time

import jax
import jax.numpy as jnp


def get_cuda_version():
    """Get the CUDA version from nvcc."""
    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        version_match = re.search(r"release (\d+\.\d+)", result.stdout)
        if version_match:
            return version_match.group(1)
        return "Unknown"
    except Exception:
        return "Unknown"


def main():
    print("\n" + "=" * 50)
    print("JAX CUDA Verification Script")
    print("=" * 50)

    # Check JAX version
    print(f"JAX version: {jax.__version__}")

    # Check CUDA version
    cuda_version = get_cuda_version()
    print(f"CUDA version: {cuda_version}")

    # Check available devices
    print("\nAvailable devices:")
    for i, device in enumerate(jax.devices()):
        print(f"  Device {i}: {device}")

    # Check if GPU is available
    gpu_available = any(d.platform == "gpu" for d in jax.devices())
    print(f"\nGPU available: {gpu_available}")

    if not gpu_available:
        print("\n⚠️  No GPU devices found! JAX is not using CUDA.")
        print("Please check your installation and GPU configuration.")
        return

    # Run a simple benchmark
    print("\nRunning simple matrix multiplication benchmark...")

    # Create large matrices
    n = 5000
    print(f"Creating {n}x{n} matrices...")

    # CPU benchmark
    with jax.devices("cpu")[0]:
        x_cpu = jnp.ones((n, n))
        y_cpu = jnp.ones((n, n))

        # Warm-up
        _ = jnp.dot(x_cpu, y_cpu)
        jax.block_until_ready(_)

        # Benchmark
        start = time.time()
        result_cpu = jnp.dot(x_cpu, y_cpu)
        jax.block_until_ready(result_cpu)
        cpu_time = time.time() - start

    # GPU benchmark
    with jax.devices("gpu")[0]:
        x_gpu = jnp.ones((n, n))
        y_gpu = jnp.ones((n, n))

        # Warm-up
        _ = jnp.dot(x_gpu, y_gpu)
        jax.block_until_ready(_)

        # Benchmark
        start = time.time()
        result_gpu = jnp.dot(x_gpu, y_gpu)
        jax.block_until_ready(result_gpu)
        gpu_time = time.time() - start

    # Print results
    print(f"\nCPU time: {cpu_time:.4f} seconds")
    print(f"GPU time: {gpu_time:.4f} seconds")
    print(f"Speedup: {cpu_time / gpu_time:.2f}x")

    if cpu_time > gpu_time:
        print("\n✅ GPU is faster than CPU! JAX with CUDA is working correctly.")
    else:
        print("\n⚠️  GPU is not faster than CPU. Something might be wrong with the CUDA setup.")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
