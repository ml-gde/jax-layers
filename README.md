# JAX Layers

[![Doc](https://github.com/ml-gde/jax-layers/actions/workflows/docs.yml/badge.svg)](https://github.com/ml-gde/jax-layers/actions/workflows/docs.yml)
[![Tests](https://github.com/ml-gde/jax-layers/actions/workflows/tests.yml/badge.svg)](https://github.com/ml-gde/jax-layers/actions/workflows/tests.yml)

![Logo](./assets/logo.jpg)

A reusable collection of high-performance neural network layers and models for JAX, aiming to match and exceed the capabilities available in the PyTorch ecosystem.

## Motivation

JAX Layers was created to provide the JAX ecosystem with a comprehensive library of well-documented, thoroughly tested, and numerically accurate implementations of neural network layers and models. The project aims to:

- Provide both functional APIs and Flax NNX wrappers for maximum flexibility
- Ensure seamless integration with the broader JAX ecosystem, especially Flax
- Facilitate easy upstreaming of implementations to core libraries
- Maintain rigorous testing and documentation standards
- Match or exceed the performance of equivalent PyTorch implementations

Initially started within the ML GDE group, the project began with a high-performance MultiHeadAttention implementation supporting various attention backends, with plans to expand to more layers and models.

## Features

- **MultiHeadAttention**: A Flax NNX-compatible implementation with support for different attention backends.
  - Supports JAX's native Flash Attention implementation through cuDNN
  - Seamlessly integrates with Flax NNX's module system
  - Provides a simple interface for switching between attention implementations

## Installation

```bash
# Install from source
git clone https://github.com/ml-gde/jax-layers.git
cd jax-layers
pip install -e .
```

## Usage

### MultiHeadAttention Module (Flax NNX)

```python
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from jax_layers.attention import MultiHeadAttention

# Create a MultiHeadAttention module with Flash Attention support
attention = MultiHeadAttention(
    num_heads=8,
    in_features=512,
    implementation="cudnn",  # Use cuDNN's Flash Attention if available
    rngs=nnx.Rngs(0),
)

# Create input data
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (2, 128, 512))  # (batch, seq_length, hidden_dim)

# Create a causal attention mask
mask = jnp.tril(jnp.ones((2, 1, 128, 128)))  # (batch, 1, q_len, kv_len)

# Apply the model
output = attention(x, mask=mask)
```

### Functional API

#### Dot Product Attention with Implementation Selection

```python
import jax
import jax.numpy as jnp
from jax_layers.functional import dot_product_attention

# Create random query, key, value tensors
key = jax.random.PRNGKey(0)
query = jax.random.normal(key, (2, 128, 8, 64))  # (batch, seq_len, heads, head_dim)
key_tensor = jax.random.normal(key, (2, 128, 8, 64))
value = jax.random.normal(key, (2, 128, 8, 64))

# Create a causal attention mask
mask = jnp.tril(jnp.ones((2, 1, 128, 128)))  # (batch, 1, q_len, kv_len)

# Apply dot product attention with Flash Attention implementation
output = dot_product_attention(
    query=query,
    key=key_tensor,
    value=value,
    mask=mask,
    implementation="cudnn",  # Use cuDNN's Flash Attention implementation
)
```

## Development

### Setup

1. Please fork the repository to your account first.
2. Follow the instructions below.

```bash
# Clone the repository
git clone https://github.com/yourusername/jax-layers.git
cd jax-layers

# Install development dependencies
pip install -e ".[dev]"
```

### Pre-commit

This project uses pre-commit hooks to ensure code quality and consistency. Pre-commit automatically runs linting and formatting tools (such as ruff) before each commit, helping to catch issues early.

```bash
# Install Pre-commit Hooks
pre-commit install

# Run Pre-commit on All Files
pre-commit run --all-files
```

Every time you attempt to commit, pre-commit automatically runs the configured hooks (e.g., ruff). If any issues are detected, the commit will be blocked until they are resolved.

### Testing

The project maintains a comprehensive test suite to ensure correctness and numerical accuracy:

```bash
# Run all tests
pytest

# Run tests with coverage
pytest tests/ --cov=jax_layers

# Run specific test file
pytest tests/test_multi_head_attention.py
```

### Code Quality

We maintain high code quality standards through automated checks:

```bash
# Run linting
ruff check .

# Run type checking
mypy jax_layers

# Run tests
pytest
```

### Documentation

Documentation is automatically generated from docstrings:

```bash
# Build documentation
cd docs
make html
```

### Development Container (for Windows users)

Since JAX doesn't support CUDA on Windows natively, we provide a development container configuration:

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) with WSL 2 backend
2. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
3. Install [Visual Studio Code](https://code.visualstudio.com/) with the [Remote - Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension
4. Open the project in VS Code
5. Click the green icon in the bottom-left corner and select "Reopen in Container"

The container provides:

- Python 3.10
- CUDA 12.4 with cuDNN 9
- JAX with CUDA support
- All dependencies from your pyproject.toml

See [.devcontainer/README.md](.devcontainer/README.md) for more details.

## Contributing

Contributions are more than welcome! Whether it's:

- Adding new layer implementations
- Improving documentation
- Adding tests
- Reporting bugs
- Suggesting improvements

Please feel free to open issues and pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Thanks to the JAX and Flax teams for their excellent libraries.
- Special thanks to the ML GDE group for initiating this project.
