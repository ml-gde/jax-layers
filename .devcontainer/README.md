# Development Container for JAX Layers

This directory contains configuration files for setting up a development container with JAX and CUDA support, which is especially useful for Windows users where JAX doesn't natively support CUDA.

## Prerequisites

To use this development container, you need:

1. [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and configured with WSL 2 backend
2. [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed
3. [Visual Studio Code](https://code.visualstudio.com/) with the [Remote - Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension

## GPU Support

The container is configured to use all available GPUs. Make sure your NVIDIA drivers are up-to-date and that Docker has access to your GPUs.

## Usage

1. Open the project in Visual Studio Code
2. Click on the green icon in the bottom-left corner of VS Code
3. Select "Reopen in Container" from the menu
4. Wait for the container to build and start (this may take a while the first time)

Once the container is running, you'll have a fully configured development environment with:

- Python 3.10
- CUDA 12.2 with cuDNN 9
- JAX with CUDA support
- All dependencies from pyproject.toml

## Dependency Management

The container installs dependencies directly from your project's `pyproject.toml` file using the `pip install -e '.[dev]'` command, ensuring consistency between your development environment and the container.

## Customization

You can customize the container by modifying:

- `devcontainer.json`: VS Code settings, extensions, and container configuration
- `Dockerfile`: Base image, dependencies, and environment setup

## Troubleshooting

If you encounter issues with GPU access:

1. Verify that Docker Desktop is configured to use WSL 2
2. Check that NVIDIA Container Toolkit is properly installed
3. Ensure your NVIDIA drivers are up-to-date
4. Run `nvidia-smi` in WSL to verify GPU access
5. Check Docker logs for any error messages related to GPU access
