import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp  # type: ignore
from flax import nnx
from huggingface_hub import snapshot_download
from safetensors import safe_open

DEFAULT_PARAMS_FILE = "jaxgarden_state"

@dataclass
class BaseConfig:
    """Base configuration for all the models implemented in the JAXgarden library.

    Each model implemented in JAXgarden should subclass this class for configuration management.
    """

    seed: int = 42
    log_level: str = "info"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__

    def update(self, **kwargs: dict) -> None:
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                self.extra[k] = v

class BaseModel(nnx.Module):
    """Base class for all the models implemented in the JAXgarden library."""

    def __init__(self,
                 config: BaseConfig,
                 *,
                 dtype: jnp.dtype | None = None,
                 param_dtype: jnp.dtype = jnp.float32,
                 precision: jax.lax.Precision | str | None = None,
                 rngs: nnx.Rngs):
        """Initialize the model.

                Args:
            config: config class for this model.
            dtype: Data type in which computation is performed.
param_dtype: Data type in which params are stored.
            precision: Numerical precision.
            rngs: Random number generators for param initialization etc.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs

    @property
    def state(self) -> dict[str, jnp.ndarray]:
        """Splits state from the graph and returns it.

        It can be used for serialization with orbax."""
        state = nnx.split(self, nnx.Param, ...)[1]
        pure_dict_state = nnx.to_pure_dict(state)
        return pure_dict_state

    def save(self, path: str) -> None:
        """Saves the model state to a directory.

        Args:
            path: The directory path to save the model state to.
        """
        state = self.state
        checkpointer = ocp.StandardCheckpointer()
        checkpointer.save(os.path.join(path, DEFAULT_PARAMS_FILE), state)
        checkpointer.wait_until_finished()

    def load(self, path: str) -> nnx.Module:
        """Loads the model state from a directory.

        Args:
            path: The directory path to load the model state from.
        """
        checkpointer = ocp.StandardCheckpointer()
        restored_pure_dict = checkpointer.restore(os.path.join(path, DEFAULT_PARAMS_FILE))
        abstract_model = nnx.eval_shape(lambda: self)
        graphdef, abstract_state = nnx.split(abstract_model)
        nnx.replace_by_pure_dict(abstract_state, restored_pure_dict)
        return nnx.merge(graphdef, abstract_state)

    @staticmethod
    def download_from_hf(repo_id: str, local_dir: str) -> None:
        """Downloads the model from the Hugging Face Hub.

        Args:
            repo_id: The repository ID of the model to download.
            local_dir: The local directory to save the model to.
        """
        snapshot_download(repo_id, local_dir=local_dir)

    @staticmethod
    def load_safetensors(path_to_model_weights: str) -> Iterator[tuple[Any, Any]]:
        """Helper function to lazily load params from safetensors file.

        Use this static method to load weights for conversion tasks.

        Args:
            model_path_to_params: Path to directory containing .safetensors files."""
        if not os.path.isdir(path_to_model_weights):
            raise ValueError(f"{path_to_model_weights} is not a valid directory.")

        safetensors_files = Path(path_to_model_weights).glob('*.safetensors')

        for file in safetensors_files:
            with safe_open(file, framework="jax", device="cpu") as f:
                for key in f:
                    yield (key, f.get_tensor(key))
