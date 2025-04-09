import os
import shutil

import jax
import numpy as np
from flax import nnx

from jax_layers.models.base import BaseConfig, BaseModel

class TwoLayerMLPConfig(BaseConfig):
  dim: int = 4
  use_bias: bool = False

class TwoLayerMLP(BaseModel):
  def __init__(self, config, rngs: nnx.Rngs):
    self.linear1 = nnx.Linear(config.dim, config.dim, rngs=rngs, use_bias=config.use_bias)
    self.linear2 = nnx.Linear(config.dim, config.dim, rngs=rngs, use_bias=config.use_bias)

  def __call__(self, x):
    x = self.linear1(x)
    return self.linear2(x)

def test_save_and_load():
  ckpt_dir = "/tmp/jaxgarden_test_ckpt"
  if os.path.exists(ckpt_dir):
    shutil.rmtree(ckpt_dir)

    config = TwoLayerMLPConfig()
  model = TwoLayerMLP(config, rngs=nnx.Rngs(0))
  x = jax.random.normal(jax.random.key(42), (3, 4))
  assert model(x).shape == (3, 4)
  state = model.state
  model.save(ckpt_dir)
  model_restored = TwoLayerMLP(config, rngs=nnx.Rngs(1)).load(ckpt_dir)
  state_restored = model_restored.state
  jax.tree.map(np.testing.assert_array_equal, state, state_restored)
