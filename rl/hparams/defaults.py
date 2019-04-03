import tensorflow as tf

from .registry import register
from .utils import HParams


@register
def default():
  return HParams(
      model=None,
      sys=None,
      env=None,
      agent=None,
      reward_augmentation=None,
      output_dir=None,
      memory="Memory",
      batch_size=64,
      n_steps=1,
      seed=1234,
      state_latent_size=256,
      state_processor="CNN",
      test_interval=5000,
      global_step=0,  # global train steps
      total_step=0  # global train and test steps
  )
