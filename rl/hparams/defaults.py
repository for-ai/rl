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
      lr={"lr": 0.001},
      lr_decay={"lr": "no_decay"},
      output_dir=None,
      memory="simple",
      batch_size=64,
      seed=1234,
      state_latent_size=256,
      state_processor="CNN",
      eval_interval=5000,
      global_step=0,  # global train steps
      total_step=0,  # global train and test steps
      atari_registry=False  # indicate if Atari Baselines env have been added
  )
