import random
import numpy as np
import tensorflow as tf

from .registry import register
from ....utils.utils import ModeKeys


@register
def random_action(hparams, distribution, worker_id=0):
  return random.randint(0, hparams.num_actions - 1)


@register
def epsilon_action(hparams, distribution, worker_id=0):
  # perform random action during training only
  if (hparams.mode[worker_id] == ModeKeys.TRAIN and
      random.random() < hparams.epsilon[worker_id]):
    return random_action(hparams, distribution)
  else:
    return max_action(hparams, distribution)


@register
def max_action(hparams, distribution, worker_id=0):
  return np.argmax(distribution)


@register
def non_uniform_random_action(hparams, distribution, worker_id=0):
  return np.random.choice(range(hparams.num_actions), p=distribution.ravel())


@register
def uniform_random_action(hparams, distribution, worker_id=0):
  if hparams.mode[worker_id] == ModeKeys.TRAIN:
    h = np.random.uniform(size=distribution.shape)
    return np.argmax(distribution - np.log(-np.log(h)))
  else:
    return max_action(hparams, distribution, worker_id=0)


@register
def normal_noise_action(hparams, action, worker_id=0):
  return np.clip(
      np.random.normal(action, hparams.variance), hparams.action_low,
      hparams.action_high)
