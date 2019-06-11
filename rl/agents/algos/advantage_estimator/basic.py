import tensorflow as tf

from .registry import register
from ..utils import compute_discounted_rewards


@register
def discounted_reward(rewards, values, dones, hparams):
  """ Advantage estimation based on discounted reward

  References:
     https://arxiv.org/abs/1602.01783
  """
  assert len(values) == len(rewards) + 1
  discounted_rewards = compute_discounted_rewards(
      rewards, dones, hparams.gamma, last_value=values[-1]
  )
  return discounted_rewards - values[:-1]


@register
def gae(rewards, values, dones, hparams):
  """ Generalized Advantage Estimation

  References:
      https://arxiv.org/pdf/1506.02438.pdf
  """
  td_deltas = rewards + hparams.gamma * values[1:] * ~dones - values[:-1]
  return compute_discounted_rewards(
      td_deltas, dones, hparams.lambda_ * hparams.gamma
  )
