import numpy as np
import tensorflow as tf


def compute_discounted_rewards(rewards, dones, discount, last_value=None):
  if last_value is not None:
    future_cumulative_reward = last_value
  else:
    future_cumulative_reward = 0
  discounted_rewards = np.empty_like(rewards, dtype=np.float32)
  for i in reversed(range(len(rewards))):
    future_cumulative_reward = (
        rewards[i] + discount * future_cumulative_reward * ~dones[i]
    )
    discounted_rewards[i] = future_cumulative_reward
  return discounted_rewards


def normalize(x):
  return (x - x.mean()) / (x.std() + 1e-10)


def one_hot(indices, depth):
  return list(np.eye(depth)[indices])


def copy_variables_op(source, target):

  source_vars = sorted(source.trainable_weights, key=lambda v: v.name)
  target_vars = sorted(target.trainable_weights, key=lambda v: v.name)

  return [
      tf.assign(target_var, source_var)
      for target_var, source_var in zip(target_vars, source_vars)
  ]


class OrnsteinUhlenbeckActionNoise:
  """
  https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
  based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
  """

  def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
    self.theta = theta
    self.mu = mu
    self.sigma = sigma
    self.dt = dt
    self.x0 = x0
    self.reset()

  def __call__(self):
    x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
    self.x_prev = x
    return x

  def reset(self):
    self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

  def __repr__(self):
    return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(
        self.mu, self.sigma)
