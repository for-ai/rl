import random
import numpy as np

from .registry import register


@register
class Memory:
  """ Simple memory """

  def __init__(self, hparams, worker_id):
    self.worker_id = worker_id
    self.capacity = hparams.memory_size
    self.memory = []

  def add_sample(self, last_state, action, reward, discount, done, state):
    if self.size() > self.capacity:
      self.memory.pop(0)
    self.memory.append((last_state, action, reward, discount, done, state))

  def sample(self, batch_size=None):
    sample_size = batch_size if batch_size <= self.size() else self.size()
    samples = random.sample(self.memory, sample_size)
    indices = np.zeros(len(samples))
    weights = np.ones(len(samples))
    last_states = np.array([s[0] for s in samples])
    actions = np.array([s[1] for s in samples])
    rewards = np.array([s[2] for s in samples])
    done = np.array([s[4] for s in samples])
    states = np.array([s[5] for s in samples])
    return indices, weights, last_states, actions, rewards, done, states

  def size(self):
    return len(self.memory)

  def clear(self):
    self.memory = []
