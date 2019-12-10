from rl.memory.prioritized import PrioritizedMemory
from rl.hparams.utils import HParams

import numpy as np
import random
import tensorflow as tf


class PrioritizedMemoryTest(tf.test.TestCase):

  def setUp(self):
    self._memory = self.get_empty_memory()

  def get_empty_memory(self):
    hparams = HParams()
    hparams.memory_size = 100
    hparams.memory_priority_control = 1
    hparams.memory_priority_compensation = 1
    return PrioritizedMemory(hparams, 0)

  def test_setup_with_invalid_params(self):
    hparams = HParams()
    hparams.memory_size = 100
    # memory_priority_control and memory_priority_compensation should be <= 1
    hparams.memory_priority_control = 1.75
    hparams.memory_priority_compensation = 0.8
    with self.assertRaises(AssertionError):
      memory = PrioritizedMemory(hparams, 0)

    hparams.memory_priority_control = 1.75
    hparams.memory_priority_compensation = 1.8
    with self.assertRaises(AssertionError):
      memory = PrioritizedMemory(hparams, 0)

  def test_get_sequence_from_added_transition(self):
    memory = self.get_empty_memory()
    for _ in range(2):
      observation = {
          'last_state': np.zeros(2),
          'action': np.zeros(2),
          'reward': 0,
          'discount': 0,
          'done': False,
          'state': np.zeros(2)
      }
      memory.add_sample(**observation)
    rewards = memory.get_sequence('reward')
    self.assertAllEqual(rewards, np.zeros(2))

  def test_sample_from_added_transition(self):
    memory = self.get_empty_memory()
    for _ in range(2):
      observation = {
          'last_state': np.zeros(2),
          'action': np.zeros(2),
          'reward': 0,
          'discount': 0,
          'done': False,
          'state': np.zeros(2)
      }
      memory.add_sample(**observation)
    sample = memory.sample(2)
    self.assertAllEqual(sample.reward, np.zeros(2))

  def test_set_sequence_on_added_transitions(self):
    memory = self.get_empty_memory()
    for _ in range(4):
      observation = {
          'last_state': np.zeros(2),
          'action': np.zeros(2),
          'reward': 0,
          'discount': 0,
          'done': False,
          'state': np.zeros(2)
      }
      memory.add_sample(**observation)
    memory.set_sequence('reward', np.ones(4))
    rewards = memory.get_sequence('reward')
    self.assertAllEqual(rewards, np.ones(4))

  def test_update_priorities_and_sample(self):
    memory = self.get_empty_memory()
    for i in range(2):
      observation = {
          'last_state': np.zeros(2),
          'action': np.zeros(2),
          'reward': i + 1,
          'discount': 0.99**i,
          'done': False,
          'state': np.zeros(2)
      }
      memory.add_sample(**observation)
    indices = [1, 0]
    weights = np.array([1, 0])
    memory.update(indices, weights)
    batch = memory.sample(1)
    self.assertEqual(batch.index, 1)


if __name__ == '__main__':
  tf.test.main()
