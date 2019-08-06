from rl.memory.simple import SimpleMemory
from rl.hparams.utils import HParams

import numpy as np
import random
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)


class SimpleMemoryTest(tf.test.TestCase):

  def setUp(self):
    self._memory = self.get_empty_memory()

  def get_empty_memory(self):
    hparams = HParams()
    hparams.memory_size = 100
    return SimpleMemory(hparams, 0)

  def test_samples_from_empty_memory(self):
    memory = self.get_empty_memory()
    with self.assertRaises(AssertionError,
                           msg="Cannot sample from empty memory."):
      memory.sample(2)

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

  def test_shuffled_batches_from_added_transition(self):
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
    for batch in memory.shuffled_batches(2):
      self.assertAllEqual(batch.reward, np.zeros(2))

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

  def test_unique_occurence_of_transitions_in_shuffled_batches(self):
    memory = self.get_empty_memory()
    for i in range(4):
      observation = {
          'last_state': np.zeros(2),
          'action': np.zeros(2),
          'reward': 0,
          'discount': 0.99 * i,
          'done': False,
          'state': np.zeros(2)
      }
      memory.add_sample(**observation)
    unique_transitions = []
    for batch in memory.shuffled_batches(1):
      self.assertTrue(batch.discount not in unique_transitions)
      unique_transitions.append(batch.discount)


if __name__ == '__main__':
  tf.test.main()
