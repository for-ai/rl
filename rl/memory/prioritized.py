import numpy as np
from collections import namedtuple

from .memory import Memory, TransitionBatch
from .registry import register


class SumTree:
  """ Sum-Tree Data Structure
  Stores arbitrary scalars in its leaves, with inner nodes storing the sum of 
  their direct children's values. Thus, the root of the tree stores the sum 
  of all the weights of the leaves.

  References:
  https://github.com/jaromiru/AI-blog/blob/master/SumTree.py
  """

  def __init__(self, capacity):
    self.tree_leaves = 2
    while self.tree_leaves < capacity:
      # Tree structure requires number of leaves to be a power of 2
      self.tree_leaves **= 2
    self.capacity = capacity
    self.priority_tree = np.zeros(self.tree_leaves * 2 - 1)
    self.data = np.zeros(capacity, dtype=object)
    self.ptr = 0  # pointer to next data index to place entry into
    self.size = 0  # number of elements stored in structure

  def __len__(self):
    return self.size

  @property
  def sum(self):
    return self.priority_tree[0]

  @property
  def max(self):
    return max(self.priority_tree[self.tree_leaves - 1:self.tree_leaves +
                                  self.size - 1])

  @property
  def min(self):
    return min(self.priority_tree[self.tree_leaves - 1:self.tree_leaves +
                                  self.size - 1])

  @property
  def leaves(self):
    return self.priority_tree[self.tree_leaves - 1:self.tree_leaves - 1 +
                              self.size]

  def add(self, entry, priority):
    self.data[self.ptr] = entry
    self.update(self.ptr, priority)
    self.size = min(self.size + 1, self.capacity)
    self.ptr = (self.ptr + 1) % self.capacity

  def update(self, data_index, priority):
    tree_index = data_index + self.tree_leaves - 1
    change = priority - self.priority_tree[tree_index]
    while tree_index >= 0:
      self.priority_tree[tree_index] += change
      tree_index = (tree_index - 1) // 2

  def get(self, priority):
    tree_index = 0

    while True:
      left = 2 * tree_index + 1
      right = left + 1
      if left >= len(self.priority_tree):
        index_weight = self.priority_tree[tree_index]
        data_index = tree_index - self.tree_leaves + 1
        return data_index, index_weight, self.data[data_index]
      left_priority = self.priority_tree[left]
      if priority <= left_priority:
        tree_index = left
      else:
        priority -= left_priority
        tree_index = right

  def clear(self):
    self.priority_tree = np.zeros(self.tree_leaves * 2 - 1)
    self.data = np.zeros(self.capacity, dtype=object)
    self.ptr = 0
    self.size = 0


@register("prioritized")
class PrioritizedMemory(Memory):
  """ Proportional Prioritized Experience Replay
  Data structure which allows for storage of arbitrary data, and weighted 
  recall of these entries according to corresponding priorities. Priorities 
  can be any positive scalar value, but in the RL context would be something 
  like a TD-error for a transition.

  Implementation is backed by a Sum Tree. Each entry's priority is stored in 
  this data structure, and to sample from these we draw uniformly from 
  [0, SumTree.sum], and retrieve the entry who's mass our priority falls into.

  Importance Sampling (IS) is used to weight the updates for each sample drawn 
  from memory. This is necessary since the prioritized memory is not the same 
  distribution of events as seen in the environment. "The estimation of the 
  expected value with stochastic updates relies on those updates corresponding 
  to the same distribution as its expectation"

  References:
  https://arxiv.org/pdf/1511.05952.pdf 
  https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
  """

  def __init__(self, hparams, worker_id=0):
    """
    Args:
      capacity: Maximum capacity of the memory
      priority_control: Scalar applied as exponent to priorities. alpha in 
                        paper 0 corresponds to the uniform case (regular memory)
      priority_compensation: Scalar applied as exponent to IS weights. beta in 
                        paper. 1 fully compensates for non-uniform sampling 
                        probabilities
    """
    assert 0 <= hparams.memory_priority_control <= 1
    assert 0 <= hparams.memory_priority_compensation <= 1

    super(PrioritizedMemory, self).__init__(hparams, worker_id)

    self.tree = SumTree(self._hparams.memory_size)
    # small value used to ensure priorities are greater than 0
    self.eps = 1e-10

  def add_sample(self, **kwargs):
    transition = TransitionBatch(**kwargs)
    priority = self.eps + (1 if self.tree.size == 0 else self.tree.max)
    self.tree.add(transition, priority)

  def __len__(self):
    return len(self.tree)

  def sample(self, batch_size):
    priorities = self.tree.sum * np.random.rand(batch_size)
    samples, indices, weights = [], [], []
    for p in priorities:
      index, weight, sample = self.tree.get(p)
      indices.append(index)
      weights.append(weight)
      samples.append(sample)

    indices = np.array(indices)
    # normalize importance sampling weights, to only scale gradient update downwards
    max_weight = ((self.tree.sum / self.tree.min) /
                  self.tree.size)**self._hparams.memory_priority_compensation
    weights = (((self.tree.sum / np.array(weights)) / self.tree.size)**
               self._hparams.memory_priority_compensation) / max_weight

    first_sample = samples[0]
    fields = {
        field: np.array([getattr(sample, field) for sample in samples])
        for field in first_sample._fields
        if getattr(first_sample, field) is not None
    }
    fields["index"] = indices
    fields["weight"] = weights
    return TransitionBatch(**fields)

  def update(self, indices, priorities):
    """ Update sample priorities """
    for i, p in zip(indices, priorities):
      p = self.eps + p
      self.tree.update(i, p**self._hparams.memory_priority_control)

  def clear(self):
    self.tree.clear()

  def size(self):
    return self.tree.size

  def get_sequence(self, name, indices=None):
    if indices is None:
      indices = range(self.size())
    return np.array([getattr(self.tree.data[index], name) for index in indices])

  def set_sequence(self, name, values, indices=None):
    if indices is None:
      indices = range(self.size())
    for (index, value) in zip(indices, values):
      self.tree.data[index] = self.tree.data[index]._replace(**{name: value})
