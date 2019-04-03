import numpy as np
from collections import namedtuple

from .memory import Memory
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


@register
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

    self.worker_id = worker_id
    self.capacity = hparams.memory_size
    # corresponds to alpha in paper
    self.priority_control = hparams.memory_priority_control
    # beta in paper
    self.priority_compensation = hparams.memory_priority_compensation
    self.tree = SumTree(self.capacity)
    # small value used to ensure priorities are greater than 0
    self.eps = 1e-10
    self.MemoryEntry = namedtuple(
        "MemoryEntry",
        ["last_state", "action", "reward", "discount", "done", "state"])

  def add_sample(self, last_state, action, reward, discount, done, state):
    memory_entry = self.MemoryEntry(last_state, action, reward, discount, done,
                                    state)
    priority = self.eps + (1 if self.tree.size == 0 else self.tree.max)
    self.tree.add(memory_entry, priority)

  def __len__(self):
    return len(self.tree)

  def sample(self, batch_size=None):
    if batch_size == None:
      indices = list(range(self.tree.size))
      samples = self.tree.data[:self.tree.size]
      weights = self.tree.leaves
    else:
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
                  self.tree.size)**self.priority_compensation
    weights = (((self.tree.sum / np.array(weights)) / self.tree.size)**
               self.priority_compensation) / max_weight

    last_states = np.array([s.last_state for s in samples])
    actions = np.array([s.action for s in samples])
    rewards = np.array([s.reward for s in samples])
    done = np.array([s.done for s in samples])
    states = np.array([s.state for s in samples])

    return indices, weights, last_states, actions, rewards, done, states

  def update(self, indices, priorities):
    """ Update sample priorities """
    for i, p in zip(indices, priorities):
      p = self.eps + p
      self.tree.update(i, p**self.priority_control)

  def clear(self):
    self.tree.clear()

  def size(self):
    return self.tree.size
