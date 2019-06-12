import random
import numpy as np

from .memory import Memory, TransitionBatch
from .registry import register


@register("simple")
class SimpleMemory(Memory):
  """ Simple memory """

  def __init__(self, hparams, worker_id):
    super(SimpleMemory, self).__init__(hparams, worker_id)
    self._memory = []

  def add_sample(self, **kwargs):
    if self.size() > self._hparams.memory_size:
      self._memory.pop(0)
    self._memory.append(TransitionBatch(**kwargs))

  def sample(self, batch_size):
    sample_size = batch_size if batch_size <= self.size() else self.size()
    assert sample_size > 0, "Cannot sample from empty memory."
    samples = random.sample(self._memory, sample_size)
    return self._make_batch(samples)

  def size(self):
    return len(self._memory)

  def clear(self):
    self._memory = []

  def get_sequence(self, name, indices=None):
    if indices is None:
      indices = range(self.size())
    return np.array([getattr(self._memory[index], name) for index in indices])

  def set_sequence(self, name, values, indices=None):
    if indices is None:
      indices = range(self.size())
    for (index, value) in zip(indices, values):
      self._memory[index] = self._memory[index]._replace(**{name: value})

  def shuffled_batches(self, batch_size):
    indices = list(range(self.size()))
    random.shuffle(indices)
    for start in range(0, self.size(), batch_size):
      samples = [
          self._memory[index] for index in indices[start:(start + batch_size)]]
      yield self._make_batch(samples)

  @staticmethod
  def _make_batch(samples):
    first_sample = samples[0]
    fields = {
        field: np.array([getattr(sample, field) for sample in samples])
        for field in first_sample._fields
        if getattr(first_sample, field) is not None
    }
    fields["weight"] = np.ones(len(samples))
    return TransitionBatch(**fields)
