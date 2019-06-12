from collections import namedtuple
import random
import numpy as np


_required_fields = ["last_state", "action", "reward", "done", "state"]
_optional_fields = [
    "discount", "discounted_reward", "advantage", "index", "weight"]
TransitionBatch = namedtuple(
    "TransitionBatch", _required_fields + _optional_fields)
TransitionBatch.__new__.__defaults__= (None,) * len(_optional_fields)


class Memory:
  """ Base class for memories """

  def __init__(self, hparams, worker_id):
    self._hparams = hparams
    self._worker_id = worker_id

  def add_sample(self, **kwargs):
    raise NotImplementedError

  def sample(self, batch_size):
    raise NotImplementedError

  def size(self):
    raise NotImplementedError

  def clear(self):
    raise NotImplementedError

  def get_sequence(self, name, indices=None):
    raise NotImplementedError

  def set_sequence(self, name, values, indices=None):
    raise NotImplementedError
