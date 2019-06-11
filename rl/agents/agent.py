import tensorflow as tf
from ..models.registry import get_model
from ..memory.registry import get_memory
from .algos.action_function.registry import get_action_function
from .algos.compute_gradient.registry import get_gradient_function


class Agent():

  def __init__(self, sess, hparams):
    self._sess = sess
    self._hparams = hparams
    # list of memory where the index corresponds to the worker_id
    self._memory = [get_memory(hparams, i) for i in range(hparams.num_workers)]

    self._action_function = get_action_function(self._hparams.action_function)

    self._state_processor_vars = None

    if hparams.pixel_input:
      self._state_processor = get_model(
          hparams, register=hparams.state_processor, name="state_processor")

    self._grad_function = get_gradient_function(self._hparams.grad_function)

  def process_states(self, states):
    """ return processed raw pixel input otherwise return raw states"""
    if self._hparams.pixel_input:
      return self._state_processor(states)
    return states

  def build(self):
    """Construct TF graph."""
    raise NotImplementedError

  def observe(self, last_states, actions, rewards, done, states):
    """Allow agent to update internal state, etc.

    Args:
      last_state: list of Tensor of previous states.
      action: list of Tensor of actions taken to reach `state`.
      reward: list of Tensor of rewards received from environment.
      state: list of Tensor of new states.
      done: list of boolean indicating completion of episode.
    """
    raise NotImplementedError

  def act(self, state):
    """Select an action to take.

    Args:
      state: a Tensor of states.
    Returns:
      action: a Tensor of the selected action.
    """
    raise NotImplementedError

  def reset(self, worker_id=0):
    """Reset the agent's internal state.

    Called when switching between train/eval phases.
    """

  def update(self, worker_id=0):
    """Called at the end of an episode. Compute updates to models, etc."""
    raise NotImplementedError

  def update_targets(self):
    """ update target model with source model in self.target_update_op """
    raise NotImplementedError

  def clone_weights(self):
    """ Clone target weights with shared weights before training to ensure all
    models have identical weights to begin with """
    raise NotImplementedError

  def _build_target_update_op(self):
    """ build update target models operations at self.target_update_op """
    raise NotImplementedError

  def clear_memory(self, worker_id=0):
    self._memory[worker_id].clear()

  def memory_size(self, worker_id=0):
    return self._memory[worker_id].size()
