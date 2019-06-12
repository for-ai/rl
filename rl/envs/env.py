import tensorflow as tf


class Environment:

  def __init__(self, hparams):
    self._reward_range = (-float('inf'), float('inf'))
    self._action_space = None
    self._observation_space = None
    self._reward_augmentation = None
    self._hparams = hparams

  def step(self, action):
    """Run environment's dynamics one step at a time."""
    raise NotImplementedError

  def reset(self):
    """Resets the state of the environment and returns an initial observation."""
    raise NotImplementedError

  def close(self):
    """Perform any necessary cleanup when environment closes."""
    return

  def seed(self, seed=None):
    """Sets the seed for this env's random number generator(s)."""
    tf.logging.warn("No seed value to be set")
    return

  def render(self, mode='human'):
    """Renders the environment."""
    raise NotImplementedError
