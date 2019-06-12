import tensorflow as tf
import numpy as np
from .env import Environment
from .registry import register_env, get_reward_augmentation


@register_env
class CoinRun(Environment):

  def __init__(self, hparams):
    # only support 1 environment currently
    super().__init__(hparams)
    try:
      from coinrun import setup_utils, make
      setup_utils.setup_and_load(use_cmd_line_args=False)

      self._env = make('standard', num_envs=1)
    except ImportError as e:
      print(e)
      print("please check README for CoinRun installation instruction")
      exit()
    self.seed(1234)
    self._observation_space = self._env.observation_space
    self._action_space = self._env.action_space
    self._hparams.num_states = self._observation_space.shape[0]
    self._hparams.num_actions = self._action_space.n
    self._hparams.state_shape = list(self._observation_space.shape)
    self._hparams.action_space_type = self._action_space.__class__.__name__
    self._hparams.pixel_input = True
    if self._hparams.reward_augmentation is not None:
      self._reward_augmentation = get_reward_augmentation(
          self._hparams.reward_augmentation)

  def step(self, action):
    """Run environment's dynamics one step at a time."""
    action = np.asarray(action)
    if action.ndim < 1:
      action = np.expand_dims(action, axis=0)

    state, reward, done, info = self._env.step(action)

    # remove single dimensional entries
    state = self._process_state(state)
    reward = np.squeeze(reward)
    done = np.squeeze(done)

    if self._reward_augmentation is not None:
      reward = self._reward_augmentation(state, reward, done, info)
    return state, reward, done, info

  def reset(self):
    """ CoinRun has no reset.
    https://github.com/openai/coinrun/blob/master/coinrun/coinrunenv.py#L181
    """
    state, _, _, _ = self._env.step_wait()

    state = self._process_state(state)

    return state

  def _process_state(self, state):
    """ Convert pixel input to int8 and remove single entry at 0 axis """
    if self._hparams.pixel_input:
      state = state.astype(np.int8)

    # remove single dimensional entry for CoinRun
    if state.ndim == len(self._hparams.state_shape) + 1:
      state = np.squeeze(state, axis=0)
    return state

  def close(self):
    """Perform any necessary cleanup when environment closes."""
    return

  def seed(self, seed):
    pass

  def render(self, mode='human'):
    """Renders the environment."""
    self._env.render()
