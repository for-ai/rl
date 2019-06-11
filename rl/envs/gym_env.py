import os
import gym
import numpy as np
from gym import wrappers
from .env import Environment
from .registry import register_env, get_reward_augmentation


@register_env
class GymEnv(Environment):
  """ https://jair.org/index.php/jair/article/view/10819/25823 """

  def __init__(self, hparams):
    super().__init__(hparams)

    try:
      self._env = gym.make(hparams.env)
    except gym.error.Error:
      print("Environment with name %s cannot not be found. "
            "Install Atari env with pip install 'gym[atari]'" % hparams.env)
      exit()

    if hparams.record_video:
      directory = hparams.run_output_dir
      if not hparams.training:
        directory = os.path.join(directory, 'eval')
      # save game play video every hparams.save_every
      self._env = wrappers.Monitor(
          self._env,
          directory=os.path.join(directory, 'video'),
          video_callable=lambda count: count % hparams.save_every == 0,
          force=True,
          mode='training' if hparams.training else 'evaluation')

    self.seed(self._hparams.seed)
    self._observation_space = self._env.observation_space
    self._action_space = self._env.action_space
    self._hparams.state_shape = list(self._observation_space.shape)

    # check if environment state is raw pixel input
    if len(self._hparams.state_shape) > 1:
      self._hparams.pixel_input = True
    else:
      self._hparams.pixel_input = False

    self._hparams.action_space_type = self._action_space.__class__.__name__

    if self._hparams.action_space_type == "Discrete":
      self._hparams.num_actions = self._action_space.n

    elif self._hparams.action_space_type == "Box":
      self._hparams.action_high = self._action_space.high
      self._hparams.action_low = self._action_space.low
      self._hparams.num_actions = len(self._hparams.action_low)

    if self._hparams.reward_augmentation is not None:
      self._reward_augmentation = get_reward_augmentation(
          self._hparams.reward_augmentation)

  def step(self, action):
    """Run environment's dynamics one step at a time."""
    state, reward, done, info = self._env.step(action)

    if self._reward_augmentation is not None:
      reward = self._reward_augmentation(state, reward, done, info)

    if self._hparams.pixel_input:
      state = state.astype(np.int8)
    return state, reward, done, info

  def reset(self):
    """Resets the state of the environment and returns an initial observation."""
    state = self._env.reset()
    if self._hparams.pixel_input:
      state = state.astype(np.int8)
    return state

  def close(self):
    """Perform any necessary cleanup when environment closes."""
    self._env.env.close()
    self._env.close()

  def seed(self, seed):
    """Sets the seed for this env's random number generator(s)."""
    self._env.seed(seed)

  def render(self, mode='human'):
    """Renders the environment."""
    self._env.render()
