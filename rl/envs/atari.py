# The MIT License
#
# Copyright (c) 2019 ForAI (http://for.ai)
# Copyright (c) 2017 OpenAI (http://openai.com)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# Based on OpenAI Baselines code:
# https://github.com/openai/baselines/blob/9b68103b737ac46bc201dfb3121cfa5df2127e53/baselines/common/atari_wrappers.py

from collections import deque
from functools import partial

import gym
from gym import spaces
from gym.envs.registration import register

import cv2
cv2.ocl.setUseOpenCL(False)

import numpy as np


class NoopResetEnv(gym.Wrapper):

  def __init__(self, env, noop_max=30):
    """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
    gym.Wrapper.__init__(self, env)
    self.noop_max = noop_max
    self.override_num_noops = None
    self.noop_action = 0
    assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

  def reset(self, **kwargs):
    """ Do no-op action for a number of steps in [1, noop_max]."""
    self.env.reset(**kwargs)
    if self.override_num_noops is not None:
      noops = self.override_num_noops
    else:
      noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
    assert noops > 0
    obs = None
    for _ in range(noops):
      obs, _, done, _ = self.env.step(self.noop_action)
      if done:
        obs = self.env.reset(**kwargs)
    return obs

  def step(self, ac):
    return self.env.step(ac)


class FireResetEnv(gym.Wrapper):

  def __init__(self, env):
    """
        Take action on reset for environments that are fixed until firing.
        :param env: (Gym Environment) the environment to wrap
        """
    gym.Wrapper.__init__(self, env)
    assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
    assert len(env.unwrapped.get_action_meanings()) >= 3

  def reset(self, **kwargs):
    self.env.reset(**kwargs)
    obs, _, done, _ = self.env.step(1)
    if done:
      self.env.reset(**kwargs)
    obs, _, done, _ = self.env.step(2)
    if done:
      self.env.reset(**kwargs)
    return obs

  def step(self, action):
    return self.env.step(action)


class EpisodicLifeEnv(gym.Wrapper):

  def __init__(self, env):
    """
        Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        :param env: (Gym Environment) the environment to wrap
        """
    gym.Wrapper.__init__(self, env)
    self.lives = 0
    self.was_real_done = True

  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    self.was_real_done = done
    # check current lives, make loss of life terminal,
    # then update lives to handle bonus lives
    lives = self.env.unwrapped.ale.lives()
    if 0 < lives < self.lives:
      # for Qbert sometimes we stay in lives == 0 condtion for a few frames
      # so its important to keep lives > 0, so that we only reset once
      # the environment advertises done.
      done = True
    self.lives = lives
    return obs, reward, done, info

  def reset(self, **kwargs):
    """
        Calls the Gym environment reset, only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        :param kwargs: Extra keywords passed to env.reset() call
        :return: ([int] or [float]) the first observation of the environment
        """
    if self.was_real_done:
      obs = self.env.reset(**kwargs)
    else:
      # no-op step to advance from terminal/lost life state
      obs, _, _, _ = self.env.step(0)
    self.lives = self.env.unwrapped.ale.lives()
    return obs


class MaxAndSkipEnv(gym.Wrapper):

  def __init__(self, env, skip=4):
    """Return only every `skip`-th frame"""
    gym.Wrapper.__init__(self, env)
    # most recent raw observations (for max pooling across time steps)
    self._obs_buffer = np.zeros(
        (2,) + env.observation_space.shape, dtype=np.uint8)
    self._skip = skip

  def step(self, action):
    """Repeat action, sum reward, and max over last observations."""
    total_reward = 0.0
    done = None
    for i in range(self._skip):
      obs, reward, done, info = self.env.step(action)
      if i == self._skip - 2:
        self._obs_buffer[0] = obs
      if i == self._skip - 1:
        self._obs_buffer[1] = obs
      total_reward += reward
      if done:
        break
    # Note that the observation on the done=True frame
    # doesn't matter
    max_frame = self._obs_buffer.max(axis=0)

    return max_frame, total_reward, done, info

  def reset(self, **kwargs):
    return self.env.reset(**kwargs)


class FrameStack(gym.Wrapper):

  def __init__(self, env, k):
    """Stack k last frames."""
    # In OpenAI Baselines it returns LazyFrames to lower the memory
    # footprint, but it caused problems with our code.
    # Let's keep that in mind and add it back if needed.
    gym.Wrapper.__init__(self, env)
    self.k = k
    self.frames = deque([], maxlen=k)
    shp = env.observation_space.shape
    self.observation_space = spaces.Box(
        low=0,
        high=255,
        shape=(shp[:-1] + (shp[-1] * k,)),
        dtype=env.observation_space.dtype)

  def reset(self):
    ob = self.env.reset()
    for _ in range(self.k):
      self.frames.append(ob)
    return self._get_ob()

  def step(self, action):
    ob, reward, done, info = self.env.step(action)
    self.frames.append(ob)
    return self._get_ob(), reward, done, info

  def _get_ob(self):
    assert len(self.frames) == self.k
    return np.concatenate(self.frames, axis=-1)


class ClipRewardEnv(gym.RewardWrapper):

  def __init__(self, env):
    """
        clips the reward to {+1, 0, -1} by its sign.
        :param env: (Gym Environment) the environment
        """
    gym.RewardWrapper.__init__(self, env)

  def reward(self, reward):
    """
        Bin reward to {+1, 0, -1} by its sign.
        :param reward: (float)
        """
    return np.sign(reward)


class WarpFrame(gym.ObservationWrapper):

  def __init__(self, env):
    """
        Warp frames to 84x84 as done in the Nature paper and later work.
        :param env: (Gym Environment) the environment
        """
    gym.ObservationWrapper.__init__(self, env)
    self.width = 84
    self.height = 84
    self.observation_space = spaces.Box(
        low=0,
        high=255,
        shape=(self.height, self.width, 1),
        dtype=env.observation_space.dtype)

  def observation(self, frame):
    """
        returns the current observation from a frame
        :param frame: ([int] or [float]) environment frame
        :return: ([int] or [float]) the observation
        """
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(
        frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
    return frame[:, :, None]


def make_baselines(env_id, max_episode_steps=None):
  env = gym.make(env_id)
  assert 'NoFrameskip' in env.spec.id
  env = NoopResetEnv(env, noop_max=30)
  env = MaxAndSkipEnv(env, skip=4)
  if max_episode_steps is not None:
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
  env = EpisodicLifeEnv(env)
  if 'FIRE' in env.unwrapped.get_action_meanings():
    env = FireResetEnv(env)
  env = WarpFrame(env)
  env = ClipRewardEnv(env)
  env = FrameStack(env, 4)
  return env


def register_atari():
  try:
    import atari_py
    # If atari envs are available, re-register them with the OpenAI Baselines
    # wrapper stack.
    for game in atari_py.list_games():
      name = ''.join([g.capitalize() for g in game.split('_')])
      for version in ('v0', 'v4'):
        register(
            id='{}Baselines-{}'.format(name, version),
            entry_point=partial(
                make_baselines,
                env_id='{}NoFrameskip-{}'.format(name, version),
            ))
  except ImportError:
    print('ImportError: Unable to import atari_py')
    exit()
