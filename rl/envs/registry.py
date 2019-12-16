import gym
import tensorflow as tf
from .atari import register_atari

_ENVS = dict()
_REWARDS = dict()


def register_env(fn):
  global _ENVS
  _ENVS[fn.__name__] = fn
  return fn


def register_reward(fn):
  global _REWARDS
  _REWARDS[fn.__name__] = fn
  return fn


def get_gym_ids(hparams):
  """ return the list of all gym environments worker_id"""
  if not hparams.atari_registry:
    register_atari()
    hparams.atari_registry = True
  return [env.id for env in gym.envs.registry.all()]


def get_env(hparams):
  gym_ids = get_gym_ids(hparams)
  if hparams.env in _ENVS:
    return _ENVS[hparams.env](hparams)
  elif hparams.env in gym_ids:
    return _ENVS['GymEnv'](hparams)
  else:
    raise Exception(
        "Environment with name %s cannot not be found" % hparams.env)


def get_reward_augmentation(name):
  return _REWARDS[name]
