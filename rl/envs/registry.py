import gym
import tensorflow as tf

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


def get_gym_ids():
  """ return the list of all gym environments worker_id"""
  return [env_spec.id for env_spec in gym.envs.registry.all()]


def get_env(hparams):
  gym_ids = get_gym_ids()
  if hparams.env in _ENVS:
    return _ENVS[hparams.env](hparams)
  elif hparams.env in gym_ids:
    return _ENVS['GymEnv'](hparams)
  else:
    raise Exception(
        "Environment with name %s cannot not be found" % hparams.env)


def get_reward_augmentation(name):
  return _REWARDS[name]
