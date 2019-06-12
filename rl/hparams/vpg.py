from .defaults import default
from .registry import register


@register
def vpg():
  hps = default()
  hps.models = ["basic"]
  hps.agent = "VanillaPG"
  hps.action_function = "non_uniform_random_action"
  hps.grad_function = "policy_gradient"
  hps.normalize_reward = True
  hps.lr = {
    "lr": 0.001
  }
  hps.ly_decay = {
    "lr": "no_decay"
  }
  hps.gamma = 0.95
  hps.hidden_size = 50
  hps.memory_size = 50000
  hps.num_episodes = 1  # number of episodes to collect for every update
  return hps


@register
def vpg_cartpole():
  hps = vpg()
  hps.env = "CartPole-v1"
  hps.gamma = 0.99
  hps.batch_size = 128
  hps.num_episodes = 4
  return hps


@register
def vpg_mountaincar():
  hps = vpg()
  hps.env = "MountainCar-v0"
  hps.reward_augmentation = "mountain_car_default"
  return hps


@register
def vpg_pong():
  hps = vpg()
  hps.env = "Pong-v0"
  hps.gamma = 0.99
  hps.hidden_size = 200
  return hps
