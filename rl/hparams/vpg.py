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
  hps.learning_rate = 0.001
  hps.gamma = 0.95
  hps.hidden_size = 50
  hps.memory_size = 50000
  return hps


@register
def vpg_cartpole():
  hps = vpg()
  hps.env = "CartPole-v1"
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
  hps.n_steps = 10
  return hps
