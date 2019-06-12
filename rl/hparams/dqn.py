from .defaults import default
from .registry import register


@register
def dqn():
  hps = default()
  hps.models = ["basic"]
  hps.agent = "DQN"
  hps.action_function = "epsilon_action"
  hps.grad_function = "huber_loss"
  hps.lr = {
    "lr": 0.001
  }
  hps.lr_decay = {
    "lr": "no_decay"
  }
  hps.max_epsilon = 1.0
  hps.epsilon_decay_rate = 0.995
  hps.gamma = 0.99
  hps.hidden_size = 50
  hps.update_target_interval = 2500
  hps.memory = "prioritized"
  hps.memory_size = 50000
  # prioritized experience replay
  hps.memory_update_priorities = True
  hps.memory_priority_control = 0.6
  hps.memory_priority_compensation = 0.4

  return hps


@register
def dqn_cartpole():
  hps = dqn()
  hps.env = "CartPole-v1"
  return hps


@register
def dqn_mountaincar():
  hps = dqn()
  hps.env = "MountainCar-v0"
  hps.reward_augmentation = "mountain_car_default"
  hps.lr = {
    "lr": 0.01
  }
  hps.gamma = 0.95
  return hps


@register
def dqn_pong_ram():
  hps = dqn()
  hps.env = "Pong-ram-v0"
  hps.lr = {
    "lr": 0.0001
  }
  return hps


@register
def dqn_pong():
  hps = dqn_pong_ram()
  hps.env = "Pong-v0"
  return hps


@register
def noisy_dqn_cartpole():
  hps = dqn()
  hps.models = ["NoisyNetwork"]
  hps.action_function = "max_action"
  return hps
