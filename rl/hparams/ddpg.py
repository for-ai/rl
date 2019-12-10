from .defaults import default
from .registry import register


@register
def ddpg():
  hps = default()
  hps.models = ['DDPGActor', 'DDPGCritic']
  hps.grad_function = "ddpg"
  hps.action_function = "normal_noise_action"
  hps.agent = "DDPG"
  hps.batch_size = 128
  hps.lr = {
    "actor_lr": 1e-4,
    "critic_lr": 1e-3
  }
  hps.lr_decay = {
    "actor_lr": "no_decay",
    "critic_lr": "no_decay"
  }
  hps.soft_replace_ratio = 1e-3
  hps.gamma = 0.99
  hps.max_variance = 2
  hps.min_variance = 0.0001
  hps.variance_decay = 0.9995
  hps.hidden_size = 50
  hps.memory_size = 50000
  return hps


@register
def ddpg_cartpole():
  hps = ddpg()
  hps.action_function = "uniform_random_action"
  hps.env = "CartPole-v1"
  hps.hidden_size = 200
  hps.memory = "prioritized"
  hps.memory_size = 50000
  # prioritized experience replay
  hps.memory_update_priorities = True
  hps.memory_priority_control = 0.6
  hps.memory_priority_compensation = 0.4
  return hps


@register
def ddpg_pendulum():
  hps = ddpg()
  hps.env = "Pendulum-v0"
  hps.memory = "prioritized"
  hps.memory_size = 50000
  # prioritized experience replay
  hps.memory_update_priorities = True
  hps.memory_priority_control = 0.6
  hps.memory_priority_compensation = 0.4
  return hps


@register
def ddpg_mountaincar():
  hps = ddpg()
  hps.env = "MountainCar-v0"
  hps.action_function = "uniform_random_action"
  hps.memory = "prioritized"
  hps.memory_size = 50000
  # prioritized experience replay
  hps.memory_update_priorities = True
  hps.memory_priority_control = 0.6
  hps.memory_priority_compensation = 0.4
  hps.reward_augmentation = "mountain_car_default"
  return hps


@register
def ddpg_mountaincar_continuous():
  hps = ddpg()
  hps.env = "MountainCarContinuous-v0"
  return hps


@register
def ddpg_pong():
  hps = ddpg()
  hps.action_function = "uniform_random_action"
  hps.env = "Pong-v0"
  return hps


@register
def ddpg_carracing():
  hps = ddpg()
  hps.env = "CarRacing-v0"
  hps.batch_size = 64
  hps.actor_lr = 0.0001
  hps.critic_lr = 0.0001
  hps.soft_replace_ratio = 0.01
  hps.gamma = 0.90
  return hps


@register
def ddpg_coinrun():
  hps = ddpg()
  hps.action_function = "uniform_random_action"
  hps.env = 'CoinRun'
  return hps
