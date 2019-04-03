from .defaults import default
from .registry import register


@register
def ppo():
  hps = default()
  hps.models = ["PPOActor", "PPOCritic"]
  hps.agent = "PPO"
  hps.actor_lr = 0.0001
  hps.critic_lr = 0.0002
  hps.batch_size = 32
  hps.hidden_size = 100
  hps.gamma = 0.9
  hps.memory_size = 50000
  hps.action_function = "uniform_random_action"
  hps.grad_function = "ppo"
  hps.num_actor_steps = 10
  hps.num_critic_steps = 10
  hps.normalize_reward = False
  hps.clipping_coef = 0.2

  return hps


@register
def ppo_cartpole():
  hps = ppo()
  hps.env = "CartPole-v1"
  hps.gamma = 0.95
  return hps


@register
def ppo_mountaincar():
  hps = ppo()
  hps.env = "MountainCar-v0"
  hps.reward_augmentation = "mountain_car_default"
  return hps


@register
def ppo_pong():
  hps = ppo()
  hps.env = "Pong-v0"
  return hps
