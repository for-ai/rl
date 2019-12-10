from .defaults import default
from .registry import register


@register
def ppo():
  hps = default()
  hps.memory = "simple"
  hps.models = ["PPOActor", "PPOCritic"]
  hps.agent = "PPO"
  hps.lr = {'actor_lr': 0.0001, 'critic_lr': 0.0002}
  hps.lr_decay = {'actor_lr': 'no_decay', 'critic_lr': 'no_decay'}
  hps.batch_size = 32
  hps.num_steps = 128  # number of steps to unroll in every update
  hps.num_epochs = 5  # number of passes through the collected data
  hps.hidden_size = 100
  hps.gamma = 0.9
  hps.lambda_ = 0.95  # discount factor of TD returns in GAE
  hps.memory_size = 50000
  hps.action_function = "uniform_random_action"
  hps.grad_function = "ppo"
  hps.advantage_estimator = "gae"
  hps.normalize_reward = False
  hps.clipping_coef = 0.2

  return hps


@register
def ppo_cartpole():
  hps = ppo()
  hps.env = "CartPole-v1"
  hps.gamma = 0.98
  hps.num_steps = 32
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
  hps.env = "PongBaselines-v4"
  hps.num_epochs = 4
  hps.batch_size = 128
  hps.clipping_coef = 0.1
  hps.gamma = 0.99
  return hps
