from .registry import register_reward


@register_reward
def mountain_car_default(observation, reward, done, info):
  """ mountain car has observation [position, velocity]

  https://github.com/openai/gym/wiki/MountainCar-v0
  """
  if observation[0] >= 0.5:
    reward += 100
  elif observation[0] >= 0.25:
    reward += 20
  elif observation[0] >= 0.1:
    reward += 10
  return reward
