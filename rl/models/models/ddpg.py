import tensorflow as tf
from tensorflow import AUTO_REUSE as reuse

from ..model import Model
from ..registry import register


@register
class DDPGActor(Model):

  def __init__(self, hparams, name="DDPGActor"):
    super().__init__(hparams, name)
    self.action_space_type = hparams.action_space_type
    self.num_actions = hparams.num_actions

    self.layer1 = tf.layers.Dense(
        units=hparams.hidden_size, activation=tf.nn.relu)

    self.layer2 = tf.layers.Dense(
        units=hparams.hidden_size, activation=tf.nn.relu)

    kernel_initializer = tf.random_uniform_initializer(
        minval=-3e-3, maxval=3e-3)

    if hparams.action_space_type == "Box":
      # one layer for each continous action
      self.last_layers = []
      for i in range((hparams.num_actions)):
        if hparams.action_low[i] == 0. and hparams.action_high[i] == 1.:
          self.last_layers += [
              tf.layers.Dense(
                  units=1,
                  activation=tf.nn.sigmoid,
                  kernel_initializer=kernel_initializer)
          ]
        else:
          self.last_layers += [
              tf.layers.Dense(
                  units=1,
                  activation=tf.nn.tanh,
                  kernel_initializer=kernel_initializer)
          ]
    elif hparams.action_space_type == "Discrete":
      self.layer3 = tf.layers.Dense(
          units=hparams.num_actions,
          activation=tf.nn.tanh,
          kernel_initializer=kernel_initializer)
    else:
      NotImplementedError(
          'We only support gym environments of type Discrete and Box')

  def call(self, states):
    with tf.variable_scope(self.name, reuse=reuse):
      layer = self.layer1(states)

      layer = self.layer2(layer)

      if self.action_space_type == "Discrete":
        output = self.layer3(layer)
      elif self.action_space_type == "Box":
        for i in range(self.num_actions):
          if i == 0:
            output = self.last_layers[i](layer)
          else:
            output = tf.concat([output, self.last_layers[i](layer)], 1)
        output = tf.multiply(output, self._hparams.action_high)
      else:
        NotImplementedError(
            'We only support gym environments of type Discrete and Box')
      return output


@register
class DDPGCritic(Model):

  def __init__(self, hparams, name="DDPGCritic"):
    super().__init__(hparams, name)
    self.layer1 = tf.layers.Dense(
        units=hparams.hidden_size, activation=tf.nn.relu)
    self.layer2 = tf.layers.Dense(
        units=hparams.hidden_size, activation=tf.nn.relu)

    kernel_initializer = tf.random_uniform_initializer(
        minval=-3e-3, maxval=3e-3)

    if hparams.action_space_type == "Discrete":
      self.layer3 = tf.layers.Dense(
          units=hparams.num_actions, kernel_initializer=kernel_initializer)
    elif hparams.action_space_type == "Box":
      self.layer3 = tf.layers.Dense(
          units=1, kernel_initializer=kernel_initializer)
    else:
      NotImplementedError(
          'We only support gym environments of type Discrete and Box')

  def call(self, states, actions):
    with tf.variable_scope(self.name, reuse=reuse):
      x = tf.concat([states, actions], axis=-1)
      layer = self.layer1(x)
      layer = self.layer2(layer)
      output = self.layer3(layer)
      return output
