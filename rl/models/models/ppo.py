import tensorflow as tf
from tensorflow import AUTO_REUSE as reuse

from ..model import Model
from ..registry import register


@register
class PPOActor(Model):

  def __init__(self, hparams, name="PPOActor"):
    super().__init__(hparams, name)
    self.layer1 = tf.layers.Dense(
        units=hparams.hidden_size, activation=tf.nn.relu)
    self.layer2 = tf.layers.Dense(
        units=hparams.hidden_size, activation=tf.nn.relu)
    self.layer3 = tf.layers.Dense(units=hparams.num_actions)

  def call(self, states):
    with tf.variable_scope(self.name, reuse=reuse):
      layer = self.layer1(states)
      layer = self.layer2(layer)
      return self.layer3(layer)


@register
class PPOCritic(Model):

  def __init__(self, hparams, name='PPOCritic'):
    super().__init__(hparams, name)
    self.layer1 = tf.layers.Dense(
        units=hparams.hidden_size, activation=tf.nn.relu)
    self.layer2 = tf.layers.Dense(units=1)

  def call(self, states):
    with tf.variable_scope(self.name, reuse=reuse):
      layer = self.layer1(states)
      return self.layer2(layer)
