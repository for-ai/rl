import tensorflow as tf
from tensorflow import AUTO_REUSE as reuse

from ..model import Model
from ..registry import register


@register
class basic(Model):

  def __init__(self, hparams, name="basic"):
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
class CNN(Model):
  """ Pixel preprocessor for Atari described in Asynchronous Methods for Deep RL """

  def __init__(self, hparams, name='CNN'):
    super().__init__(hparams, name)
    self.conv1 = tf.layers.Conv2D(
        filters=16,
        kernel_size=8,
        strides=4,
        activation=tf.nn.relu,
        name="conv1")
    self.conv2 = tf.layers.Conv2D(
        filters=32,
        kernel_size=4,
        strides=2,
        activation=tf.nn.relu,
        name="conv2")
    self.flatten = tf.layers.Flatten()
    self.dense = tf.layers.Dense(
        units=hparams.state_latent_size, activation=tf.nn.relu, name="dense")

  def call(self, state):
    with tf.variable_scope(self.name, reuse=reuse):
      layer = self.conv1(state)
      layer = self.conv2(layer)
      layer = self.flatten(layer)
      return self.dense(layer)
