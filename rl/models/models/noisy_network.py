import tensorflow as tf
from tensorflow import AUTO_REUSE as reuse

from ..model import Model
from ..registry import register


class NoisyLayer(tf.keras.layers.Layer):
  """ Noisy Layer as described in Noisy Networks For Exploration """

  def __init__(self, hparams, units, activation=None, name="noisy_layer"):
    super().__init__()
    self._hparams = hparams
    self._units = units
    self._activation = activation
    self._name = name

  def build(self, input_shape):
    self._input_shape = int(input_shape[-1])

    def mu_initializer(size):
      return tf.random_uniform_initializer(
          minval=-tf.sqrt(3 / size), maxval=tf.sqrt(3 / size))

    self._w_mu = self.add_variable(
        "w_mu",
        shape=[self._input_shape, self._units],
        initializer=mu_initializer(self._input_shape))
    self._w_sigma = self.add_variable(
        "w_sigma",
        shape=[self._input_shape, self._units],
        initializer=tf.constant_initializer(0.017))

    self._b_mu = self.add_variable(
        "b_mu", shape=[self._units], initializer=mu_initializer(self._units))
    self._b_sigma = self.add_variable(
        "b_sigma",
        shape=[self._units],
        initializer=tf.constant_initializer(0.017))

  def call(self, x):
    with tf.name_scope(self.name):
      w = self._w_mu + tf.multiply(
          tf.random_normal(self._w_sigma.shape), self._w_sigma)
      b = self._b_mu + tf.multiply(
          tf.random_normal(self._b_sigma.shape), self._b_sigma)
      y = tf.matmul(x, w) + b
      if self._activation is not None:
        y = self._activation(y)
      return y


@register
class NoisyNetwork(Model):

  def __init__(self, hparams, scope="NoisyNetwork"):
    super().__init__(hparams, scope)
    self.layer1 = NoisyLayer(
        hparams,
        units=self._hparams.hidden_size,
        activation=tf.nn.relu,
        name="noisy_layer_1")
    self.layer2 = NoisyLayer(
        hparams,
        units=self._hparams.hidden_size,
        activation=tf.nn.relu,
        name="noisy_layer_2")
    self.layer3 = NoisyLayer(
        hparams, units=self._hparams.num_actions, name="noisy_layer_3")

  def call(self, states):
    with tf.variable_scope(self.name, reuse=reuse):
      layer = self.layer1(states)
      layer = self.layer2(layer)
      return self.layer3(layer)
