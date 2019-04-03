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
class NoisyNetwork(basic):

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
