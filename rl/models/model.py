import tensorflow as tf


class Model(tf.keras.Model):

  def __init__(self, hparams, name):
    super().__init__(name=name)
    self._hparams = hparams

  def call(self):
    raise NotImplementedError
