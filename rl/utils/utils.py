import tensorflow as tf


class ModeKeys(object):
  TRAIN = tf.estimator.ModeKeys.TRAIN
  TEST = "test"
  EVAL = tf.estimator.ModeKeys.EVAL
  PREDICT = tf.estimator.ModeKeys.PREDICT
