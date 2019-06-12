import tensorflow as tf
from .logger import log_scalar

_LR = dict()

def register(name):

  def add_to_dict(fn):
    global _LR
    _LR[name] = fn
    return fn

  return add_to_dict

@register("exponential")
def exponential_decay(hparams, lr, delay=0):
  gs = hparams.global_step
  gs -= delay
  return tf.train.exponential_decay(
      lr,
      gs,
      hparams.learning_rate_decay_interval,
      hparams.learning_rate_decay_rate,
      staircase=hparams.staircased)

@register("no_decay")
def no_decay(hparams, lr, delay=0):
  return lr

def update_learning_rate(hparams):
  for lr_name in hparams.lr:
    decay_fn = _LR[hparams.lr_decay[lr_name]]
    hparams.lr[lr_name] = decay_fn(hparams, hparams.lr[lr_name])
