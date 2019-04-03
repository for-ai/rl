import tensorflow as tf

_MODELS = dict()


def register(fn):
  global _MODELS
  _MODELS[fn.__name__] = fn
  return fn


def get_model(hparams, register, name):
  '''
  register: string, the register name of the model
  name: string, the name scope of the model
  '''

  return _MODELS[register](hparams, name)
