_GRADIENT_FUNCTIONS = dict()


def register(fn):
  global _GRADIENT_FUNCTIONS
  _GRADIENT_FUNCTIONS[fn.__name__] = fn
  return fn


def get_gradient_function(name):
  return _GRADIENT_FUNCTIONS[name]
