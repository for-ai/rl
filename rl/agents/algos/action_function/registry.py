_ACTION_FUNCTIONS = dict()


def register(fn):
  global _ACTION_FUNCTIONS
  _ACTION_FUNCTIONS[fn.__name__] = fn
  return fn


def get_action_function(name):
  return _ACTION_FUNCTIONS[name]
