_ADVANTAGE_ESTIMATORS = {}


def register(fn):
  global _ADVANTAGE_ESTIMATORS
  _ADVANTAGE_ESTIMATORS[fn.__name__] = fn
  return fn


def get_advantage_estimator(name):
  return _ADVANTAGE_ESTIMATORS[name]
