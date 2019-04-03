import os
_SYS = dict()


def register(cls):
  global _SYS
  _SYS[cls.__name__.lower()] = cls()
  return cls


def get_sys(name):
  return _SYS[name]


@register
class GCP(object):
  data_dir = "gs://for-ai/data"
  output_dir = "gs://for-ai/runs"


@register
class TPU(object):
  data_dir = "gs://for-ai/data"
  output_dir = "gs://for-ai/runs"


@register
class Local(object):
  data_dir = os.path.join(os.path.abspath(os.sep), 'tmp', 'data')
  output_dir = os.path.join(os.path.abspath(os.sep), 'tmp', 'runs')
