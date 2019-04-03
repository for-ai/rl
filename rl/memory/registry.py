_MEMORIES = dict()


def register(fn):
  global _MEMORIES
  _MEMORIES[fn.__name__] = fn
  return fn


def get_memory(hparams, worker_id=0):
  return _MEMORIES[hparams.memory](hparams, worker_id)
