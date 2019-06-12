_MEMORIES = dict()


def register(name):
  def add_to_dict(fn):
    global _MEMORIES
    _MEMORIES[name] = fn
    return fn
  return add_to_dict


def get_memory(hparams, worker_id=0):
  return _MEMORIES[hparams.memory](hparams, worker_id)
