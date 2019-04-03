_AGENTS = dict()


def register(fn):
  global _AGENTS
  _AGENTS[fn.__name__] = fn
  return fn


def get_agent(sess, hparams):
  return _AGENTS[hparams.agent](sess, hparams)
