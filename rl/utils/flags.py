import os
import time
import getpass
import tensorflow as tf

from .utils import ModeKeys
from .sys import get_sys


def log_hparams(hparams):
  if not tf.gfile.Exists(hparams.output_dir):
    tf.gfile.MakeDirs(hparams.output_dir)

  with tf.gfile.GFile(os.path.join(hparams.output_dir, 'hparams.txt'),
                      'a') as file:
    file.write("{}\n".format(time.ctime()))
    file.write("{}\n".format(str(vars(hparams))))


def validate_flags(FLAGS):
  messages = []
  if not FLAGS.sys:
    messages.append("Missing required flag --sys")
  if not FLAGS.hparams:
    messages.append("Missing required flag --hparams")

  if len(messages) > 0:
    raise Exception("\n".join(messages))

  return FLAGS


def update_hparams(FLAGS, hparams):
  # set hparams from FLAGS attribtues
  for attr in dir(FLAGS):
    if attr not in [
        'hparams', 'hparams_override', 'h', 'help', 'helpshort', 'env'
    ]:
      setattr(hparams, attr, getattr(FLAGS, attr))

  hparams.env = FLAGS.env or hparams.env
  if hparams.env is None:
    print("please specify training environment")
    exit()

  # check if agent support multi workers
  if hparams.agent not in ['DQN', 'DDPG', 'PPO'] and hparams.num_workers > 1:
    print("%s does not support multiple workers." % hparams.agent)
    exit()

  # set the mode for each thread to training mode
  hparams.mode = [ModeKeys.TRAIN] * hparams.num_workers

  # number of step for each thread
  hparams.local_step = [0] * hparams.num_workers

  # number of episode for each thread
  hparams.local_episode = [0] * hparams.num_workers

  sys = get_sys(FLAGS.sys)
  hparams.output_dir = FLAGS.output_dir or os.path.join(
      sys.output_dir, getpass.getuser(), FLAGS.hparams)

  log_hparams(hparams)

  return hparams
