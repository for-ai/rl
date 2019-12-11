import os
import time
import pickle
import tensorflow as tf

from .logger import log_scalar


class Checkpoint():
  """ Save and restore using tf.train.Saver()

  Save hparams to pickle file with format
  {
    checkpoints: list of checkpoint_path chronological  order
    checkpoint_path: (hparams, all_rewards)
  }
  """

  def __init__(self, sess, hparams):
    self._sess = sess
    self._hparams = hparams
    self._checkpoints = []
    self._run_dir = hparams.run_output_dir
    self._pickle = os.path.join(self._run_dir, 'checkpoint.pickle')
    self._saver = tf.train.Saver(max_to_keep=3)
    self._last_record_time = -1
    self._last_record_step = -1

  def save(self):
    if self._hparams.test_only:
      return

    save_path = os.path.join(self._run_dir,
                             'model.ckpt-%d' % self._hparams.global_step)
    path_prefix = self._saver.save(self._sess, save_path)

    self._checkpoints.append(path_prefix)

    with open(self._pickle, "wb") as file:
      pickle.dump({
          'checkpoints': self._checkpoints,
          path_prefix: (self._hparams)
      }, file)

    print("saved checkpoint: %s" % path_prefix)

    # log fps
    if self._last_record_step > 0 and self._last_record_time > 0:
      fps = round((self._hparams.total_step - self._last_record_step) /
                  (time.time() - self._last_record_time), 2)
      log_scalar('fps', fps)
    self._last_record_time = time.time()
    self._last_record_step = self._hparams.total_step

  def restore(self):
    """ Restore from latest checkpoint
    Returns:
      restored: boolean, True if restored from a checkpoint, False otherwise.
    """
    latest_checkpoint = tf.train.latest_checkpoint(self._run_dir)

    if latest_checkpoint is None:
      if self._hparams.test_only:
        raise FileNotFoundError("no checkpoint found in %s" % self._run_dir)
      return False

    self._saver.restore(self._sess, latest_checkpoint)

    with open(self._pickle, "rb") as file:
      checkpoints = pickle.load(file)

      self._checkpoints = checkpoints['checkpoints']
      previous_hparams = checkpoints[latest_checkpoint]

      # check the checkpoint is compatilbe with current hparams
      if previous_hparams.agent != self._hparams.agent:
        print("incompatible agent from checkpoint %s" % latest_checkpoint)
        exit()
      if (not self._hparams.test_only and
          previous_hparams.num_workers != self._hparams.num_workers):
        print("number of workers in checkpoint %s is not equal" %
              latest_checkpoint)
        exit()

      # restore hparams
      self._hparams.total_step = previous_hparams.total_step
      self._hparams.global_step = previous_hparams.global_step
      self._hparams.local_step = previous_hparams.local_step
      self._hparams.local_episode = previous_hparams.local_episode
      if hasattr(previous_hparams, "epsilon"):
        self._hparams.epsilon = previous_hparams.epsilon
        self._hparams.min_epsilon = previous_hparams.min_epsilon

    print("\nrestored from checkpoint %s\n" % latest_checkpoint)

    return True
