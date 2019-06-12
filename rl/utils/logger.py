import os
import numpy as np
import tensorflow as tf

LOGGER = None


def init_logger(hparams):
  global LOGGER
  LOGGER = Logger(hparams)


def log_scalar(tag, value):
  global LOGGER
  LOGGER.log_scalar(tag, value)


def log_histogram(tag, values):
  global LOGGER
  LOGGER.log_histogram(tag, values)


def log_graph():
  global LOGGER
  LOGGER.log_graph()


class Logger(object):
  """ 
  Log tf.Summary to output_dir during training and output_dir/eval during 
  evaluation
  """

  def __init__(self, hparams):
    self._hparams = hparams
    self._writer = tf.summary.FileWriter(hparams.run_output_dir)

  def log_scalar(self, tag, value):
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
    self._writer.add_summary(summary, self._hparams.global_step)

  def log_histogram(self, tag, values, bins=1000):
    """ https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514 """
    values = np.array(values)

    counts, bin_edges = np.histogram(values, bins=bins)

    hist = tf.HistogramProto()
    hist.min = float(np.min(values))
    hist.max = float(np.max(values))
    hist.num = int(np.prod(values.shape))
    hist.sum = float(np.sum(values))
    hist.sum_squares = float(np.sum(values**2))

    bin_edges = bin_edges[1:]

    for edge in bin_edges:
      hist.bucket_limit.append(edge)
    for c in counts:
      hist.bucket.append(c)

    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
    self.writer.add_summary(summary, self._hparams.global_step)
    self.writer.flush()

  def log_graph(self):
    self._writer.add_graph(tf.get_default_graph())

  @property
  def writer(self):
    return self._writer
