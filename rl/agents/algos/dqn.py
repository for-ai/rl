import numpy as np
import tensorflow as tf
from ..agent import Agent
from ..registry import register
from ...models.registry import get_model
from ..algos.utils import copy_variables_op
from ...utils.logger import log_scalar, log_histogram


@register
class DQN(Agent):
  """ Deep Q Network """

  def __init__(self, sess, hparams):
    super().__init__(sess, hparams)
    # list of epsilon for each thread
    hparams.epsilon = [hparams.max_epsilon] * hparams.num_workers
    # set minimum epsilon for each worker
    # https://arxiv.org/pdf/1602.01783.pdf Section 8
    self._hparams.min_epsilon = list(
        np.random.choice([0.1, 0.01, 0.5],
                         size=self._hparams.num_workers,
                         p=[0.4, 0.3, 0.3]))

    self.model = get_model(hparams, register="basic", name="q_values")

    self.target_model = get_model(
        hparams, register="basic", name="target_q_values")

    self.build()

  def _epsilon_decay(self, worker_id=0):
    if self._hparams.epsilon[worker_id] > self._hparams.min_epsilon[worker_id]:
      self._hparams.epsilon[worker_id] *= self._hparams.epsilon_decay_rate
    else:
      self._hparams.epsilon[worker_id] = self._hparams.min_epsilon[worker_id]

    log_scalar("epsilon/worker_%d" % worker_id,
               self._hparams.epsilon[worker_id])

  def act(self, state, worker_id=0):
    action_distribution = self._sess.run(
        self.logits, feed_dict={self.last_states: state[None, :]})
    return self._action_function(self._hparams, action_distribution, worker_id)

  def observe(self, last_state, action, reward, done, state, worker_id=0):
    if done:
      state = np.zeros(state.shape)

    self._memory[worker_id].add_sample(last_state, action, reward,
                                       self._hparams.gamma, done, state)

    if self._hparams.local_step[
        worker_id] % self._hparams.batch_size == 0 or done:
      self.update(worker_id)

    if self._hparams.global_step % self._hparams.update_target_interval == 0:
      self.update_target()

  def _build_target_update_op(self):
    with tf.variable_scope("update_target_networks"):
      self.target_update_op.extend(
          copy_variables_op(source=self.model, target=self.target_model))

  def build(self):
    self.last_states = tf.placeholder(
        tf.float32, [None] + self._hparams.state_shape, name="last_states")
    self.rewards = tf.placeholder(tf.float32, [None], name="rewards")
    self.actions = tf.placeholder(tf.int32, [None], name="actions")
    self.done = tf.placeholder(tf.bool, [None], name="done")
    self.states = tf.placeholder(tf.float32, [None] + self._hparams.state_shape,
                                 "states")
    self.importance_sampling_weights = tf.placeholder(
        tf.float32, [None], name="importance_sampling_weights")

    last_states = self.process_states(self.last_states)
    states = self.process_states(self.states)

    # predict q value Q(s, a)
    self.logits = self.model(last_states)
    # convert action to one hot vector
    action_mask = tf.one_hot(self.actions, self._hparams.num_actions, axis=-1)
    predict_q = tf.boolean_mask(self.logits, action_mask)

    # target q value Q(s', a')
    target_q = tf.where(
        self.done, self.rewards, self.rewards +
        self._hparams.gamma * tf.reduce_max(self.target_model(states), axis=-1))

    # temporal difference
    self.td_error = tf.abs(target_q - predict_q)

    self.loss, self.train_op = self._grad_function(
        preds=predict_q,
        targets=target_q,
        hparams=self._hparams,
        weights=self.importance_sampling_weights,
        var_list=self.model.trainable_weights)

    # update target network
    self._build_target_update_op()

  def update(self, worker_id=0):
    memory = self._memory[worker_id]
    if self._hparams.training and memory.size() > self._hparams.batch_size:
      indices, weights, last_states, actions, rewards, done, states = memory.sample(
          self._hparams.batch_size)

      loss, _, td_errors = self._sess.run(
          [self.loss, self.train_op, self.td_error],
          feed_dict={
              self.last_states: last_states,
              self.actions: actions,
              self.rewards: rewards,
              self.done: done,
              self.states: states,
              self.importance_sampling_weights: weights
          })

      if self._hparams.memory == "PrioritizedMemory":
        memory.update(indices, td_errors)

      if self._hparams.num_workers > 1:
        memory.clear()

      self._epsilon_decay(worker_id)

      log_scalar("loss/worker_%d" % worker_id, loss)
