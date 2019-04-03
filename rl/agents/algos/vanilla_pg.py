import tensorflow as tf

from .utils import compute_discounted_rewards, one_hot
from ..agent import Agent
from ..registry import register
from ...utils.logger import log_scalar
from ...models.registry import get_model


@register
class VanillaPG(Agent):
  """ Vanilla Policy Gradient """

  def __init__(self, sess, hparams):
    super().__init__(sess, hparams)
    self.model = get_model(hparams, register="basic", name="model")
    self.build()

  def act(self, state, worker_id=0):
    action_distribution = self._sess.run(
        self.probs, feed_dict={self.states: state[None, :]})
    return self._action_function(self._hparams, action_distribution)

  def observe(self, last_state, action, reward, done, state, worker_id=0):
    action = one_hot(action, self._hparams.num_actions)

    self._memory[worker_id].add_sample(last_state, action, reward,
                                       self._hparams.gamma, done, state)

    if done:
      self.update(worker_id)

  def build(self):
    self.states = tf.placeholder(
        tf.float32, [None] + self._hparams.state_shape, name="states")
    self.rewards = tf.placeholder(tf.float32, [None], name="rewards")
    self.actions = tf.placeholder(
        tf.int32, [None, self._hparams.num_actions], name="actions")

    states = self.process_states(self.states)

    discounted_reward = compute_discounted_rewards(self._hparams, self.rewards)

    self.logits = self.model(states)

    self.probs = tf.nn.softmax(self.logits, -1)

    self.loss, self.train_op = self._grad_function(
        self.logits, self.actions, discounted_reward, self._hparams)

  def update(self, worker_id=0):
    memory = self._memory[worker_id]
    if self._hparams.training:
      _, _, states, actions, rewards, _, _ = memory.sample(
          self._hparams.batch_size)

      loss, _ = self._sess.run([self.loss, self.train_op],
                               feed_dict={
                                   self.states: states,
                                   self.actions: actions,
                                   self.rewards: rewards,
                               })
      log_scalar("loss/worker_%d" % worker_id, loss)

      memory.clear()
