import numpy as np
import tensorflow as tf

from .utils import compute_discounted_rewards, normalize, one_hot
from ..agent import Agent
from ..registry import register
from ...utils.logger import log_scalar
from ...models.registry import get_model


@register
class VanillaPG(Agent):
  """ Vanilla Policy Gradient """

  def __init__(self, sess, hparams):
    assert hparams.memory == "simple", (
        "VanillaPG only works with simple memory."
    )
    super().__init__(sess, hparams)
    self.model = get_model(hparams, register="basic", name="model")
    self.build()
    self._num_episodes_left = hparams.num_episodes

  def act(self, state, worker_id=0):
    if state.ndim < len(self._hparams.state_shape) + 1:
      state = np.expand_dims(state, axis=0)

    action_distribution = self._sess.run(
        self.probs, feed_dict={self.last_states: state})
    return self._action_function(self._hparams, action_distribution)

  def observe(self, last_state, action, reward, done, state, worker_id=0):
    action = one_hot(action, self._hparams.num_actions)

    self._memory[worker_id].add_sample(
        last_state=last_state,
        action=action,
        reward=reward,
        discount=self._hparams.gamma,
        done=done,
        state=state,
    )

    if done:
      self._num_episodes_left -= 1
      if self._num_episodes_left == 0:
        self.update(worker_id)
        self._num_episodes_left = self._hparams.num_episodes

  def reset(self, worker_id=0):
    self._memory[worker_id].clear()

  def clone_weights(self):
    pass

  def update_targets(self):
    pass

  def build(self):
    self.last_states = tf.placeholder(
        tf.float32, [None] + self._hparams.state_shape, name="last_states")
    self.discounted_rewards = tf.placeholder(
        tf.float32, [None], name="discounted_rewards")
    self.actions = tf.placeholder(
        tf.int32, [None, self._hparams.num_actions], name="actions")

    last_states = self.process_states(self.last_states)

    if self._hparams.pixel_input:
      self.cnn_vars = self._state_processor.trainable_weights
    else:
      self.cnn_vars = None

    self.logits = self.model(last_states)
    self.probs = tf.nn.softmax(self.logits, -1)

    self.loss, self.train_op = self._grad_function(
        self.logits, self.actions, self.discounted_rewards, self._hparams)

  def update(self, worker_id=0):
    if not self._hparams.training:
      return

    memory = self._memory[worker_id]
    rewards = memory.get_sequence("reward")
    dones = memory.get_sequence("done")
    discounted_rewards = compute_discounted_rewards(
        rewards, dones, self._hparams.gamma
    )
    if self._hparams.normalize_reward:
      discounted_rewards = normalize(discounted_rewards)
    memory.set_sequence("discounted_reward", discounted_rewards)

    for batch in memory.shuffled_batches(self._hparams.batch_size):
      loss, _ = self._sess.run(
          [self.loss, self.train_op],
          feed_dict={
              self.last_states: batch.last_state,
              self.actions: batch.action,
              self.discounted_rewards: batch.discounted_reward,
          }
      )
      log_scalar("loss/worker_%d" % worker_id, loss)

    memory.clear()
