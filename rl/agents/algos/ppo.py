import numpy as np
import tensorflow as tf
from ..agent import Agent
from ..registry import register
from .utils import copy_variables_op
from ...utils.logger import log_scalar
from ...models.registry import get_model
from .utils import normalize, one_hot
from .advantage_estimator.registry import get_advantage_estimator


@register
class PPO(Agent):
  """ Proximal Policy Optimization """

  def __init__(self, sess, hparams):
    assert hparams.memory == "simple", "PPO only works with simple memory."
    super().__init__(sess, hparams)
    self.actor = get_model(hparams, register="PPOActor", name="actor")
    self.critic = get_model(hparams, register="PPOCritic", name="critic")
    self.target_actor = get_model(
        hparams, register="PPOActor", name="target_actor")
    self.advantage_estimator = get_advantage_estimator(
        self._hparams.advantage_estimator
    )

    self.build()

  def act(self, state, worker_id):
    if state.ndim < len(self._hparams.state_shape) + 1:
      state = np.expand_dims(state, axis=0)

    action_distribution = self._sess.run(
        self.probs, feed_dict={self.last_states: state})
    return self._action_function(self._hparams, action_distribution, worker_id)

  def observe(self, last_state, action, reward, done, state, worker_id=0):
    action = one_hot(action, self._hparams.num_actions)

    memory = self._memory[worker_id]
    memory.add_sample(
        last_state=last_state,
        action=action,
        reward=reward,
        discount=self._hparams.gamma,
        done=done,
        state=state,
    )

    if memory.size() == self._hparams.num_steps:
      self.update(worker_id)

  def reset(self, worker_id=0):
    self._memory[worker_id].clear()

  def clone_weights(self):
    self.target_actor.set_weights(self.actor.get_weights())

  def update_targets(self):
    self._sess.run(self.target_update_op)

  def _build_target_update_op(self):
    with tf.variable_scope("update_target_networks"):
      self.target_update_op = copy_variables_op(
          source=self.actor, target=self.target_actor)

  def build(self):
    self.last_states = tf.placeholder(
        tf.float32, [None] + self._hparams.state_shape, name="last_states")
    self.advantages = tf.placeholder(tf.float32, [None], name="advantages")
    self.discounted_rewards = tf.placeholder(
        tf.float32, [None], name="discounted_rewards")
    self.actions = tf.placeholder(
        tf.int32, [None, self._hparams.num_actions], name="actions")

    last_states = self.process_states(self.last_states)

    if self._hparams.pixel_input:
      self.cnn_vars = self._state_processor.trainable_weights
    else:
      self.cnn_vars = None

    self.logits = self.actor(last_states)

    self.probs = tf.nn.softmax(self.logits, -1)

    target_logits = self.target_actor(last_states)

    self.values = self.critic(last_states)[:, 0]

    losses, train_ops = self._grad_function(
        logits={
            "target_logits": target_logits,
            "logits": self.logits
        },
        actions=self.actions,
        advantages=self.advantages,
        values=self.values,
        discounted_rewards=self.discounted_rewards,
        hparams=self._hparams,
        var_list={
            "actor_vars": self.actor.trainable_weights,
            "critic_vars": self.critic.trainable_weights,
            "cnn_vars": self.cnn_vars
        })

    self.actor_loss = losses['actor_loss']
    self.critic_loss = losses['critic_loss']
    self.actor_train_op = train_ops['actor_train_op']
    self.critic_train_op = train_ops['critic_train_op']
    self.state_processor_train_op = train_ops['state_processor_train_op']

    self._build_target_update_op()

  def update(self, worker_id=0):
    if not self._hparams.training:
      return

    memory = self._memory[worker_id]
    states = np.concatenate((
        memory.get_sequence('last_state'),
        memory.get_sequence('state', indices=[-1]),
    ))
    rewards = memory.get_sequence('reward')
    dones = memory.get_sequence('done')
    values = self._sess.run(self.values, feed_dict={self.last_states: states})
    advantages = self.advantage_estimator(rewards, values, dones, self._hparams)
    discounted_rewards = advantages + values[:-1]
    memory.set_sequence('discounted_reward', discounted_rewards)
    if self._hparams.normalize_reward:
      advantages = normalize(advantages)
    memory.set_sequence('advantage', advantages)

    for _ in range(self._hparams.num_epochs):
      for batch in memory.shuffled_batches(self._hparams.batch_size):
        feed_dict = {
            self.last_states: batch.last_state,
            self.actions: batch.action,
            self.advantages: batch.advantage,
            self.discounted_rewards: batch.discounted_reward,
        }

        self._sess.run(self.state_processor_train_op, feed_dict=feed_dict)

        actor_loss, _ = self._sess.run(
            [self.actor_loss, self.actor_train_op], feed_dict=feed_dict)
        log_scalar("loss/actor/worker_%d" % worker_id, actor_loss)

        critic_loss, _ = self._sess.run(
            [self.critic_loss, self.critic_train_op], feed_dict=feed_dict)
        log_scalar("loss/critic/worker_%d" % worker_id, critic_loss)

    memory.clear()

    self.update_targets()
