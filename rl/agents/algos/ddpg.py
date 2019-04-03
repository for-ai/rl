import numpy as np
import tensorflow as tf

from ..agent import Agent
from .utils import one_hot, OrnsteinUhlenbeckActionNoise
from ..registry import register
from ...utils.logger import log_scalar
from ...models.registry import get_model
from ...utils.utils import ModeKeys


@register
class DDPG(Agent):
  """ Deep Deterministic Policy Gradient """

  def __init__(self, sess, hparams):
    super().__init__(sess, hparams)
    hparams.variance = hparams.max_variance
    hparams.learning_rate = hparams.critic_lr

    self.actor = get_model(hparams, register="DDPGActor", name="actor")
    self.critic = get_model(hparams, register="DDPGCritic", name="critic")

    self.target_actor = get_model(
        hparams, register="DDPGActor", name="target_actor")
    self.target_critic = get_model(
        hparams, register="DDPGCritic", name="target_critic")

    self.actor_noise = OrnsteinUhlenbeckActionNoise(
        mu=np.zeros(hparams.num_actions))

    self.build()

  def _variance_decay(self, worker_id):
    self._hparams.variance = max(
        self._hparams.min_variance,
        self._hparams.variance * self._hparams.variance_decay)
    log_scalar("variance/worker_%d" % worker_id, self._hparams.variance)

  def observe(self, last_state, action, reward, done, state, worker_id=0):
    if self._hparams.action_space_type == "Discrete":
      action = one_hot(action, self._hparams.num_actions)

    self._memory[worker_id].add_sample(last_state, action, reward,
                                       self._hparams.gamma, done, state)

    self.update(worker_id)

    self._variance_decay(worker_id)

  def act(self, state, worker_id=0):
    action = self._sess.run(
        self.action_pred, feed_dict={self.last_states: state[None, :]})

    if self._hparams.mode[worker_id] == ModeKeys.TRAIN:
      action = action + self.actor_noise()

    if self._hparams.num_actions == 1:
      # Box enviroments with one action, e.g. pendulum.
      action = np.squeeze(action, axis=-1)
    elif self._hparams.action_space_type == "Discrete":
      action = np.argmax(action)
    else:
      # Box enviroments with mutli-actions, e.g. carracing.
      action = np.squeeze(action)
    return action

  def _build_target_update_op(self):
    with tf.variable_scope("update_target_networks"):

      def soft_replace(source, target):
        ratio = self._hparams.soft_replace_ratio

        source_vars = sorted(source.trainable_weights, key=lambda v: v.name)
        target_vars = sorted(target.trainable_weights, key=lambda v: v.name)

        return [
            tf.assign(target_var, (1 - ratio) * target_var + ratio * source_var)
            for target_var, source_var in zip(target_vars, source_vars)
        ]

      self.target_update_op.extend(
          soft_replace(source=self.actor, target=self.target_actor))

      self.target_update_op.extend(
          soft_replace(source=self.critic, target=self.target_critic))

  def build(self):
    state_shape = [None] + self._hparams.state_shape
    self.last_states = tf.placeholder(
        tf.float32, state_shape, name="last_states")
    self.rewards = tf.placeholder(tf.float32, [None, 1], name="rewards")
    self.actions = tf.placeholder(
        tf.float32, [None, self._hparams.num_actions], name="actions")
    self.done = tf.placeholder(tf.float32, [None, 1], name="done")
    self.states = tf.placeholder(tf.float32, state_shape, name="states")

    last_states = self.process_states(self.last_states)
    states = self.process_states(self.states)

    self.actor_pred = self.actor(last_states)

    self.action_pred = self.actor_pred
    if self._hparams.action_space_type == "Discrete":
      self.action_pred = tf.nn.softmax(self.actor_pred, axis=-1)

    with tf.variable_scope("actor_loss"):
      critic_pred_from_actor = self.critic(last_states, self.actor_pred)

      # maximize q
      self.actor_loss = -tf.reduce_mean(critic_pred_from_actor)

    with tf.variable_scope("critic_loss"):
      target_actor_pred = self.target_actor(states)

      target_critic_pred = self.target_critic(states, target_actor_pred)

      td_target = self.rewards + self._hparams.gamma * tf.multiply(
          (1 - self.done), target_critic_pred)

      critic_pred_from_experience = self.critic(last_states, self.actions)

      self.critic_loss = tf.losses.mean_squared_error(
          labels=td_target, predictions=critic_pred_from_experience)

    self.critic_train_op = tf.train.AdamOptimizer(
        learning_rate=self._hparams.critic_lr,
        name="critic_optimizer").minimize(
            self.critic_loss, var_list=self.critic.trainable_weights)

    self.actor_train_op = tf.train.AdamOptimizer(
        learning_rate=self._hparams.actor_lr, name="actor_optimizer").minimize(
            self.actor_loss, var_list=self.actor.trainable_weights)

    self._build_target_update_op()

  def update(self, worker_id=0):
    if self._hparams.local_step[worker_id] == 0:
      self.target_actor.set_weights(self.actor.get_weights())
      self.target_critic.set_weights(self.critic.get_weights())

    memory = self._memory[worker_id]

    if self._hparams.local_step[worker_id] == 0:
      # make sure target and source models are the same in the beginning
      self.target_actor.set_weights(self.actor.get_weights())
      self.target_critic.set_weights(self.critic.get_weights())

    if self._hparams.training and memory.size() >= self._hparams.batch_size:
      _, _, last_states, actions, rewards, done, states = memory.sample(
          self._hparams.batch_size)

      critic_loss, _ = self._sess.run(
          [self.critic_loss, self.critic_train_op],
          feed_dict={
              self.last_states: last_states,
              self.actions: np.reshape(actions,
                                       (-1, self._hparams.num_actions)),
              self.done: np.expand_dims(done.astype(float), axis=-1),
              self.rewards: np.expand_dims(rewards.astype(float), axis=-1),
              self.states: states
          })

      actor_loss, _ = self._sess.run([self.actor_loss, self.actor_train_op],
                                     feed_dict={self.last_states: last_states})

      log_scalar("loss/actor/worker_%d" % worker_id, actor_loss)
      log_scalar("loss/critic/worker_%d" % worker_id, critic_loss)

      self.update_target()
