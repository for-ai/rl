import tensorflow as tf

from .utils import *
from ..agent import Agent
from ..registry import register
from ...utils.logger import log_scalar
from ...models.registry import get_model


@register
class PPO(Agent):
  """ Proximal Policy Optimization """

  def __init__(self, sess, hparams):
    super().__init__(sess, hparams)
    self.actor = get_model(hparams, register="PPOActor", name="actor")
    self.critic = get_model(hparams, register="PPOCritic", name="critic")
    self.target_actor = get_model(
        hparams, register="PPOActor", name="target_actor")

    self.build()

  def act(self, state, worker_id):
    action_distribution = self._sess.run(
        self.probs, feed_dict={self.last_states: state[None, :]})
    return self._action_function(self._hparams, action_distribution, worker_id)

  def observe(self, last_state, action, reward, done, state, worker_id=0):
    action = one_hot(action, self._hparams.num_actions)

    self._memory[worker_id].add_sample(last_state, action, reward,
                                       self._hparams.gamma, done, state)

    if done:
      self.update(worker_id)

  def _build_target_update_op(self):
    with tf.variable_scope("update_target_network"):
      self.target_update_op.extend(
          copy_variables_op(source=self.actor, target=self.target_actor))

  def build(self):
    self.last_states = tf.placeholder(
        tf.float32, [None] + self._hparams.state_shape, name="states")
    self.rewards = tf.placeholder(tf.float32, [None], name="rewards")
    self.actions = tf.placeholder(
        tf.int32, [None, self._hparams.num_actions], name="actions")

    last_states = self.process_states(self.last_states)

    self.logits = self.actor(last_states)

    self.probs = tf.nn.softmax(self.logits, -1)

    target_logits = self.target_actor(last_states)

    discounted_reward = compute_discounted_rewards(self._hparams, self.rewards)

    values = self.critic(last_states)

    advantage = discounted_reward - values

    losses, train_ops = self._grad_function(
        logits={
            "target_logits": target_logits,
            "logits": self.logits
        },
        actions=self.actions,
        advantage=advantage,
        hparams=self._hparams,
        var_list={
            "actor_vars": self.actor.trainable_weights,
            "critic_vars": self.critic.trainable_weights
        })

    self.actor_loss = losses['actor_loss']
    self.critic_loss = losses['critic_loss']
    self.actor_train_op = train_ops['actor_train_op']
    self.critic_train_op = train_ops['critic_train_op']

    self._build_target_update_op()

  def update(self, worker_id=0):
    memory = self._memory[worker_id]
    if self._hparams.training:

      _, _, last_states, actions, rewards, _, _ = memory.sample(
          self._hparams.batch_size)

      for _ in range(self._hparams.num_actor_steps):
        actor_loss, _ = self._sess.run(
            [self.actor_loss, self.actor_train_op],
            feed_dict={
                self.last_states: last_states,
                self.actions: actions,
                self.rewards: rewards,
            })
      log_scalar("loss/actor/worker_%d" % worker_id, actor_loss)

      for _ in range(self._hparams.num_critic_steps):
        critic_loss, _ = self._sess.run(
            [self.critic_loss, self.critic_train_op],
            feed_dict={
                self.last_states: last_states,
                self.actions: actions,
                self.rewards: rewards,
            })
      log_scalar("loss/critic/worker_%d" % worker_id, critic_loss)

      memory.clear()

      self.update_target()
