import tensorflow as tf

from .registry import register


@register
def mean_squared_error(preds, targets, hparams, weights=1.0, var_list=None):
  loss = tf.losses.mean_squared_error(
      predictions=preds, labels=targets, weights=weights)

  train_op = tf.train.AdamOptimizer(
      learning_rate=hparams.learning_rate).minimize(
          loss, var_list=var_list)
  return loss, train_op


@register
def huber_loss(preds, targets, hparams, weights=1.0, var_list=None):
  loss = tf.losses.huber_loss(
      predictions=preds, labels=targets, weights=weights)

  train_op = tf.train.AdamOptimizer(
      learning_rate=hparams.learning_rate).minimize(
          loss, var_list=var_list)
  return loss, train_op


@register
def policy_gradient(logits, actions, discounted_rewards, hparams,
                    var_list=None):
  ce = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=actions)
  loss = tf.reduce_mean(ce * discounted_rewards)

  train_op = tf.train.AdamOptimizer(
      learning_rate=hparams.learning_rate).minimize(
          loss, var_list=var_list)
  return loss, train_op


@register
def ppo(logits, actions, advantage, hparams, var_list):
  '''
  logits: A dict containing logits corresponding to target and current policies
  var_list: A dict containing trainable variables of the actor and critic
  '''

  target_logits, logits = logits['target_logits'], logits['logits']
  critic_loss = tf.reduce_mean(tf.square(advantage))

  def log_probs(prob_dist):
    return -tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=prob_dist, labels=actions)

  ratio = tf.exp(log_probs(logits) - log_probs(target_logits))

  clipped_ratio = tf.clip_by_value(ratio, 1 - hparams.clipping_coef,
                                   1 + hparams.clipping_coef)
  surrogate_objective = tf.minimum(clipped_ratio * advantage, ratio * advantage)

  actor_loss = -tf.reduce_mean(surrogate_objective)

  actor_train_op = tf.train.AdamOptimizer(hparams.actor_lr).minimize(
      actor_loss, var_list=var_list['actor_vars'])
  critic_train_op = tf.train.AdamOptimizer(hparams.critic_lr).minimize(
      critic_loss, var_list=var_list['critic_vars'])

  return {
      "actor_loss": actor_loss,
      "critic_loss": critic_loss
  }, {
      "actor_train_op": actor_train_op,
      "critic_train_op": critic_train_op
  }
