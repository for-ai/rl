import os
import random
import threading
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from rl.utils import flags
from rl.utils.utils import ModeKeys
from rl.utils.lr_schemes import update_learning_rate
from rl.envs.registry import get_env
from rl.utils.checkpoint import Checkpoint
from rl.utils.logger import init_logger, log_scalar, log_graph
from rl.hparams.registry import get_hparams
from rl.agents.registry import get_agent


def init_flags():
  tf.flags.DEFINE_string("hparams", None, "Which hparams to use.")
  tf.flags.DEFINE_string("sys", None, "Which system environment to use.")
  tf.flags.DEFINE_string("env", None, "Which RL environment to use.")
  tf.flags.DEFINE_string("hparam_override", "",
                         "Run-specific hparam settings to use.")
  tf.flags.DEFINE_string("output_dir", None, "The output directory.")
  tf.flags.DEFINE_integer("train_steps", 2000000,
                          "Number of steps to train the agent")
  tf.flags.DEFINE_integer("eval_episodes", 10,
                          "Number of episodes to evaluate the agent")
  tf.flags.DEFINE_integer('test_episodes', 10,
                          "Number of episodes to test the agent")
  tf.flags.DEFINE_boolean("training", True, "training or testing")
  tf.flags.DEFINE_integer("copies", 1,
                          "Number of independent training/testing runs to do.")
  tf.flags.DEFINE_boolean("render", False, "Render game play")
  tf.flags.DEFINE_boolean("record_video", False, "Record game play")
  tf.flags.DEFINE_integer("num_workers", 1, "number of workers")


def init_random_seeds(hparams):
  tf.set_random_seed(hparams.seed)
  random.seed(hparams.seed)
  np.random.seed(hparams.seed)


def init_hparams(FLAGS):
  flags.validate_flags(FLAGS)

  tf.reset_default_graph()

  hparams = get_hparams(FLAGS.hparams)
  hparams = hparams.parse(FLAGS.hparam_override)
  hparams = flags.update_hparams(FLAGS, hparams)

  return hparams


def init_agent(sess, hparams):
  # initialize environment to update hparams
  env = get_env(hparams)
  env.close()
  agent = get_agent(sess, hparams)
  checkpoint = Checkpoint(sess, hparams)
  return agent, checkpoint


def log_start_of_run(FLAGS, hparams, run):
  tf.logging.warn("\n-----------------------------------------\n"
                  "BEGINNING RUN #%s:\n"
                  "\t hparams: %s\n"
                  "\t env: %s\n"
                  "\t agent: %s\n"
                  "\t num_workers: %d\n"
                  "\t output_dir: %s\n"
                  "-----------------------------------------\n" %
                  (run, FLAGS.hparams, hparams.env, hparams.agent,
                   hparams.num_workers, hparams.output_dir))

  hparams.run_output_dir = os.path.join(hparams.output_dir, 'run_%d' % run)
  init_logger(hparams)


def step(hparams, agent, state, env, worker_id):
  """ run envrionment for one step and return the output """
  if hparams.render:
    env.render()

  action = agent.act(state, worker_id)
  state, reward, done, _ = env.step(action)

  if done:
    state = env.reset()

  return action, reward, done, state


def train(worker_id, agent, hparams, checkpoint):
  env = get_env(hparams)

  state = env.reset()
  while hparams.global_step < hparams.train_steps:
    hparams.mode[worker_id] = ModeKeys.TRAIN

    last_state = state

    action, reward, done, state = step(
        hparams, agent, last_state, env, worker_id)

    agent.observe(last_state, action, reward, done, state, worker_id)

    if done:
      hparams.local_episode[worker_id] += 1
      log_scalar('episodes/worker_%d' % worker_id,
                 hparams.local_episode[worker_id])

    hparams.global_step += 1
    hparams.total_step += 1
    hparams.local_step[worker_id] += 1
    update_learning_rate(hparams)

    if hparams.local_step[worker_id] % hparams.eval_interval == 0:
      agent.reset(worker_id)
      evaluate(worker_id, agent, env, hparams)
      if worker_id == 0:
        checkpoint.save()
      state = env.reset()
      agent.reset(worker_id)

  env.close()


def evaluate(worker_id, agent, env, hparams):
  hparams.mode[worker_id] = ModeKeys.EVAL
  rewards = []

  for i in range(hparams.eval_episodes):
    state = env.reset()
    done = False
    episode_reward = 0
    while not done:
      last_state = state
      action, reward, done, state = step(
          hparams, agent, last_state, env, worker_id=worker_id)
      episode_reward += reward
      hparams.total_step += 1
    rewards.append(episode_reward)

  log_scalar('rewards/worker_%d' % worker_id, np.mean(rewards))
  log_scalar('rewards_std/worker_%d' % worker_id, np.std(rewards))


def test(hparams, agent):
  hparams.mode[0] = ModeKeys.TEST
  env = get_env(hparams)

  for i in range(hparams.test_episodes):
    state = env.reset()
    done = False
    episode_reward = 0
    while not done:
      if hparams.render:
        env.render()
      last_state = state
      action, reward, done, state = step(
          hparams, agent, last_state, env, worker_id=0)
      episode_reward += reward
    print("episode %d\trewards %d" % (i, episode_reward))


def _run(FLAGS):
  hparams = init_hparams(FLAGS)
  init_random_seeds(hparams)

  for run in range(hparams.copies):
    log_start_of_run(FLAGS, hparams, run)

    with tf.Session() as sess:
      K.set_session(sess)
      agent, checkpoint = init_agent(sess, hparams)

      restored = checkpoint.restore()
      if not restored:
        sess.run(tf.global_variables_initializer())

      if hparams.training:
        log_graph()

        agent.clone_weights()

        if hparams.num_workers == 1:
          train(0, agent, hparams, checkpoint)
        else:
          workers = [
              threading.Thread(
                  target=train, args=(worker_id, agent, hparams, checkpoint))
              for worker_id in range(hparams.num_workers)
          ]

          for worker in workers:
            worker.start()

          for worker in workers:
            worker.join()
      else:
        test(hparams, agent)

    hparams = init_hparams(FLAGS)


def main(_):
  FLAGS = tf.app.flags.FLAGS
  _run(FLAGS)


if __name__ == "__main__":
  init_flags()
  tf.app.run()
