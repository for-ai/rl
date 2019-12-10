import os
import random
import threading
import numpy as np
import argparse
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
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--hparams", required=True, type=str, help="Which hparams to use.")
  parser.add_argument(
      "--sys",
      required=True,
      type=str,
      choices=['local', 'gcp', 'tpu'],
      help="Which system environment to use.")
  parser.add_argument("--env", default="", help="Which RL environment to use.")
  parser.add_argument(
      "--hparam_override",
      default="",
      type=str,
      help="Run-specific hparam settings to use.")
  parser.add_argument(
      "--output_dir", required=True, type=str, help="The output directory.")
  parser.add_argument(
      "--train_steps",
      default=2000000,
      type=int,
      help="Number of steps to train the agent.")
  parser.add_argument(
      "--eval_episodes",
      default=10,
      type=int,
      help="Number of episodes to evaluate the agent.")
  parser.add_argument(
      "--test_episodes",
      default=10,
      type=int,
      help="Number of episodes to test the agent.")
  parser.add_argument(
      "--test_only", action="store_true", help="Test agent without training.")
  parser.add_argument(
      "--copies", default=1, type=int, help="Which hparams to use.")
  parser.add_argument("--render", action="store_true", help="Render game play.")
  parser.add_argument(
      "--record_video", action="store_true", help="Record game play.")
  parser.add_argument(
      "--num_workers", default=1, type=int, help="Number of workers.")

  FLAGS = parser.parse_args()
  return FLAGS


def init_random_seeds(hparams):
  tf.set_random_seed(hparams.seed)
  random.seed(hparams.seed)
  np.random.seed(hparams.seed)


def init_hparams(FLAGS):
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
  print("\n-----------------------------------------\n"
        "BEGINNING RUN #%s:\n"
        "\t hparams: %s\n"
        "\t env: %s\n"
        "\t agent: %s\n"
        "\t num_workers: %d\n"
        "\t output_dir: %s\n"
        "-----------------------------------------\n" %
        (run, FLAGS.hparams, hparams.env, hparams.agent, hparams.num_workers,
         hparams.output_dir))

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
  eval_env = get_env(hparams)

  state = env.reset()
  while hparams.global_step < hparams.train_steps:
    hparams.mode[worker_id] = ModeKeys.TRAIN

    last_state = state

    action, reward, done, state = step(hparams, agent, last_state, env,
                                       worker_id)

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
      evaluate(worker_id, agent, eval_env, hparams)
      if worker_id == 0:
        checkpoint.save()
      agent.reset(worker_id)

  env.close()
  eval_env.close()


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

      if not hparams.test_only:
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


def main():
  FLAGS = init_flags()
  _run(FLAGS)


if __name__ == "__main__":
  main()
