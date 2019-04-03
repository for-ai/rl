## [FOR.ai](https://for.ai) Reinforcement Learning Codebase
Modular codebase for reinforcement learning models training, testing and visualization.

### Features
- Agents: [DQN](rl/agents/algos/dqn.py), [Vanilla Policy Gradient](rl/agents/algos/vanilla_pg.py), [DDPG](rl/agents/algos/ddpg.py), [PPO](rl/agents/algos/ppo.py)
- OpenAI Gym integration
  - support both `Discrete` and `Box` environments
  - Render (`--render`) and save (`--record_video`) environment replay
- Model-free asynchronous training  (`--num_workers`)
- Memory replay: [Simple](rl/memory/memory.py), [Proportional Prioritized Experience Replay](rl/memory/prioritized.py)
- Modularized hyper-parameters setting (`--hparams` and [hparams.py](rl/hparams/defaults.py))
- Modularized action functions ([action functions](rl/agents/algos/action_function/basic.py)) and gradient update functions ([compute gradient](rl/agents/algos/compute_gradient/basic.py))
### Requirements
- TensorFlow
- OpenAI Gym
    - Atari `pip install 'gym[atari]'`
- FFmpeg (`apt install ffmpeg` on Linux or `brew install ffmpeg` on macOS)

### Quick Start
```
# start training
python train.py --sys ... --hparams ... --output_dir ...
# run tensorboard
tensorboard --logdir ...
# test agnet
python train.py --sys ... --hparams ... --output_dir ... --training False --render True
```

### Hyper-parameters
Check [init_flags()](https://github.com/for-ai/rl/blob/master/train.py#L17), [defaults.py](rl/hparams/defaults.py) for default hyper-parameters, and check [hparams/dqn.py](rl/hparams/dqn.py) agent specific hyper-parameters examples.
- `hparams`: Which hparams to use, defined under [rl/hparams](rl/hparams)
- `sys`: Which system environment to use.
- `env`: Which RL environment to use.
- `output_dir`: The directory for model checkpoints and TensorBoard summary.
- `train_steps`:, Number of steps to train the agent.
- `test_episodes`: Number of episodes to test the agent.
- `eval_episodes`: Number of episodes to evaluate the agent.
- `training`: train or test agent.
- `copies`: Number of independent training/testing runs to do.
- `render`: Render game play.
- `record_video`: Record game play.
- `num_workers`, number of workers.
