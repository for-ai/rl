---
title: 'RL: Generic reinforcement learning codebase in TensorFlow'
tags:
  - Reinforcement Learning
  - Tensorflow
  - Gym
authors:
  - name: Bryan M. Li
    affiliation: 1
  - name: Alexander Cowen-Rivers
    affiliation: 1
  - name: Piotr Kozakowski
    affiliation: 1
  - name: David Tao
    affiliation: 1
  - name: Siddhartha Rao Kamalakara
    affiliation: 1
  - name: Nitarshan Rajkumar
    affiliation: 1
  - name: Hariharan Sezhiyan
    affiliation: 1
  - name: Sicong Huang
    affiliation: 1
  - name: Aidan N. Gomez
    affiliation: 1  
affiliations:
 - name: FOR.ai
   index: 1
date: 12 June 2019
bibliography: paper.bib
output:
  html_document:
    keep_md: TRUE
---

# Abstract 

Vast reinforcement learning (RL) research groups, such as DeepMind and OpenAI, have their internal (private) reinforcement learning codebases, which enable quick prototyping and comparing of ideas to many SOTA methods. We argue the five fundamental properties of a sophisticated research codebase are; modularity, reproducibility, many RL algorithms pre-implemented, speed and ease of running on different hardware/ integration with visualization packages. 
Currently, there does not exist any RL codebase, to the author's knowledge, which contains all the five properties, particularly with TensorBoard logging and abstracting away cloud hardware such as TPU's from the user. The codebase aims to help distil the best research practices into the community as well as ease the entry access and accelerate the pace of the field. More detailed documentation can be found [here](https://rl-codebase.readthedocs.io/en/latest/).


# Related Work

There are currently various implementations available for reinforcement learning codebase like OpenAI baselines[@dhariwal:2017],  Stable baselines [@hill2019], Tensorforce[@schaarschmidt:2017], Ray rllib[@liang:2017], Intel Coach[@caspi:2017], Keras-RL [@kerasrl:2019], Dopamine baselines [@castro:2018] and TF-Agents [@guadarramatf]. Ray rllib [@liang:2017] is amongst the strongest of existing RL frameworks, supporting; distributed operations, TensorFlow[@abadi:2016], PyTorch[@paszke:2017] and multi-agent reinforcement learning (MARL). Unlike Ray rllib, we choose to focus on Tensorflow support, allowing us to integrate specific framework visualisation and experiment tracking into our codebase. On top of this, we are developing a Kuberenetes script for MacOS and Linux users to connect to any cloud computing platform, such as Google TPU’s, Amazon AWS etc. Most other frameworks are plagued with problems like usability issues (difficult to get started and increment over), very little modularity in code (no/ little hierarchy and code reuse), no asynchronous training support, weak support for TensorBoard logging and so on. All these problems are solved by our project, which is a generic codebase built for reinforcement learning (RL) research in Tensorflow[@schaarschmidt:2017], with favoured RL agents pre-implemented as well as integration with OpenAI Gym[@brockman:2016] environment focusing on quick prototyping and visualisation.

Deep Reinforcement Learning 
Reinforcement learning refers to a paradigm in artificial intelligence where an agent performs a sequence of actions in an environment to maximise rewards[@sutton:1998]. It is in many ways more general and challenging than supervised learning since it requires no labels to train on; instead, the agent interacts continuously with the environment, gathering more and more data and guiding its learning process.

# Introduction: for-ai/rl 
Further to the core ideas mentioned in the beginning, a good research codebase should enable good development practices such as continually checkpointing the model's parameters as well as instantly restoring them to the latest checkpoint when available. Moreover, it should be composed of simple, interchangeable building blocks, making it easy to understand and to prototype new research ideas.

We will first introduce the framework for this project, and then we will detail significant components. Lastly, we will discuss how to get started with training an agent under this framework.

This codebase allows training RL agents by a training script as simple as the below for loop.

```python
for epoch in range(epochs):
    state = env.reset()
	for step in range(max_episode_steps):
        last_state = state
        action = agent.act(state)
        state, reward, done = env.step(action)
        agent.observe(last_state, action, reward, state)
        agent.update()
```

To accomplish this, we chose to modularise the codebase in the hierarchy shown below. 

```
rl_codebase
|- train.py
|---> memory
|   |- registry.py
|---> hparams
|   |- registry.py
|---> envs
|   |- registry.py
|---> models
|   |- registry.py
|---> agents
|   |- registry.py
|   |---> algos
|   |   |- registry.py
|   |   |---> act_select
|   |   |   |- registry.py
|   |   |---> grad_comp
|   |   |   |- registry.py
```

Our modularisation enables simple and easy-to-read implementation of each component, such as the Agent, Algo and Environment class, as shown below.

```python
class Agent:
	self.model: Model
	self.algo: Algo

	def observe(last_state, action, reward, new_state)
	def act(state) -> action
	def update()

class Algo(Agent):
	def select_action(distribution) -> action
	def compute_gradients(trajectory, parameters) -> gradients

class Environment:
	def reset() -> state
	def step(action) -> state, reward, done
```

The codebase includes agents like Deep Q Network[@mnih:2013], Noisy DQN[@plappert:2017], Vanilla Policy Gradient[@sutton:2000], Deep Deterministic Policy Gradient[@silver2014deterministic] and Proximal Policy Optimization[@schulman2017proximal]. The project also includes simple random sampling and proportional prioritized experience replay approaches, support for Discrete and Box environments, option to render environment replay and record the replay in a video. The project also gives the possibility to conduct model-free asynchronous training, setting hyperparameters for your algorithm of choice, modularized action and gradient update functions and option to show your training logs in a TensorBoard summary.

In order to run an experiment, run: 

`python train.py --sys ... --hparams ... --output_dir .... `

Ideally, “train.py” should never need to be modified for any of the typical single agent environments. It covers the logging of reward, checkpointing, loading, rendering environment/ dealing with crashes and saving the experiments hyperparameters, which takes a significant workload off the average reinforcement learning researcher. 

```
“--sys”(str) defines the system chosen to run experiment with;  e.g. “local” for running on the local machine. 
“--env”(str) specifies the environment. 
“--hparam_override”(str) overrides hyperparameters. 
“--train_steps”(int) sets training length. 
“--test_episodes”(int) tests episodes.
“--eval_episodes”(int) sets Validation episodes.
“--training"(bool) freeze model weights is set to False. 
“--copies”(int) set the number of times to perform multiple versions of training/ testing.
“--render”(bool) turns rendering on/ off. 
“--record_video”(bool) records the video with, which outputs a .mp4 of each recorded episode.
“--num_workers"(int) seamlessly brings our synchronous agent into an asynchronous agent.
```

# Full Example

Before you run a full examples, it would be to your benefit to install the following:

* Nvidia CUDA on machines with GPUs to enable faster training. Installation instructions [here](https://developer.nvidia.com/cuda-downloads)
* Tensorboard for training visualization. Install by running `pip install tensorboard`

This tuturial will make use of a Conda environment as the preferred package manager. Installation instructions can be found [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).

After installing Conda, create and activate an environment, and install all dependencies within that environment:

```
conda create -n rl python=3.6
conda activate rl
sh setup.sh
```

To run locally, we will train DQN on the `Carpole-v1` Gym environment:

```
# start training
python train.py --sys local --hparams dqn_cartpole --output_dir /tmp/rl-testing
# run tensorboard
tensorboard --logdir /tmp/rl-testing
# test agent
python train.py --sys local --hparams dqn_cartpole --output_dir /tmp/rl-testing --training False --render True
```


# Conclusion
We have outlined the benefits of using a highly modularised reinforcement learning codebase. The next stages of development for the RL codebase are implementing more SOTA model-free RL techniques (GAE, Rainbow, SAC, IMPALA), introducing model-based approaches, such as World Models[@ha:2018], integrating into an open-sourced experiment managing tool and expanding the codebases compatibility with a broader range of environments, such as Habitat [@savva:2019]. We would also like to see automatic hyperparameter optimization techniques to be integrated, such as Bayesian Optimization method which was crucial to the success of some of DeepMinds most considerable reinforcement learning feats[@chen:2018].

# Acknowledgements

We would like to thank all other members of For.ai, for useful discussions and feedback. 

# References


