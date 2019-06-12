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
  - name: Sourav Singh
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

RL: Generic reinforcement learning codebase in TensorFlow
Bryan M. Li1, David Tao1, Alexander Cowen-Rivers1, Siddhartha Rao Kamalakara1, Nitarshan Rajkumar1, Aidan N. Gomez1
1FOR.ai


# Abstract 
Large reinforcement learning (RL) research groups, such as DeepMind and OpenAI, have their internal (private) reinforcement learning codebases, which enable quick prototyping and comparing of ideas to many SOTA methods. We argue the five fundamental properties of a sophisticated research codebase are; modularity, reproducibility, many RL algorithms pre-implemented, speed and ease of running on different hardware/ integration with visualization packages. 
Currently, there does not exist any RL codebase, to the authors knowledge, which contains all the five properties, particularly with TensorBoard logging and abstracting away cloud hardware such as TPU’s from the user. The aim of the codebase is to help distill the best research practices into the community as well as ease the entry access and accelerate the pace of the field. 

# Related Work

There are currently various implementations available for reinforcement learning codebase like OpenAI baselines[2],  Stable baselines [19], Tensorforce[3], Ray rllib[4], Intel Coach[5], Keras-RL [8], Dopamine baselines [9] and TF-Agents [18]. Ray rllib [4] is amongst the strongest of existing RL frameworks, supporting; distributed operations, TensorFlow, PyTorch and multi-agent reinforcement learning (MARL). Unlike Ray rllib, we choose to focus on Tensorflow support, allowing us to integrate specific framework visulisation and experiment tracking into our codeabse. Ontop of this, we are developing a Kuberenetes script for MacOS and Linux users to connect to any cloud computing platform, such as Google TPU’s, Amazon AWS etc. Most other frameworks are plagued with problems like usability issues (difficult to get started and increment over), very little modularity in code (no/ little hierarchy and code reuse), no asynchronous training support, weak support for TensorBoard logging and so on. All these problems are solved by our project, which is a generic codebase built for reinforcement learning (RL) research in Tensorflow[3], with popular RL agents pre-implemented as well as integration with OpenAI Gym[4] environment focusing on quick prototyping and visualization.

Deep Reinforcement Learning 
Reinforcement learning refers to a paradigm in artificial intelligence where an agent performs a sequence of actions in an environment to maximize rewards[1]. It is in many ways more general and challenging than supervised learning since it requires no labels to train on; instead, the agent interacts continuously with the environment, gathering more and more data and guiding its own learning process.

# Introduction: for-ai/rl 
Further to the core ideas mentioned in the beginning, a good research codebase should enable good development practices such as continually checkpointing the model's parameters as well as instantly restoring them to the latest checkpoint when available. Moreover, it should be composed of simple, interchangeable building blocks, making it easy to understand and to prototype new research ideas.

We will first introduce the framework for this project, then we will detail already implemented and significant components. Lastly, we will discuss we can begin training an agent under this framework. 

The aim of the codebase was to almost have a codebase as simple as the below for loop. 

for epoch in range(epochs):
state = env.reset()

	for step in range(max_episode_steps):
	last_state = state
action = agent.act(state)
state, reward, done = env.step(action)
	agent.observe(last_state, action, reward, state)

agent.update()

In order to successfully accomplish this, we chose to modularise the codebase in the hierarchy shown below. 

rl_codebase
|- train.py
|---> agents
|   |- registry.py
|   |---> models
|   |   |- registry.py
|   |---> algos
|   |   |- registry.py
|   |   |---> act_select
|   |   |   |- registry.py
|   |   |---> grad_comp
|   |   |   |- registry.py
|---> hparams
|   |- registry.py
|---> envs
|   |- registry.py


This enabled simple and easy to read implementations of each component, such as the Agent, Algo and Environment class is shown below. 

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


The project includes agents like Deep Q Network[10], Noisy DQN[11], Vanilla Policy Gradient[12], Deep Deterministic Policy Gradient[13] and Proximal Policy Optimization[14]. The project also includes simple random sampling and proportional prioritized experience replay approaches, support for Discrete and Box environments, option to render environment replay and record the replay in a video. The project also gives the option to conduct model-free asynchronous training, setting hyperparameters for your algorithm of choice, modularized action and gradient update functions and option to show your training logs in a TensorBoard summary.

In order to run an experiment, we run the below line. 

python train.py --sys ... --hparams ... --output_dir .... 

Where “train.py” should never need to be modified for any of the typical single agent environments. This already takes a large workload from the typical reinforcement learning researcher as this deals with logging of reward, checkpointing, loading, rendering environment/ dealing with crashes and saving the experiments hyperparameters. 

We define the system we choose to run this on with;
“--sys str” e.g. “local” for running on the local machine. 
Choose the environment with “--env str”
Override hyperparameters with “--hparam_override str”
Set training length “--train_steps int”
Test episodes “--test_episodes int”
Validation episodes “--eval_episodes int”
Freeze model weights “--training bool”
Performing multiple versions of training/ testing with “--copies int”
Turn rendering on/ off with “--render bool”
Record the video with “--record_video bool”, which ouputs a .mp4 of each recorded episode, soon to automatically generate a GIF. 
“--num_workers int”, which seamlessly brings our synchronous agent into an asynchronous agent.  

# Conclusion
We have outlined the benefits of using a highly modularised reinforcement learning codebase. The next stages of development for the RL codebase are implementing more SOTA model-free RL techniques (GAE, Rainbow, SAC, IMPALA), introducing model-based approaches (World Models), integrating into an open-sourced experiment managing tool and expanding the codebases compatibility with a broader range of environments, such as Habitat [15]. We would also like to see automatic hyperparameter optimization techniques to be integrated, such as Bayesian Optimisation method which was crucial to the success of some of DeepMinds largest reinforcement learning feats[16].

# References
[1] Sutton, Richard S., and Andrew G. Barto. Introduction to reinforcement learning. Vol. 135. Cambridge: MIT press, 1998.
[2] Dhariwal, Prafulla, et al. "Openai baselines." GitHub, GitHub repository (2017).
[3] Schaarschmidt, Michael, Alexander Kuhnle, and Kai Fricke. "Tensorforce: A tensorflow library for applied reinforcement learning." Web page (2017).
[4] Liang, Eric, et al. "Ray rllib: A composable and scalable reinforcement learning library." arXiv preprint arXiv:1712.09381 (2017).
[5] Caspi, Itai, et al. "Reinforcement Learning Coach (2017)." URL https://doi. org/10.5281/zenodo 1134899.
[6] Abadi, Martín, et al. "Tensorflow: A system for large-scale machine learning." 12th {USENIX} Symposium on Operating Systems Design and Implementation ({OSDI} 16). 2016.
[7] Brockman, Greg, et al. "Openai gym." arXiv preprint arXiv:1606.01540 (2016).
[8] Plappert, Matthias. “Keras-rl” (2016).  URL https://github.com/keras-rl/keras-rl
[9] Samuel Castro, Pablo et al. “Dopamine: A Research Framework for Deep Reinforcement Learning”. arXiv preprint arXiv:1812.06110 (2018). 
[10] Mnih, Volodymyr, et al. "Playing atari with deep reinforcement learning." arXiv preprint arXiv:1312.5602 (2013).
[11] Plappert, Matthias, et al. "Parameter space noise for exploration." arXiv preprint arXiv:1706.01905 (2017).
[12] Sutton, Richard S., et al. "Policy gradient methods for reinforcement learning with function approximation." Advances in neural information processing systems. 2000.
[13] Silver, David, et al. "Deterministic policy gradient algorithms." ICML. 2014.
[14] Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
[15] Savva, Manolis ,et al. “Habitat: A Platform for Embodied AI Research”. arXiv preprint arXiv:1904.01201 (2019). 
[16] Chen, Yutian, et al. "Bayesian optimization in alphago." arXiv preprint arXiv:1812.06855 (2018).
[17] Ha, David and Schmidhuber, Jurgen, et al. “World Models.” arXiv preprint arXiv:1803.10122
[18] Guadarrama, Sergio et al. “TF-Agents: A library for Reinforcement Learning in TensorFlow” (2018).  URL https://github.com/tensorflow/agents
[19] Hill, et al. “Stable Baselines”.  GitHub, GitHub repository (2018).



