---
title: 'rl: A Modular codebase for reinforcement learning research'
tags:
  - Reinforcement Learning
  - Tensorflow
  - DQN
  - Policy Gradient
  - DDPG
  - PPO
authors:
  - name: Bryan M. Li
    affiliation: 1
  - name: David Tao
    affiliation: 1
  - name: Alexander Cowen-Rivers
    affiliation: 1
  - name: Siddhartha Rao Kamalakara
    affiliation: 1
  - name: Nitarshan Rajkumar
    affiliation: 1
  - name: Sourav Singh
    affiliation: 1
  - name: Aidan N. Gomez
    affiliation: 1  
affiliations:
 - name: FOR.ai
   index: 1
 date: 8 April 2019
bibliography: paper.bib
output:
  html_document:
    keep_md: TRUE
---

# Summary

This is a generic codebase built for reinforcement learning (RL) research in [TensorFlow](https://tensorflow.org), with popular RL agents pre-implemented as well as integration with [OpenAI Gym](https://gym.openai.com/) environment focusing on quick prototyping and deployment.

Example for recorded envrionment on various RL agents.

| MountainCar-v0 |  Pendulum-v0 | VideoPinball-v0 | Tennis-v0 |
|---|---|---|---|
![MountainCar-v0](gif/mountaincar.gif)|![Pendulum-v0](gif/pendulum.gif)|![VideoPinball-v0](gif/pinball.gif)|![Tennis-v0](gif/tennis.gif)

# Functionality

The following is a list of implemented features in the RL codebase.
- Agents `hparams.agent`
  - Deep Q Networks (DQN)
  - Noisy DQN
  - Vanilla Policy Gradient
  - Deep Deterministic Policy Gradient (DDPG)
  - Proximal Policy Optimization (PPO)
- Memory `hparams.memory`
  - Simple random sampling
  - Proportional Prioritized Experience Replay
- OpenAI Gym integration `--env`
  - support both `Discrete` and `Box` environments
  - Render `--render` and record `--record_video` environment replay
- Model-free asynchronous training  `--num_workers`
- Modularized hyper-parameters setting `--hparams` and [hparams/defaults.py](rl/hparams/defaults.py)
- Modularized action functions [action functions](rl/agents/algos/action_function/basic.py)
- Modularized gradient update functions [compute gradient](rl/agents/algos/compute_gradient/basic.py)
- TensorBoard summary `tensorboard --logdir <output_dir>`


# References
