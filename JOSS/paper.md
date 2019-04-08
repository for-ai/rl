---
title: 'rl: A Modular codebase for reinforcement learning models training, testing and visualization.'
tags:
  - Python
  - Reinforcement Learning
  - Tensorflow
  - Q-Learning
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

Reinforcement learning refers to a group of methods from artificial intelligence where an agent performs learning through trial and error [@Sutton.1998]. It differs from machine learning, since it requires no explicit labels; instead, the agent interacts continuously with its environment. That is, the agent starts in a specific state and then performs an action, based on which it transitions to a new state and, depending on the outcome, receives a reward. Different strategies (e.g. Policy Gradients) have been proposed to maximize the overall reward, resulting in a so-called policy, which defines the best possible action in each state. 

# Functionality

The *rl* package utilizes different mechanisms for reinforcement learning, including Q-learning, Policy Gradients, Proximal Policy Optimization, Deterministic Policy Gradient. It thereby learns an optimal policy based on past experience in the form of sample sequences consisting of states, actions and rewards. 

# References
