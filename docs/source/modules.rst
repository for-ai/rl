Modules
========

This section covers the main components of the RL codebase.

Agents
------

All agents are encapsulated into a single class, ``Agent()``. An agent object is instantiated with the following member variables:

- TensorFlow session
- Hyperparameters
- Memory
- Action function
- Gradient function


Algos
------

This codebase currently supports the following RL algorithms: ``DDPG``, ``DQN``, ``PPO``, and ``Vanilla Policy Gradient``.

Environments
--------

The codebase has a single environment class, and two environment subclasses:

- ``class Environment``
	- ``class CoinRun(Environment)`` as described `here <https://github.com/openai/coinrun/>`_
	- ``class GymEnv(Environment)`` as descibed `here <https://jair.org/index.php/jair/article/view/10819/25823/>`_ 

The ``Environment`` class, located at ``rl/rl/envs/env.py``, acts as a template contains all the member variables and functions of a standard RL environment. The specific member variables and functions are implemented within the two subclasses.


Hyperparameters
------------

The class to represent hyperparameters, ``HParams``, is located at ``rl/rl/hparams/utils.py``. The default set of hyperparameters located at ``rl/rl/hparams/default.py``.

The list of hyperparameters is located at ``rl/rl/hparams``, segmented into the different RL algorithms each set of hyperparameters belongs to. Each RL algorithms calls the default set of hyperparameters, and depending on different environments, modifies the default set.

For example, the function ``ddpg()`` returns a default set of ``HParams`` and adds a number of fields, while the 
function ``ddpg_cartpole()`` further modifies some of the hyperparameter fields to fit with the ``Cartpole-v1`` Gym environment


Memory
-------

The base class for memory is located at ``rl/rl/memory/memory.py``. This codebase currently supports three memory types: simple experience reply, sum tree, and prioritized experience replay.


Models
-------

The list of models can be found at ``rl/rl/models/models``. All models are subclassed from Keras models. Each model is instantiated with a network structure and a ``call()`` function to call the forward pass of the network. All layers are taken from the TensorFlow Layers class. 

