========
Usage
========

To start training::

	$ python train.py --sys ... --hparams ... --output_dir ...

Run tensorboard to visualize training::

	$ tensorboard --logdir ...

Test agent::

	$ python train.py --sys ... --hparams ... --output_dir ... --training False --render True

The sys command can be one of two options: ``local`` or ``tpu`` for GCP enabled tpu training. A list of environments and hyperparameters can be found under ``rl/rl/hparams``. A full training and evaluation example can be found in the tutorial section.

Below we summerize the key arguments ::

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
