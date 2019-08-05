========
Tutorial
========

Before you run a full example, it would be to your benefit to install the following:

- Nvidia CUDA on machines with GPUs to enable faster training. Installation instructions `here <https://developer.nvidia.com/cuda-downloads>`_
- Tensorboard for training visualization. Install by running `pip install tensorboard`
This tuturial will make use of a Conda environment as the preferred package manager. Installation instructions can be found `here <https://docs.conda.io/projects/conda/en/latest/user-guide/install/>`_

After installing Conda, create and activate an environment, and install all dependencies within that environment::

	$ conda create -n rl-codebase python=3.6
	$ conda activate rl-codebase
	$ pip install -r requirements.txt

To run locally, we will train DQN on the `Carpole-v1` Gym environment::

	$ # start training
	$ python train.py --sys local --hparams dqn_cartpole --output_dir /tmp/rl-testing
	$ # run tensorboard
	$ tensorboard --logdir /tmp/rl-testing
	$ # test agent
	$ python train.py --sys local --hparams dqn_cartpole --output_dir /tmp/rl-testing --training False --render True
