from rl.models.models.ddpg import DDPGActor, DDPGCritic
from rl.hparams.utils import HParams

import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

inp_dim = 80

class DDPGTest(tf.test.TestCase):

    def test_ddpg_actor_box(self):
        hparams = HParams()
        hparams.action_space_type = "Box"
        hparams.hidden_size = 25
        hparams.num_actions = 10
        hparams.action_low = np.zeros(hparams.num_actions)
        hparams.action_high = np.ones(hparams.num_actions)

        DDPGActor_model = DDPGActor(hparams)
        input_state = np.ones((inp_dim, inp_dim))

        output_state = DDPGActor_model(input_state)
        self.assertAllEqual(output_state.shape,
                            (inp_dim, hparams.num_actions))

    def test_ddpg_actor_discrete(self):
        hparams = HParams()
        hparams.action_space_type = "Discrete"
        hparams.hidden_size = 25
        hparams.num_actions = 10

        DDPGActor_model = DDPGActor(hparams)
        input_state = np.ones((inp_dim, inp_dim))

        output_state = DDPGActor_model(input_state)
        self.assertAllEqual(output_state.shape,
                            (inp_dim, hparams.num_actions))

    def test_ddpg_critic_box(self):
        hparams = HParams()
        hparams.action_space_type = "Box"
        hparams.hidden_size = 25
        hparams.num_actions = 10

        DDPGCritic_model = DDPGCritic(hparams)
        input_state = np.ones((inp_dim, inp_dim))
        input_actions = np.ones((inp_dim, hparams.num_actions))

        output_state = DDPGCritic_model(input_state, input_actions)
        self.assertAllEqual(output_state.shape, (inp_dim, 1))

    def test_ddpg_critic_box(self):
        hparams = HParams()
        hparams.action_space_type = "Discrete"
        hparams.hidden_size = 25
        hparams.num_actions = 10

        DDPGCritic_model = DDPGCritic(hparams)
        input_state = np.ones((inp_dim, inp_dim))
        input_actions = np.ones((inp_dim, hparams.num_actions))

        output_state = DDPGCritic_model(input_state, input_actions)
        self.assertAllEqual(output_state.shape,
                            (inp_dim, hparams.num_actions))

if __name__ == '__main__':
    tf.test.main()
