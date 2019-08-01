from rl.models.models.ppo import PPOActor, PPOCritic
from rl.hparams.utils import HParams

import numpy as np
import tensorflow as tf

inp_dim = 80


class PPOTest(tf.test.TestCase):

    def test_ppo_actor(self):
        hparams = HParams()
        hparams.hidden_size = 100
        hparams.num_actions = 10

        PPOActor_model = PPOActor(hparams)
        input_state = np.ones((inp_dim, inp_dim))

        output_state = PPOActor_model(input_state)
        self.assertAllEqual(output_state.shape, (inp_dim, hparams.num_actions))

    def test_ppo_critic(self):
        hparams = HParams()
        hparams.hidden_size = 100
        hparams.num_actions = 10

        PPOActor_model = PPOCritic(hparams)
        input_state = np.ones((inp_dim, inp_dim))

        output_state = PPOActor_model(input_state)
        self.assertAllEqual(output_state.shape, (inp_dim, 1))

if __name__ == '__main__':
    tf.test.main()
