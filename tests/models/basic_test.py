from rl.models.models.basic import basic, CNN
from rl.hparams.utils import HParams

import numpy as np
import tensorflow as tf

inp_dim = 80


class BasicTest(tf.test.TestCase):

    def test_basic_model(self):
        hparams = HParams()
        hparams.hidden_size = 100
        hparams.num_actions = 10

        basic_model = basic(hparams)
        input_state = np.ones((inp_dim, inp_dim))

        output_state = basic_model(input_state)
        self.assertAllEqual(output_state.shape, (inp_dim, hparams.num_actions))

    def test_CNN_model(self):
        hparams = HParams()
        hparams.state_latent_size = 10

        num_examples = 25
        inp_channels = 1

        CNN_model = CNN(hparams)
        input_state = np.ones((num_examples, inp_dim, inp_dim,
                               inp_channels))

        output_state = CNN_model(input_state)
        self.assertAllEqual(output_state.shape,
                            (num_examples, hparams.state_latent_size))

if __name__ == '__main__':
    tf.test.main()
