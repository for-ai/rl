from rl.models.models.noisy_network import NoisyLayer, NoisyNetwork
from rl.hparams.utils import HParams

import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

inp_dim = 80


class NoisyNetworkTest(tf.test.TestCase):

    def test_noisy_layer(self):
        hparams = HParams()
        units = 10
        activation = tf.nn.relu

        NoisyLayer_model = NoisyLayer(hparams, units, activation)
        input_state = np.ones((inp_dim, inp_dim), dtype="float32")

        output_state = NoisyLayer_model(input_state)
        self.assertAllEqual(output_state.shape, (inp_dim, units))

    def test_noisy_network(self):
        hparams = HParams()
        hparams.hidden_size = 100
        hparams.num_actions = 10

        NoisyNetwork_model = NoisyNetwork(hparams)
        input_state = np.ones((inp_dim, inp_dim), dtype="float32")

        output_state = NoisyNetwork_model(input_state)
        self.assertAllEqual(output_state.shape, (inp_dim, hparams.num_actions))

if __name__ == '__main__':
    tf.test.main()
