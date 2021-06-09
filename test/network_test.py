import os
import sys
import unittest

import torch

from pointer_network import PointerNetwork

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


class TestModel(unittest.TestCase):

    def test_pointer_network_in_out(self):
        embedding_size = 128
        hidden_size = 128
        n_glimpses = 2
        seq_len = 3
        tanh_exploration = 10

        model = PointerNetwork(embedding_size,
                               hidden_size,
                               seq_len,
                               n_glimpses,
                               tanh_exploration)

        log_probs, actions = model(torch.Tensor(3, seq_len, 2))

        self.assertTrue((3, seq_len) == log_probs.size())
        self.assertTrue((3, seq_len) == actions.size())


if __name__ == "__main__":
    unittest.main()
