import torch.nn as nn
from pointer_network import PointerNetwork
import torch.nn.functional as F

class CriticNetwork(nn.Module):
    def __init__(self,
                 embedding_size,
                 hidden_size,
                 seq_len,
                 n_glimpses,
                 tanh_exploration):
        super(CriticNetwork, self).__init__()

        self.ptrNet = PointerNetwork(embedding_size, hidden_size, seq_len, n_glimpses, tanh_exploration)
        self.fc1 = nn.Linear(seq_len, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        """
        Args:
            x: [batch_size x seq_len x 2]
        """
        probs, _ = self.ptrNet(x)
        x = self.fc1(probs)
        x = self.fc2(x)

        return x
