import torch
import torch.nn as nn


# Linear Embedding
class GraphEmbedding(nn.Module):
    def __init__(self, input_size, embedding_size):
        super(GraphEmbedding, self).__init__()
        self.embedding_size = embedding_size
        self.embedding = nn.Linear(input_size, embedding_size)

    def forward(self, inputs):
        return self.embedding(inputs)


# Attention/Pointer module using Bahanadu Attention
class Attention(nn.Module):
    def __init__(self, hidden_size, C=10):
        super(Attention, self).__init__()
        self.C = C
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.W_v = nn.Linear(hidden_size, 1)

    def forward(self, query, target):
        """
        Args:
            query: [batch_size x hidden_size]
            target:   [batch_size x seq_len x hidden_size]

        Return:
            target: [batch_size x seq_len x hidden_size]
            logits: [batch_size x seq_len]
        """
        batch_size, seq_len, _ = target.shape
        query = self.W_q(query).unsqueeze(1).repeat(1, seq_len, 1)  # [batch_size x seq_len x hidden_size]
        target = self.W_k(target)  # [batch_size x seq_len x hidden_size]
        logits = self.W_v(torch.tanh(query + target)).squeeze(-1)
        logits = self.C * torch.tanh(logits)
        return target, logits
