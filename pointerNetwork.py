import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Categorical

import math


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


class PointerNetwork(nn.Module):
    def __init__(self,
                 embedding_size,
                 hidden_size,
                 seq_len,
                 n_glimpses,
                 tanh_exploration):
        super(PointerNetwork, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_glimpses = n_glimpses
        self.seq_len = seq_len

        self.embedding = GraphEmbedding(2, embedding_size)
        self.encoder = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.pointer = Attention(hidden_size, C=tanh_exploration)
        self.glimpse = Attention(hidden_size)

        self.decoder_start_input = nn.Parameter(torch.FloatTensor(embedding_size))
        self.decoder_start_input.data.uniform_(-(1. / math.sqrt(embedding_size)), 1. / math.sqrt(embedding_size))

    def forward(self, x):
        """
        Args:
            param x: [batch_size x seq_len x 2]
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        embedded = self.embedding(x)

        encoder_outputs, (hidden, context) = self.encoder(embedded)

        prev_chosen_logprobs = []
        prev_chosen_indices = []
        mask = torch.zeros(batch_size, self.seq_len, dtype=torch.bool)

        decoder_input = self.decoder_start_input.unsqueeze(0).repeat(batch_size, 1)
        for idx in range(seq_len):
            _, (hidden, context) = self.decoder(decoder_input.unsqueeze(1), (hidden, context))

            query = hidden.squeeze(0)
            for _ in range(self.n_glimpses):
                ref, logits = self.glimpse(query, encoder_outputs)
                _mask = mask.clone()
                logits[_mask] = -100000.0

                ref = ref.transpose(-1, -2)  # [batch_size x hidden_size x seq_len]
                logits_softmax = torch.softmax(logits, dim=-1).unsqueeze(-1)
                query = torch.matmul(ref, logits_softmax).squeeze(-1)  # [batch_size x seq_len]

            _, logits = self.pointer(query, encoder_outputs)

            _mask = mask.clone()
            logits[_mask] = -100000.0
            probs = torch.softmax(logits, dim=-1)
            cat = Categorical(probs)
            chosen = cat.sample()
            mask[[i for i in range(batch_size)], chosen] = True
            log_probs = cat.log_prob(chosen)

            # chosen(next city)[batch_size x 1 x hidden_size ] 의 값으로 embedded[batch_size x seq_len x hidden_size 를 같게
            tmp_chosen = chosen[:, None, None].repeat(1, 1, self.hidden_size)
            decoder_input = embedded.gather(1, tmp_chosen).squeeze(1)

            prev_chosen_logprobs.append(log_probs)
            prev_chosen_indices.append(chosen)

        return torch.stack(prev_chosen_logprobs, 1), torch.stack(prev_chosen_indices, 1)
