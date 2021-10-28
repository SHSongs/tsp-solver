import math
import random

import torch
import torch.nn as nn
from torch.distributions import Categorical

from module import GraphEmbedding, Attention


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

        self.batch_size = None
        self.prev_chosen_logprobs = None
        self.prev_chosen_indices = None
        self.encoder_outputs = None
        self.decoder_input = None
        self.hidden = None
        self.context = None
        self.embedded = None
        self.pass_prepare = False

    def prepare(self, x):
        self.batch_size = x.shape[0]
        self.embedded = self.embedding(x)

        self.encoder_outputs, (self.hidden, self.context) = self.encoder(self.embedded)
        self.decoder_input = self.decoder_start_input.unsqueeze(0).repeat(self.batch_size, 1)

        # init chosen
        self.prev_chosen_logprobs = []
        self.prev_chosen_indices = []
        self.pass_prepare = True

    def one_step(self, visited):
        assert self.pass_prepare is True, 'execute prepare func'

        visited = torch.Tensor(visited).unsqueeze(0)
        mask = visited > 0
        _, (self.hidden, self.context) = self.decoder(self.decoder_input.unsqueeze(1), (self.hidden, self.context))

        query = self.hidden.squeeze(0)
        for _ in range(self.n_glimpses):
            ref, logits = self.glimpse(query, self.encoder_outputs)
            _mask = mask.clone()
            logits[_mask] = -100000.0

            ref = ref.transpose(-1, -2)  # [batch_size x hidden_size x seq_len]
            logits_softmax = torch.softmax(logits, dim=-1).unsqueeze(-1)
            query = torch.matmul(ref, logits_softmax).squeeze(-1)  # [batch_size x seq_len]

        _, logits = self.pointer(query, self.encoder_outputs)

        _mask = mask.clone()
        logits[_mask] = -100000.0
        probs = torch.softmax(logits, dim=-1)
        cat = Categorical(probs)
        chosen = cat.sample()
        log_probs = cat.log_prob(chosen)

        return log_probs, chosen

    def sample_action(self, visited, epsilon):
        assert self.pass_prepare is True, 'execute prepare func'

        log_probs, chosen = self.one_step(visited)

        coin = random.random()
        if coin < epsilon:
            cities = list(range(0, visited.shape[0]))

            for i in self.prev_chosen_indices:
                cities.remove(i)

            chosen = random.choice(cities)

        self.prev_chosen_logprobs.append(log_probs)
        self.prev_chosen_indices.append(chosen)

        return log_probs, chosen

    def result(self):
        assert len(self.prev_chosen_logprobs) > 0, 'execute prepare, one step func'

        return torch.stack(self.prev_chosen_logprobs, 1), torch.stack(self.prev_chosen_indices, 1)

    def forward(self, x):
        """
        Args:
            x: [batch_size x seq_len x 2]
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
