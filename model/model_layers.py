"""
Model Layers:

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax


# Embedding layer:
# word vectors are embedded using GloVe embeddings
# projected using linear transoform and passed on to
# Highway Encoder to get final embedding
class EmbeddingLayer(nn.Module):

    # args:
    # word_vecs: pretrained word vectors
    # hidden_size: size of hidden activation
    # drop_prob: probability of activations going to zero.

    def __init__(self, word_vecs, hidden_size, drop_prop):
        super(EmbeddingLayer, self).__init__()
        self.drop_prop = drop_prop
        self.embeding = nn.Embedding.from_pretrained(word_vecs)
        self.proj = nn.Linear(word_vecs.size(1), hidden_size, bias=False)
        self.highway = HighwayEncoder(2, hidden_size)

    def forward(self, input):
        embed = self.embeding(input)
        embed = F.dropout(embed, self.drop_prop, self.training)
        embed = self.proj(embed)
        embed = self.highway(embed)

        return embed


# Highway Encoder:
# Encodes input embedding
class HighwayEncoder(nn.Module):
    # args:
    # num_layers: numbere of layers in highway encoder
    # hidden_size: size of hidden activations
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transf = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.gates = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])

    def forward(self, input):
        for gate, transform in zip(self.gates, self.tranf):
            g = torch.sigmoid(gate(input))
            t = F.relu(transform(input))
            output = g * t + (1 - g) * input
        return output

# LSTMEncoder:
# layer for encoding a sequence using a bidirectional LSTM
class LSTMEncoder(nn.Module):
    # args:
    # input_size: size of a a timestep in the input.
    # hidden_size: RNN hidden state size.
    # num_layers: number of layers of RNN cells to use
    # drop_prob: probability of zero-ing out activations
    def __init__(self, input_size, hidden_size, num_layers, drop_prop=0.):
        super(LSTMEncoder, self).__init__()
        self.drop_prop = drop_prop
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=drop_prop if num_layers > 1 else .0)

    def forward(self, input, lengths):
        # Saving original length for pad_packed_sequence
        og_length = input.size(1)

        # Sorting by length and pack sequence for RNN
        lengths, sort_idinput = lengths.sort(0, descending=True)
        input = input[sort_idinput] # (batch size, sequence length, input size)
        input = pack_padded_sequence(input, lengths, batch_first=True)

        # Applying rnn
        input, _ = self.lstm(input)

        # Unpack and reverse sort
        input, _ = pad_packed_sequence(input, batch_first=True, total_length=og_length)
        _, unsort_idinput = sort_idinput.sort(0)
        input = input[unsort_idinput]

        # Applying drpout
        input = F.dropout(input, self.drop_prob, self.training)

        return input
# BiDAFAtt:
# Bidirectional attention computes the attention in two directions
# (context to query) and (query to context)
# the output is the concatenation of
# [context ,c2q_attention, context * c2q_attention, context * q2c_attention]
# This allows the attention vector at each timestep, along with embeddings from
# previous layers to flow through the att layer to the mod layer
class BiDAFAtt(nn.Module):
    # args:
    # hidden_size: size of hidden activations
    # drop_prob: prob of zero0iing out activations
    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAtt, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for w in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(w)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)

        # added masking to get have correct softmax results
        c_mask = c_mask.view(batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)
        s2 = masked_softmax(s, c_mask, dim=1)

        # Conext2Question Attention
        a = torch.bmm(s1, q)

        # Question2Conext Attention
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        # Bi-directional Attention Flow for all context
        input = torch.cat([c, a, c * a, c * b], dim=2)

        return input

    def get_similarity_matrix(self, c, q):
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)
        q = F.dropout(q, self.drop_prob, self.training)

        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1,2)\
            .expand([-1,c_len, -1])


        s2 = torch.matmul(c * self.cq_weight, q.transpose(1,2))
        s = s0 + s1 + s2 + self.bias

        return s

# BiDAFOutput:
# Computes a linear transformation of the attention and modeling  outputs
# Then takes softmax of result to get start pointer.
# A bidirectional LSTM is then applied to modeling output to create 'mod_2'
# A second linear+softmax of the attention output and 'mod_2' is used to get
# end pointer
class BiDAFOutput(nn.Module):

    # args:
    # hidden_size: hidden size used in BiDAF model
    # drop_prob: probability of zero-ing out activations
    def __init__(self, hidden_size, drop_prob):
        super(BiDAFOutput, self).__init__()
        self.att_linear_1 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.lstm = LSTMEncoder(input_size=2*hidden_size,
                                hidden_size=hidden_size,
                                num_layers=1, drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, mod, mask):
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.lstm(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2
