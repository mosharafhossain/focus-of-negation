# -*- coding: utf-8 -*-
"""
The Attention class is adapted from the link: https://mlwhiz.com/blog/2019/03/09/deeplearning_architectures_text_classification/
"""

import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, feature_size, seq_len, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.bias = bias
        self.feature_size = feature_size
        self.seq_len = seq_len

        weight = torch.zeros(feature_size, 1)  # feature_size x1
        nn.init.kaiming_uniform_(weight)
        self.weight = nn.Parameter(weight)  # feature_size x1

        if bias:
            self.b = nn.Parameter(torch.zeros(seq_len))  # seq_len

    def forward(self, x, mask=None):
        x1 = x.contiguous().view(-1, self.feature_size)  # (batch * seq_len) x feature_size
        x1 = torch.mm(x1, self.weight)  # (batch * seq_len) x 1
        x1 = x1.view(-1, self.seq_len)  # batch x seq_len
        if self.bias: x1 = x1 + self.b  # batch x seq_len

        x1 = torch.tanh(x1)  # batch x seq_len
        x1 = torch.exp(x1)  # batch x seq_len
        if mask is not None: x1 = x1 * mask
        x1 = x1 / (torch.sum(x1, 1,
                           keepdim=True) + 1e-10)  # batch x seq_len, softmax fn: a = F.softmax(eij, dim=-1)

        weighted_input_seq = x * torch.unsqueeze(x1, -1)  # batch x seq_len x feature_size
        weighted_input_notseq = torch.sum(weighted_input_seq, 1)  # sum over 2nd dimension (seq_len), Dim: batch  x feature_size.
        return weighted_input_seq, weighted_input_notseq
