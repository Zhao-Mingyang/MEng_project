import math, random
import numpy as np

import torch
import torch.nn as nn

import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from common.Noisy_Linear import NoisyLinear

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args,
                                                                                                                **kwargs)


class Encoder(nn.Module):
    def __init__(self, din=32, hidden_dim=512):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(din, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        # self.rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional= True)

    def forward(self, x):
        embedding = F.relu(self.fc(x))
        # print(embedding.size())
        embedding, _ = self.rnn(embedding)
        return embedding


class AttModel(nn.Module):
    def __init__(self, n_node, din, hidden_dim, dout, mask_dim):
        super(AttModel, self).__init__()
        self.fca = nn.Linear(mask_dim, n_node)
        self.fcv = nn.Linear(din, hidden_dim)
        self.fck = nn.Linear(din, n_node * 4)
        self.fcq = nn.Linear(din, n_node * 4)
        # self.fck = nn.Linear(din, hidden_dim)
        # self.fcq = nn.Linear(din, hidden_dim)
        self.fcout = nn.Linear(hidden_dim, dout)

    def forward(self, x, mask):
        v = F.relu(self.fcv(x))
        q = F.relu(self.fcq(x))
        # print(x.size())
        k = F.relu(self.fck(x)).permute(0, 2, 1)
        bmm = torch.bmm(q, k)
        # mask = torch.squeeze(mask)
        # print('k',k.size())
        # print('q',q.size())
        # print('bmm', bmm.size())
        # print('mask', mask.size())
        # value = torch.matmul(bmm, mask)
        value = torch.mul(bmm, mask)
        value = value - 1e3 * (1 - mask)
        # att= F.relu(self.fca(value))
        att = F.softmax(value, dim=2)
        # print('att', att.size())
        out = torch.bmm(att, v)
        # out = torch.add(out,v)
        out = F.relu(self.fcout(out))
        return out, att


class Q_Net(nn.Module):
    def __init__(self, hidden_dim, dout):
        super(Q_Net, self).__init__()
        self.fc = nn.Linear(hidden_dim, dout)
        # self.fc = NoisyLinear(hidden_dim, dout, True, 'cuda')

    def forward(self, x):
        q = self.fc(x)
        return q


class DGN(nn.Module):
    def __init__(self, n_agent, num_inputs, hidden_dim, num_actions, mask_dim):
        super(DGN, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_agent = n_agent

        self.encoder = Encoder(num_inputs, hidden_dim)
        # print(hidden_dim)
        self.linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim * 2, hidden_dim * 2)

        self.att_1 = AttModel(n_agent, hidden_dim * 2, hidden_dim, hidden_dim, mask_dim)
        self.att_2 = AttModel(n_agent, hidden_dim, hidden_dim, hidden_dim, mask_dim)
        self.q_net1 = Q_Net(hidden_dim, num_actions)

    # self.q_net2 = Q_Net(hidden_dim, num_actions[1])

    def init_hidden(self):
        # make hidden states on same device as model
        hiddin_weight = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2)
        return hiddin_weight.weight.new(self.n_agent, self.hidden_dim * 2).zero_().cuda()

    def forward(self, x, hidden_state=None):
        # x = torch.squeeze(x)
        # print(x.size())
        b, a, e = x.size()
        h1 = self.encoder(x)

        if hidden_state is not None:
            hidden_state = hidden_state.reshape(-1, self.hidden_dim * 2)
        else:
            hidden_state = self.init_hidden()
        # print(hidden_state.size())
        h1 = h1.reshape(-1, self.hidden_dim * 2)
        # print(h1.size())
        h_out = self.gru(h1, hidden_state)
        h3 = F.relu(self.linear(h_out))

        # h_out = h_out.reshape(b, -1, self.hidden_dim * 2)
        # print(h1.size())
        # print(mask.size())
        # h2,_ = self.att_1(h_out, mask)
        # h3,a_w = self.att_2(h2, mask)
        q1 = self.q_net1(h3).view(b, a, -1)
        # q1 = F.softmax(q1, dim=-1)
        # q2 = self.q_net2(h3)
        # print(q1.size())
        return q1, h_out.view(b, a, -1)


