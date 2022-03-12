# import math, random
# import numpy as np
#
# import torch
# import torch.nn as nn
#
# import torch.optim as optim
# import torch.autograd as autograd
# import torch.nn.functional as F
# USE_CUDA = torch.cuda.is_available()
# Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)
#
# class Encoder(nn.Module):
# 	def __init__(self, din=32, hidden_dim=128):
# 		super(Encoder, self).__init__()
# 		self.fc = nn.Linear(din, hidden_dim)
#
# 	def forward(self, x):
# 		embedding = F.relu(self.fc(x))
# 		return embedding
#
# class AttModel(nn.Module):
# 	def __init__(self, n_node, din, hidden_dim, dout,mask_dim):
# 		super(AttModel, self).__init__()
# 		self.fca = nn.Linear(mask_dim, n_node)
# 		self.fcv = nn.Linear(din, hidden_dim)
# 		self.fck = nn.Linear(din, n_node*2)
# 		self.fcq = nn.Linear(din, n_node*2)
# 		# self.fck = nn.Linear(din, hidden_dim)
# 		# self.fcq = nn.Linear(din, hidden_dim)
# 		self.fcout = nn.Linear(hidden_dim, dout)
#
# 	def forward(self, x, mask):
# 		v = F.relu(self.fcv(x))
# 		q = F.relu(self.fcq(x))
# 		k = F.relu(self.fck(x)).permute(0,2,1)
# 		bmm = torch.bmm(q,k)
# 		# mask = torch.squeeze(mask)
# 		# print('k',k.size())
# 		# print('q',q.size())
# 		# print('bmm', bmm.size())
# 		# print('mask', mask.size())
# 		value = torch.matmul(bmm, mask)
# 		value= value- 10*(1 - mask)
# 		att= F.relu(self.fca(value))
# 		# att = F.softmax(value,dim=2)
# 		# print('att', att.size())
# 		out = torch.bmm(att,v)
# 		#out = torch.add(out,v)
# 		out = F.relu(self.fcout(out))
# 		return out
#
# class Q_Net(nn.Module):
# 	def __init__(self, hidden_dim, dout):
# 		super(Q_Net, self).__init__()
# 		self.fc = nn.Linear(hidden_dim, dout)
#
# 	def forward(self, x):
# 		q = self.fc(x)
# 		return q
#
# class DGN(nn.Module):
# 	def __init__(self,n_agent,num_inputs,hidden_dim,num_actions,mask_dim):
# 		super(DGN, self).__init__()
#
# 		self.encoder = Encoder(num_inputs,hidden_dim)
# 		self.att_1 = AttModel(n_agent,hidden_dim,hidden_dim,hidden_dim,mask_dim)
# 		self.att_2 = AttModel(n_agent,hidden_dim,hidden_dim,hidden_dim,mask_dim)
# 		self.q_net1 = Q_Net(hidden_dim,num_actions[0])
# 		self.q_net2 = Q_Net(hidden_dim, num_actions[1])
#
# 	def forward(self, x, mask):
# 		# x = torch.squeeze(x)
# 		# print(x.size())
# 		h1 = self.encoder(x)
# 		# print(h1.size())
# 		# print(mask.size())
# 		h2 = self.att_1(h1, mask)
# 		h3 = self.att_2(h2, mask)
# 		q1 = self.q_net1(h3)
# 		q2 = self.q_net2(h3)
# 		# print(q1.size())
# 		return q1,q2
#
#
#
#
import math, random
import numpy as np

import torch
import torch.nn as nn

import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args,
                                                                                                                **kwargs)


class Encoder(nn.Module):
    def __init__(self, din=32, hidden_dim=128):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(din, hidden_dim)
        # self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        # self.rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)

    def forward(self, x):
        embedding = F.relu(self.fc(x))
        # print(embedding.size())
        # embedding, _ = self.rnn(embedding)
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

    def forward(self, x):
        q = self.fc(x)
        return q


class DGN(nn.Module):
    def __init__(self, n_agent, num_inputs, hidden_dim, num_actions, mask_dim):
        super(DGN, self).__init__()

        self.encoder = Encoder(num_inputs, hidden_dim)
        # print(hidden_dim)
        self.linear = nn.Linear(hidden_dim, hidden_dim//2)
        self.att_1 = AttModel(n_agent, hidden_dim, hidden_dim, hidden_dim, mask_dim)
        self.att_2 = AttModel(n_agent, hidden_dim, hidden_dim, hidden_dim, mask_dim)
        self.q_net1 = Q_Net(hidden_dim//2, num_actions)

    # self.q_net2 = Q_Net(hidden_dim, num_actions[1])

    def forward(self, x, mask):
        # x = torch.squeeze(x)
        # print(x.size())
        h1 = self.encoder(x)
        h3 = F.relu(self.linear(h1))

        # print(h1.size())
        # print(mask.size())
        # h2,_ = self.att_1(h1, mask)
        # h3,a_w = self.att_2(h1, mask)
        q1 = self.q_net1(h3)
        # q2 = self.q_net2(h3)
        # print(q1.size())
        return q1, q1
















