import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QMixer(nn.Module):
    def __init__(self, n_agents, state_shape, mixing_embed_dim = 128, hypernet_layers = 1):
        super(QMixer, self).__init__()

        self.n_agents = n_agents
        self.state_dim = int(np.prod(state_shape)) * self.n_agents

        self.embed_dim = mixing_embed_dim
        self.abs =  False

        if hypernet_layers == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            # self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim)
            # self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
            self.hyper_w_final = nn.Linear(self.embed_dim, 1)
        elif hypernet_layers == 2:
            hypernet_embed = mixing_embed_dim
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(hypernet_embed, self.embed_dim))
        elif hypernet_layers > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(inplace=True),
                               nn.Linear(self.embed_dim, 1))
        

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        # print('action',agent_qs.size())
        states = states.reshape(-1, self.state_dim)
        # print('states', states.size())
        agent_qs = agent_qs.reshape(-1, 1, self.n_agents)
        # First layer
        w1 = self.hyper_w_1(states).abs() if self.abs else self.hyper_w_1(states)
        b1 = self.hyper_b_1(states)
        # print('w1',w1.size())
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        # print('w1',w1.size())
        b1 = b1.view(-1, 1, self.embed_dim)
        # print(agent_qs.size(), w1.size())
        # print(th.bmm(agent_qs, w1).size(), b1.size())
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        
        # Second layer
        # w_final = self.hyper_w_final(states).abs() if self.abs else self.hyper_w_final(states)
        # w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        w_final = self.hyper_w_final(hidden).abs() if self.abs else self.hyper_w_final(hidden)
        # print(w_final.size())
        v = self.V(states).view(-1, 1, 1)

        # Compute final output
        # print(hidden.size(),w_final.size())
        # y = th.bmm(hidden, w_final) + v
        y = w_final + v
        # Reshape and return
        q_tot = y.view(bs, -1)
        return q_tot

    def k(self, states):
        bs = states.size(0)
        w1 = th.abs(self.hyper_w_1(states))
        w_final = th.abs(self.hyper_w_final(states))
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        w_final = w_final.view(-1, self.embed_dim, 1)
        k = th.bmm(w1,w_final).view(bs, -1, self.n_agents)
        k = k / th.sum(k, dim=2, keepdim=True)
        return k

    def b(self, states):
        bs = states.size(0)
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        b1 = self.hyper_b_1(states)
        b1 = b1.view(-1, 1, self.embed_dim)
        v = self.V(states).view(-1, 1, 1)
        b = th.bmm(b1, w_final) + v
        return b
