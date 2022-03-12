import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LICACritic(nn.Module):
    def __init__(self, n_agents, n_actions, state_shape, mixing_embed_dim=128, hypernet_layers=1):
        super(LICACritic, self).__init__()

        # self.args = args
        self.n_actions = n_actions
        self.n_agents = n_agents

        self.output_type = "q"

        # Set up network layers
        self.state_dim = int(np.prod(state_shape)) * self.n_agents

        self.embed_dim = mixing_embed_dim * self.n_agents * self.n_actions
        self.hid_dim = mixing_embed_dim

        if hypernet_layers == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim)
            self.hyper_w_final = nn.Linear(self.state_dim, self.hid_dim)
        elif hypernet_layers == 2:
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                                           nn.ReLU(),
                                           nn.Linear(self.embed_dim, self.embed_dim))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, self.hid_dim),
                                               nn.ReLU(),
                                               nn.Linear(self.hid_dim, self.hid_dim))
        elif hypernet_layers > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.hid_dim)

        self.hyper_b_2 = nn.Sequential(nn.Linear(self.state_dim, self.hid_dim),
                                       nn.ReLU(),
                                       nn.Linear(self.hid_dim, 1))

    def forward(self, act, states):
        bs = states.size(0)
        # print('action',act.size())
        states = states.reshape(-1, self.state_dim)
        # print('states',states.size())
        action_probs = act.reshape(-1, 1, self.n_agents * self.n_actions)

        w1 = self.hyper_w_1(states)
        b1 = self.hyper_b_1(states)
        # print('w1', w1.size())
        w1 = w1.view(-1, self.n_agents * self.n_actions, self.hid_dim)
        # print('w1', w1.size())
        b1 = b1.view(-1, 1, self.hid_dim)
        # print(action_probs.size())
        # print(w1.size())
        h = torch.relu(torch.bmm(action_probs, w1) + b1)
        # print(h.detach().cpu().numpy())
        w_final = self.hyper_w_final(states)
        w_final = w_final.view(-1, self.hid_dim, 1)
        # print(h.size())
        # print(w_final.size())
        b2 = self.hyper_b_2(states).view(-1, 1, 1)
        # print(b2.size())
        h2 = torch.bmm(h, w_final)
        # print(h2.detach().cpu().numpy())

        q = h2 + b2

        q = q.view(bs, -1, 1)

        return q


class Critic(nn.Module):
    def __init__(self, n_agents, n_actions, state_shape, mixing_embed_dim=128, hypernet_layers=1):
        super(Critic, self).__init__()
        self.max_action = n_actions
        self.agent_dim = state_shape * n_agents + n_actions * n_agents
        print(self.agent_dim)
        self.fc1 = nn.Linear(self.agent_dim, mixing_embed_dim)
        self.fc2 = nn.Linear(mixing_embed_dim, mixing_embed_dim)
        self.fc3 = nn.Linear(mixing_embed_dim, mixing_embed_dim)
        self.q_out = nn.Linear(mixing_embed_dim, 1)

    def forward(self, state, action):
        state = torch.cat(state, dim=1)
        for i in range(len(action)):
            action[i] /= self.max_action
        action = torch.cat(action, dim=1)
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)
        return q_value
