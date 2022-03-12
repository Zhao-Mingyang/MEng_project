import math, random, copy
import numpy as np
import os, sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

from DGN import DGN
from buffer_R import ReplayBuffer
from env import multi_agent_env as Surviving
from config import *

USE_CUDA = torch.cuda.is_available()
print('yes')
env = Surviving(n_agent=5)
print(env.landmarks)
n_ant = env.n_agent
observation_space = env.len_obs
n_actions = env.n_action
mask_dim = env.mask_dim

# buff = ReplayBuffer(capacity, observation_space, n_actions, n_ant)
buff = ReplayBuffer(capacity,hidden_dim, observation_space, n_actions, n_ant)
model = DGN(n_ant, observation_space, hidden_dim, n_actions,mask_dim)
model = model.cuda()
# model.load_state_dict(torch.load('model_cifar_BiGRU_hidden7070.pt'))
model.load_state_dict(torch.load('model_cifar.pt'))


best_score = 0
M_Null = torch.Tensor(np.array([np.eye(n_ant)] * batch_size)).cuda()
KL = nn.KLDivLoss()

i_episode += 1
steps = 0
obs = env.reset()
env.render()
terminated = False
hidden_states = None
epsilon = 0.0
# env.state_n= [np.array([71, 71,  0]) for _ in
#                         range(n_ant)]
    # print(adj.shape)
# while True:
#         # print(obs)
#     steps += 1
#     action = []
#     n_adj = adj + np.eye(n_ant)
#     q, a_w = model(torch.Tensor(np.array([obs])).cuda(), torch.Tensor(np.array([n_adj])).cuda())
#     q = q[0]
#     for i in range(n_ant):
#         if np.random.rand() < epsilon:
#             a = np.random.randint(n_actions)
#         else:
#             a = q[i].argmax().item()
#         action.append(a)
#
#     next_obs, next_adj, reward, terminated = env.step(action)
#     env.render()
#     # break
#     buff.add(np.array(obs), action, reward, np.array(next_obs), n_adj, next_adj, terminated)
#     obs = next_obs
#     adj = next_adj
#     score += sum(reward)
#     if terminated:
#         break


while True:
    # print(obs)
    steps += 1
    action = []
    if hidden_states is None:
        hidden_states = model.init_hidden()
    q, next_hidden_states = model(torch.Tensor(np.array([obs])).cuda(), hidden_states)
    env.render()
    # if hidden_states is not None:
    hidden_states = hidden_states.detach().cpu().numpy()
    next_hidden_states = next_hidden_states.detach().cpu().numpy()
    q = q[0]
    for i in range(n_ant):
        a = q[i].argmax().item()
        action.append(a)
    # print(action)
    next_obs, reward, terminated = env.step(action)
    # break
    buff.add(np.array(obs), action, reward, np.array(next_obs), hidden_states, next_hidden_states, terminated)
    hidden_states = torch.Tensor(next_hidden_states).cuda()
    obs = next_obs

    score += sum(reward)
    if terminated:
        break
