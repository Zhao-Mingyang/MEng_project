import math, random, copy
import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

from DGN import DGN
from buffer import ReplayBuffer
from env import multi_agent_env as Surviving
from config import *

USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)
env = Surviving(n_agent=2)
n_ant = env.n_agent
observation_space = env.len_obs
n_actions = env.n_action
mask_dim = env.mask_dim
buff = ReplayBuffer(capacity)

model = DGN(n_ant, observation_space, hidden_dim, n_actions,mask_dim)
model_tar = DGN(n_ant, observation_space, hidden_dim, n_actions,mask_dim)
model = model.cuda()
model_tar = model_tar.cuda()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

O = np.ones((batch_size, n_ant, observation_space))
Next_O = np.ones((batch_size, n_ant, observation_space))
Matrix = np.ones((batch_size, n_ant, mask_dim))
Next_Matrix = np.ones((batch_size, n_ant, mask_dim))

while i_episode < n_episode:

    if i_episode > 100:
        epsilon -= 0.0004
        if epsilon < 0.1:
            epsilon = 0.1
    i_episode += 1
    steps = 0
    obs, adj = env.reset()
    # print(obs.shape)
    # print(adj.shape)
    while steps < max_step:
        steps += 1
        action = []
        q,q_t = model(torch.Tensor(np.array([obs])).cuda(), torch.Tensor(np.array([adj])).cuda())
        q, q_t=q[0], q_t[0]

        # print('q',q,'q_t',q_t)
        for i in range(n_ant):
            # print(i)
            if np.random.rand() < epsilon:
                a = np.random.randint(n_actions)
                # print('a',a)
            else:
                a = [q[i].argmax().item(), q_t[i].argmax().item()]
            action.append(a)
        # print('action',action)
        next_obs, next_adj, reward, terminated = env.step(action)

        buff.add(np.array(obs), action, reward, np.array(next_obs), adj, next_adj, terminated)
        obs = next_obs
        adj = next_adj
        score += sum(reward)

    if i_episode % 20 == 0:
        print(score / 2000)
        with open('r.txt', 'w') as f:
            f.write(str(score / 2000) + '\n')
        score = 0

    if i_episode < 100:
        continue

    for e in range(n_epoch):

        batch = buff.getBatch(batch_size)
        for j in range(batch_size):
            sample = batch[j]
            # print(sample)
            O[j] = sample[0]
            Next_O[j] = sample[3]
            Matrix[j] = sample[4]
            Next_Matrix[j] = sample[5]

        # print('0', sample[0])
        # print('1', sample[1])
        # print('2', sample[2])
        # print('3', sample[3])
        # print('4', sample[4])
        # print('5', sample[5])
        # print('6', sample[6])
        q_values, q_values_t = model(torch.Tensor(O).cuda(), torch.Tensor(Matrix).cuda())
        # print('q_values',q_values, 'q_values_t', q_values_t)
        target_q_values, target_q_values_t = model_tar(torch.Tensor(Next_O).cuda(), torch.Tensor(Next_Matrix).cuda())
        target_q_values  = target_q_values.max(dim=2)[0]
        target_q_values_t = target_q_values_t.max(dim=2)[0]
        target_q_values = np.array(target_q_values.cpu().data)
        target_q_values_t = np.array(target_q_values_t.cpu().data)
        expected_q = np.array(q_values.cpu().data)
        expected_q_t = np.array(q_values_t.cpu().data)

        for j in range(batch_size):
            sample = batch[j]
            for i in range(n_ant):
                # print(expected_q[j][i][sample[1][i][0]])
                # print(sample[1][i][0])
                expected_q[j][i][sample[1][i][0]] = sample[2][i] + (1 - sample[6]) * GAMMA * target_q_values[j][i]
                expected_q_t[j][i][sample[1][i][1]] = sample[2][i] + (1 - sample[6]) * GAMMA * target_q_values_t[j][i]

        loss = (q_values - torch.Tensor(expected_q).cuda()).pow(2).mean()
        loss_t = (q_values_t - torch.Tensor(expected_q_t).cuda()).pow(2).mean()
        loss = loss+loss_t
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if i_episode % 5 == 0:
        model_tar.load_state_dict(model.state_dict())



