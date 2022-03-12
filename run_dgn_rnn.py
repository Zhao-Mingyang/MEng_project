import math, random, copy
import numpy as np
import os, sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn.functional as F

from mixers.vdn import VDNMixer
from mixers.qmix import QMixer
from DGN import DGN
from buffer_R import ReplayBuffer
from env import multi_agent_env as Surviving
from config import *
from critics.lica import LICACritic
# from critics.lica import Critic as LICACritic
from common.action_selectors import multinomial_entropy
from common.rl_utils import build_td_lambda_targets

USE_CUDA = torch.cuda.is_available()

env = Surviving(n_agent=2)
n_ant = env.n_agent
observation_space = env.len_obs
n_actions = env.n_action
mask_dim = env.mask_dim

buff = ReplayBuffer(capacity,hidden_dim, observation_space, n_actions, n_ant)

critic = LICACritic(n_ant,n_actions, observation_space).cuda()
target_critic = copy.deepcopy(critic)

model = DGN(n_ant, observation_space, hidden_dim, n_actions,mask_dim)
# model_tar = DGN(n_ant, observation_space, hidden_dim, n_actions,mask_dim)
model_tar = copy.deepcopy(model)
model = model.cuda()
meansloss = nn.MSELoss()
model_tar = model_tar.cuda()

# agent_params = list(model.parameters())
# critic_params = list(critic.parameters())
# params = agent_params + critic_params
# agent_optimiser = optim.Adam(params=agent_params, lr=0.0025)
# critic_optimiser = optim.Adam(params=critic_params, lr=0.0005)

best_score = 0
M_Null = torch.Tensor(np.array([np.eye(n_ant)] * batch_size)).cuda()
KL = nn.KLDivLoss()
# mixer = VDNMixer().cuda()
mixer = QMixer(n_ant, observation_space).cuda()
actor_params = list(model.parameters())
critic_params = list(mixer.parameters())
params = actor_params + critic_params
target_mixer = copy.deepcopy(mixer)
# optimizer = optim.Adam(model.parameters(), lr=0.0001)
# agent_optimiser = optim.Adam(params=actor_params, lr=0.0001)
# critic_optimiser = optim.Adam(params=critic_params, lr=0.0001)
optimizer = optim.Adam(params=params,  lr=0.0001)

critic_training_steps = 0

with open('r_DGN+R.txt', 'w') as f:
    f.write(str(' ') + '\n')

while i_episode < n_episode:

    if i_episode > 50:
        epsilon -= 0.001
        if epsilon < 0.01:
            epsilon = 0.01
    i_episode += 1
    steps = 0
    obs = env.reset()
    terminated = False
    hidden_states = None

    # print(adj.shape)
    while True:
        # print(obs)
        steps += 1
        action = []
        if hidden_states is None:
            hidden_states = model.init_hidden()
        q, next_hidden_states = model(torch.Tensor(np.array([obs])).cuda(), hidden_states)

        # if hidden_states is not None:
        hidden_states = hidden_states.detach().cpu().numpy()
        next_hidden_states = next_hidden_states.detach().cpu().numpy()
        q = q[0]
        for i in range(n_ant):
            if np.random.rand() < epsilon:
                a = np.random.randint(n_actions)
            else:
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
    if i_episode % 20 == 0:
        print(str(score / 200))
        with open('r_DGN+R.txt', 'a') as f:
            f.write(str(score / 200) + '\n')
        if score > best_score:
            torch.save(model.state_dict(), 'model_cifar.pt')
            print('Model Svaed Score from', best_score, 'to', score)
            best_score = score

        score = 0
    # if i_episode % 200 ==0:
    #     target_mixer.load_state_dict(mixer.state_dict())

    if i_episode < 40:
        continue

    for e in range(n_epoch):

        O, A, R, Next_O, hidden_states,next_hidden_states, D = buff.getBatch(batch_size)
        O = torch.Tensor(O).cuda()

        Next_O = torch.Tensor(Next_O).cuda()
        hidden_states = torch.Tensor(hidden_states).cuda()
        next_hidden_states = torch.Tensor(next_hidden_states).cuda()

        q_values, attention = model(O, hidden_states)
        # print(q_values.detach().cpu().numpy())
        target_q_values, target_attention = model_tar(Next_O, next_hidden_states)

        # print(target_q_values)
        target_q_values = target_q_values.max(dim=2)[0]
        # print(target_q_values.size(),'1')
        # print(target_q_values)
        target_q_values = np.array(target_q_values.cpu().data)
        # print(target_q_values.shape, '2')
        chosen_action_qvals = np.zeros((batch_size, n_ant))
        expected_q = np.array(q_values.cpu().data)

        for j in range(batch_size):
            for i in range(n_ant):
                chosen_action_qvals[j][i] = expected_q[j][i][A[j][i]]

        chosen_action_qvals = mixer(torch.Tensor(chosen_action_qvals).cuda(), O)
        # print(chosen_action_qvals.size(), '1')
        target_q_qvals = target_mixer(torch.Tensor(target_q_values).cuda(), Next_O)
        # print(target_q_qvals.size(), '3')
        R=np.sum(R,axis=1)
        R = np.expand_dims(R, axis=1)
        # print(R.shape,'R')
        # print(R,D,target_q_qvals.detach().cpu().numpy())
        target_q_qvals = R + (1 - D) * GAMMA * target_q_qvals.detach().cpu().numpy()
        # print(target_q_qvals.shape,'4')
        # target_q_qvals =  target_q_qvals
        # print(target_q_qvals.shape, '4')
        # print(chosen_action_qvals.size(),'2')
        # critic_loss = (chosen_action_qvals - torch.Tensor(target_q_qvals).cuda()).sum() * 0.1
        critic_loss = meansloss(chosen_action_qvals, torch.Tensor(target_q_qvals).cuda())
        # chosen_action_qvals = chosen_action_qvals.detach().cpu().numpy()
        # current_q = np.array(q_values.cpu().data)
        # for j in range(batch_size):
        #     for i in range(n_ant):
        #
        #         # print(target_q_qvals.shape, chosen_action_qvals.shape)
        #         expected_q[j][i][A[j][i]] = target_q_qvals[j]
        #         current_q[j][i][A[j][i]] = chosen_action_qvals[j]
        # current_q = torch.Tensor(current_q).cuda()
        # current_q = Variable(current_q, requires_grad=True)
        # expected_q = torch.Tensor(expected_q).cuda()
        # expected_q = Variable(expected_q, requires_grad=True)
        # loss = meansloss(current_q, expected_q )
        # loss = -current_q
        # print(loss,critic_loss)

        optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        optimizer.step()

        # critic_optimiser.zero_grad()
        # critic_loss.backward(retain_graph=True)
        # critic_optimiser.step()

        with torch.no_grad():
            for p, p_targ in zip(model.parameters(), model_tar.parameters()):
                p_targ.data.mul_(tau)
                p_targ.data.add_((1 - tau) * p.data)
            for p, p_targ in zip(mixer.parameters(), target_mixer.parameters()):
                p_targ.data.mul_(tau)
                p_targ.data.add_((1 - tau) * p.data)



