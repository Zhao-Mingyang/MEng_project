import math, random, copy
import numpy as np
import os, sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

from mixers.vdn import VDNMixer
from mixers.qmix import QMixer
from DGN import DGN
from buffer_R import ReplayBuffer
from env import multi_agent_env as Surviving
from config import *

USE_CUDA = torch.cuda.is_available()

env = Surviving(n_agent=5)
n_ant = env.n_agent
observation_space = env.len_obs
n_actions = env.n_action
mask_dim = env.mask_dim

buff = ReplayBuffer(capacity,hidden_dim, observation_space, n_actions, n_ant)
model = DGN(n_ant, observation_space, hidden_dim, n_actions,mask_dim)
model_tar = copy.deepcopy(model)
# model_tar = DGN(n_ant, observation_space, hidden_dim, n_actions,mask_dim)
model = model.cuda()
model_tar = model_tar.cuda()
# model.load_state_dict(torch.load('model_cifar.pt'))
# model_tar.load_state_dict(torch.load('model_cifar.pt'))

best_score = 0
M_Null = torch.Tensor(np.array([np.eye(n_ant)] * batch_size)).cuda()
KL = nn.KLDivLoss()
# mixer = VDNMixer().cuda()
# mixer = QMixer(n_ant, observation_space).cuda()
# params = list(model.parameters())
# params += list(mixer.parameters())
# target_mixer = copy.deepcopy(mixer)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
# optimizer = optim.Adam(params=params,  lr=0.0001)


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
        print(str(score / 2000))
        with open('r_DGN+R.txt', 'a') as f:
            f.write(str(score / 2000) + '\n')
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

        O, A, R, Next_O, hidden_states, next_hidden_states, D = buff.getBatch(batch_size)
        O = torch.Tensor(O).cuda()

        Next_O = torch.Tensor(Next_O).cuda()
        hidden_states = torch.Tensor(hidden_states).cuda()
        next_hidden_states = torch.Tensor(next_hidden_states).cuda()


        q_values, attention = model(O, hidden_states)
        # print(q_values.detach().cpu().numpy())
        target_q_values, target_attention = model_tar(Next_O, next_hidden_states)
        # target_q_values, target_attention = model_tar(Next_O, Next_Matrix)
        # print(target_q_values)
        target_q_values = target_q_values.max(dim=2)[0]
        # print(target_q_values)
        target_q_values = np.array(target_q_values.cpu().data)
        # chosen_action_qvals = np.zeros((batch_size, n_ant))
        expected_q = np.array(q_values.cpu().data)

        # for j in range(batch_size):
        #     for i in range(n_ant):
        #         chosen_action_qvals[j][i] = expected_q[j][i][A[j][i]]
        #
        # chosen_action_qvals = mixer(torch.Tensor(chosen_action_qvals).cuda(), O)
        # target_q_qvals = target_mixer(torch.Tensor(target_q_values).cuda(), Next_O)
        # R=np.sum(R,axis=1)
        # # print(R,D,target_q_qvals.detach().cpu().numpy())
        # target_q_qvals = R + (1 - D) * GAMMA * target_q_qvals.detach().cpu().numpy()
        # # print(target_q_qvals)
        # loss = (chosen_action_qvals - torch.Tensor(target_q_qvals).cuda()).pow(2).sum()/torch.abs(((chosen_action_qvals - torch.Tensor(target_q_qvals).cuda()).sum())) * 0.05

        for j in range(batch_size):
            for i in range(n_ant):
                expected_q[j][i][A[j][i]] = R[j][i] + (1 - D[j]) * GAMMA * target_q_values[j][i]

        attention = F.log_softmax(attention, dim=2)
        target_attention = F.softmax(target_attention, dim=2)
        target_attention = target_attention.detach()
        loss_kl = F.kl_div(attention, target_attention, reduction='batchmean')

        # print(loss_kl)
        loss = (q_values - torch.Tensor(expected_q).cuda()).pow(2).mean()
               # + lamb * loss_kl
        # print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            for p, p_targ in zip(model.parameters(), model_tar.parameters()):
                p_targ.data.mul_(tau)
                p_targ.data.add_((1 - tau) * p.data)


# -------------------------------------------------
# import math, random, copy
# import numpy as np
# import os, sys
#
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.autograd as autograd
# import torch.nn.functional as F
#
# from mixers.vdn import VDNMixer
# from mixers.qmix import QMixer
# from DGN import DGN
# from buffer_R import ReplayBuffer
# from env import multi_agent_env as Surviving
# from config import *
# from critics.lica import LICACritic
# # from critics.lica import Critic as LICACritic
# from common.action_selectors import multinomial_entropy
# from common.rl_utils import build_td_lambda_targets
#
# USE_CUDA = torch.cuda.is_available()
#
# env = Surviving(n_agent=2)
# n_ant = env.n_agent
# observation_space = env.len_obs
# n_actions = env.n_action
# mask_dim = env.mask_dim
#
# buff = ReplayBuffer(capacity,hidden_dim, observation_space, n_actions, n_ant)
#
# critic = LICACritic(n_ant,n_actions, observation_space).cuda()
# target_critic = copy.deepcopy(critic)
#
# model = DGN(n_ant, observation_space, hidden_dim, n_actions,mask_dim)
# # model_tar = DGN(n_ant, observation_space, hidden_dim, n_actions,mask_dim)
# model = model.cuda()
# meansloss = nn.MSELoss()
# # model_tar = model_tar.cuda()
#
# agent_params = list(model.parameters())
# critic_params = list(critic.parameters())
# params = agent_params + critic_params
# agent_optimiser = optim.Adam(params=agent_params, lr=0.0025)
# critic_optimiser = optim.Adam(params=critic_params, lr=0.0005)
#
# best_score = 0
# M_Null = torch.Tensor(np.array([np.eye(n_ant)] * batch_size)).cuda()
# KL = nn.KLDivLoss()
# # mixer = VDNMixer().cuda()
# mixer = QMixer(n_ant, observation_space).cuda()
# # params = list(model.parameters())
# # params += list(mixer.parameters())
# # target_mixer = copy.deepcopy(mixer)
# # optimizer = optim.Adam(model.parameters(), lr=0.0001)
# # optimizer = optim.Adam(params=params,  lr=0.0001)
#
# critic_training_steps = 0
#
# with open('r_DGN+R.txt', 'w') as f:
#     f.write(str(' ') + '\n')
#
# while i_episode < n_episode:
#
#     if i_episode > 50:
#         epsilon -= 0.001
#         if epsilon < 0.01:
#             epsilon = 0.01
#     i_episode += 1
#     steps = 0
#     obs = env.reset()
#     terminated = False
#     hidden_states = None
#
#     # print(adj.shape)
#     while True:
#         # print(obs)
#         steps += 1
#         action = []
#         if hidden_states is None:
#             hidden_states = model.init_hidden()
#         q, next_hidden_states = model(torch.Tensor(np.array([obs])).cuda(), hidden_states)
#
#         # if hidden_states is not None:
#         hidden_states = hidden_states.detach().cpu().numpy()
#         next_hidden_states = next_hidden_states.detach().cpu().numpy()
#         q = q[0]
#         for i in range(n_ant):
#             if np.random.rand() < epsilon:
#                 a = np.random.randint(n_actions)
#             else:
#                 a = q[i].argmax().item()
#             action.append(a)
#         # print(action)
#         next_obs, reward, terminated = env.step(action)
#         # break
#         buff.add(np.array(obs), action, reward, np.array(next_obs), hidden_states, next_hidden_states, terminated)
#         hidden_states = torch.Tensor(next_hidden_states).cuda()
#         obs = next_obs
#
#         score += sum(reward)
#         if terminated:
#             break
#     if i_episode % 20 == 0:
#         print(str(score / 2000))
#         with open('r_DGN+R.txt', 'a') as f:
#             f.write(str(score / 2000) + '\n')
#         if score > best_score:
#             torch.save(model.state_dict(), 'model_cifar.pt')
#             print('Model Svaed Score from', best_score, 'to', score)
#             best_score = score
#
#         score = 0
#     # if i_episode % 200 ==0:
#     #     target_mixer.load_state_dict(mixer.state_dict())
#
#     if i_episode < 40:
#         continue
#
#     for e in range(n_epoch):
#
#         O, A, R, Next_O, hidden_states,next_hidden_states, D = buff.getBatch(batch_size)
#         O = torch.Tensor(O).cuda()
#
#         Next_O = torch.Tensor(Next_O).cuda()
#         hidden_states = torch.Tensor(hidden_states).cuda()
#         next_hidden_states = torch.Tensor(next_hidden_states).cuda()
#
#         one_hot_action = np.zeros((batch_size,n_ant, n_actions))
#         # print(chosen_action_qvals.shape, one_hot_action.shape)
#         for j in range(batch_size):
#             for i in range(n_ant):
#                 one_hot_action[j][i][A[j][i]] = 1
#
#         agent_out, attention = model(O, hidden_states)
#         # q_values = nn.functional.softmax(q_values, dim=-1)
#         q_values = agent_out.detach().cpu().numpy()
#         q_maxes = np.max(q_values, axis=2)
#         q_mins = np.min(q_values, axis=2)
#         q_maxes = np.expand_dims(q_maxes, axis=2)
#         q_mins = np.expand_dims(q_mins, axis=2)
#         q_values = (q_values - q_mins)/(q_maxes - q_mins)
#         # for j in range(batch_size):
#         #     for i in range(n_ant):
#         #         q_values[j,i,:] = (q_values[j,i,:] - q_maxes[j][i])(q_maxes[j][i] - q_mins[j][i])
#
#
#         # agent_entropy = multinomial_entropy(agent_out).mean(dim=-1, keepdim=True)
#         # agent_probs = nn.functional.softmax(q_values, dim=-1)
#
#         # next_q, next_attention = model(Next_O, next_hidden_states)
#         # next_q = next_q.argmax(-1)
#         # # print(next_q)
#         # # next_q = np.array(next_q.cpu().data).astype(int)
#         # next_one_hot_action = np.zeros((batch_size, n_ant, n_actions))
#         # # print(chosen_action_qvals.shape, one_hot_action.shape)
#         # for j in range(batch_size):
#         #     for i in range(n_ant):
#         #         next_one_hot_action[j][i][next_q[j][i]] = 1
#         # print('next_one_hot_action',next_one_hot_action)
#         # print('q_values',q_values.detach().cpu().numpy())
#         print('q_values', q_values)
#         q_values = torch.Tensor(q_values).cuda()
#         mix_loss = critic(q_values, O)
#
#         # raise KeyError('stop')
#         target_q_vals = target_critic(torch.Tensor(one_hot_action).cuda(), Next_O)
#         R = torch.Tensor(R).cuda()
#         D = torch.Tensor(D).cuda()
#         # print(test_R.size(),target_q_vals.size())
#         for j in range(batch_size):
#             temp_R = 0
#             for i in range(n_ant):
#                 temp_R = temp_R + R[j][i]
#             target_q_vals[j,:,:] = temp_R + (1 - D[j]) * GAMMA * target_q_vals[j,:,:]
#
#         q_t = critic(torch.Tensor(one_hot_action).cuda(), O)
#         loss = meansloss(q_t, target_q_vals)
#             # .pow(2).mean()
#
#         actor_loss = -mix_loss.mean()
#         #     # .pow(2).mean()
#         # print(mix_loss.detach().cpu().numpy())
#         # print(target_q_vals.detach().cpu().numpy())
#         # print(actor_loss)
#         agent_optimiser.zero_grad()
#         actor_loss.backward(retain_graph=True)
#         # grad_norm = nn.utils.clip_grad_norm_(agent_params, 10)
#         agent_optimiser.step()
#
#         critic_optimiser.zero_grad()
#         loss.backward(retain_graph=True)
#         # grad_norm = nn.utils.clip_grad_norm_(critic_params, 10)
#         critic_optimiser.step()
#
#         for target_param, param in zip(target_critic.parameters(), critic.parameters()):
#             target_param.data.copy_(tau * target_param.data + (1 - tau) * param.data)
#
#
#
