from collections import deque
import random
import numpy as np

class ReplayBuffer(object):

	def __init__(self, buffer_size, hidden_dim, obs_space, n_action, n_ant):
		self.buffer_size = buffer_size
		self.hidden_dim = hidden_dim
		self.n_ant = n_ant
		self.pointer = 0
		self.len = 0
		self.actions = np.zeros((self.buffer_size,self.n_ant),dtype = np.int32)
		self.rewards = np.zeros((self.buffer_size,n_ant))
		self.dones = np.zeros((self.buffer_size,1))
		self.obs = np.zeros((self.buffer_size,self.n_ant,obs_space))
		self.next_obs = np.zeros((self.buffer_size,self.n_ant,obs_space))
		self.hidden_states = np.zeros((self.buffer_size, self.n_ant, self.hidden_dim*2))
		self.next_hidden_states = np.zeros((self.buffer_size, self.n_ant, self.hidden_dim * 2))
		# self.next_hidden_states = np.zeros((self.buffer_size, self.n_ant, self.hidden_dim))

	def getBatch(self, batch_size):

		index = np.random.choice(self.len, batch_size, replace=False)
		return self.obs[index], self.actions[index], self.rewards[index], self.next_obs[index], self.hidden_states[index],  self.next_hidden_states[index], self.dones[index]

	def add(self, obs, action, reward, next_obs, hidden_states,next_hidden_states, done):

		self.obs[self.pointer] = obs
		self.actions[self.pointer] = action
		self.rewards[self.pointer] = reward
		self.next_obs[self.pointer] = next_obs

		self.hidden_states[self.pointer] = hidden_states
		self.next_hidden_states[self.pointer] = next_hidden_states
		# self.next_hidden_states[self.pointer] = next_hidden_states

		self.dones[self.pointer] = done
		self.pointer = (self.pointer + 1)%self.buffer_size
		self.len = min(self.len + 1, self.buffer_size)

