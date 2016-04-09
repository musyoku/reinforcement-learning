# -*- coding: utf-8 -*-
import numpy as np
from config import config

class Environment:
	def __init__(self):
		self.state = np.zeros((config.rl_chain_length,), dtype=np.float32)
		self.max_episodes = config.rl_chain_length - 4 + 9
		self.init()

	def init(self):
		self.current_episode = 0
		self.move_to(2)

	def agent_step(self, action):
		if action == config.actions[0]:
			self.move_right()
		elif action == config.actions[1]:
			self.move_left()
		else:
			pass	# no-op
		reward = self.get_reward()
		self.current_episode += 1
		ends = False if self.current_episode < self.max_episodes else True
		return self.state, reward, ends

	def move_to(self, index):
		if index < 0:
			return False
		if index >= config.rl_chain_length:
			return False
		self.state.fill(0.0)
		self.state[index] = 1.0
		return True

	def get_current_index(self):
		return np.argwhere(self.state == 1.0)[0]

	def get_current_state(self):
		return self.state

	def move_right(self):
		current_index = self.get_current_index()
		self.move_to(current_index + 1)

	def move_left(self):
		current_index = self.get_current_index()
		self.move_to(current_index - 1)

	def get_reward(self):
		current_index = self.get_current_index()
		if current_index == 1:
			return 1.0 / 1000.0
		if current_index == config.rl_chain_length - 2:
			return 1.0
		return 0.0