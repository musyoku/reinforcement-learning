# -*- coding: utf-8 -*-
import argparse
from activations import activations

class Config:
	def __init__(self):
		self.use_gpu = True
		self.apply_batchnorm = True

		# 11: rigtht
		# 22: left
		# 33: no-op
		# 44: no-op
		self.actions = [11, 22, 33, 44]


		# "dqn"
		# "double_dqn"
		# "dueling_double_dqn"
		# "bootstrapped_dqn"
		self.rl_model = "bootstrapped_dqn"

		self.rl_chain_length = 10

		self.rl_minibatch_size = 3
		self.rl_replay_memory_size = 10 ** 6
		self.rl_replay_start_size = 10 ** 5
		self.rl_target_network_update_frequency = 10 ** 4
		self.rl_discount_factor = 0.99
		self.rl_update_frequency = 1
		self.rl_learning_rate = 0.00025
		self.rl_gradient_momentum = 0.95
		self.rl_initial_exploration = 1.0
		self.rl_final_exploration = 0.1
		self.rl_final_exploration_step = 10 ** 6 * 2

		# Deep Q-Network
		## 全結合層の各レイヤのユニット数を入力側から出力側に向かって並べる
		## e.g 500(input vector)->250->100(output vector)
		## self.q_fc_units = [500, 250, 100]
		## For DQN / Double DQN / DDQN + Dueling Network
		self.q_fc_units = [self.rl_chain_length, 50, 50, len(self.actions)]
		## For Bootstrapped DQN
		self.q_bootstrapped_shared_fc_units = [self.rl_chain_length, 50]
		self.q_bootstrapped_head_fc_units = [self.q_bootstrapped_shared_fc_units[-1], 50, len(self.actions)]

		# Number of bootstrap heads
		self.q_k_heads = 10
		# We sample a bootstrap mask from Bernoulli(p) 
		self.q_p_mask_sampling = 0.5

		# Common
		## See activations.py
		self.q_fc_activation_function = "elu"

		self.q_fc_apply_dropout = False

		self.q_fc_apply_batchnorm_to_input = True

		## Default: 1.0
		self.q_wscale = 0.1

	def check(self):
		if self.q_fc_activation_function not in activations:
			raise Exception("Invalid activation function for q_fc_activation_function.")
		if len(self.q_fc_units) < 3:
			raise Exception("You need to add one or more hidden layers.")
		if len(self.q_bootstrapped_shared_fc_units) < 2:
			raise Exception("You need to add one or more hidden layers.")
		if len(self.q_bootstrapped_head_fc_units) < 3:
			raise Exception("You need to add one or more hidden layers.")
		if self.rl_replay_start_size > self.rl_replay_memory_size:
			self.rl_replay_start_size = self.rl_replay_memory_size

config = Config()
