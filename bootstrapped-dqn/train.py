# -*- coding: utf-8 -*-
import os, sys, time
import numpy as np
from config import config
import model
from env import Environment

config.use_gpu = False
config.rl_model = "bootstrapped_double_dqn"

model = model.load()
env = Environment()

max_episode = 2000
total_steps = 0

if config.rl_model in ["bootstrapped_double_dqn"]:
	for episode in xrange(max_episode):
		env.init()
		k = np.random.randint(0, len(config.actions))
		print "episode:", episode, "k:", k
		while True:
			state = env.get_current_state()
			action, q = model.explore(state, k)
			next_state, reward, episode_ends = env.agent_step(action)
			mask = np.random.binomial(1, config.q_p_mask_sampling, (config.q_k_heads,))
			total_steps += 1
			print action, q, mask
			model.store_transition_in_replay_memory(state, action, reward, next_state, mask)
			if total_steps % config.rl_update_frequency == 0:
				model.replay_experience()
			if episode_ends is True:
				break
else:
	pass
