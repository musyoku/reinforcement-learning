# -*- coding: utf-8 -*-
import os, sys, time
import numpy as np
from config import config
import model
from env import Environment

config.rl_model = "bootstrapped_dqn"

model = model.load()
env = Environment()

max_episode = 2000

if config.rl_model in ["bootstrapped_dqn"]:
	for episode in xrange(max_episode):
		env.init()
		print "episode", episode
		k = np.random.randint(0, len(config.actions))
		print k
		while True:
			state = env.get_current_state()
			action, q = model.explore(state, k)
			print action, q
			new_state, reward, episode_ends = env.agent_step(action)
			if episode_ends is True:
				break
else:
	pass
