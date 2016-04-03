# -*- coding: utf-8 -*-
import os, sys, time
import numpy as np
from config import config
import model
from env import Environment

config.use_gpu = False
config.rl_model = "bootstrapped_double_dqn"
config.rl_model = "double_dqn"
config.rl_final_exploration_step = 100

model = model.load()
env = Environment()

max_episode = 2000
total_steps = 0

dump_freq = 10
episode_rewards = 0
num_optimal_episodes = 0
sum_reward = 0
sum_loss = 0.0

save_freq = 100

if config.rl_model in ["bootstrapped_double_dqn"]:
	for episode in xrange(max_episode):
		episode_rewards = 0
		env.init()
		k = np.random.randint(0, len(config.actions))
		# print "episode:", episode, "k:", k
		while True:
			state = env.get_current_state()
			action, q = model.explore(state, k)
			next_state, reward, episode_ends = env.agent_step(action)
			mask = np.random.binomial(1, config.q_p_mask_sampling, (config.q_k_heads,))
			total_steps += 1
			sum_reward += reward
			episode_rewards += reward
			model.store_transition_in_replay_memory(state, action, reward, next_state, mask)
			if total_steps % config.rl_update_frequency == 0:
				sum_loss += model.replay_experience()
			if episode_ends is True:
				break
		if episode % dump_freq == 0:
			print "episode:", episode, "reward:", sum_reward / float(dump_freq), "loss:", sum_loss / float(dump_freq)
			sum_reward = 0
			sum_loss = 0

		if episode % save_freq == 0:
			model.save()

		if episode_rewards == 10:
			num_optimal_episodes += 1
else:
	exploration_rate = config.rl_initial_exploration
	for episode in xrange(max_episode):
		episode_rewards = 0
		env.init()
		k = np.random.randint(0, len(config.actions))
		# print "episode:", episode, "k:", k
		while True:
			state = env.get_current_state()
			action, q = model.eps_greedy(state, exploration_rate=exploration_rate)
			next_state, reward, episode_ends = env.agent_step(action)
			total_steps += 1
			sum_reward += reward
			episode_rewards += reward
			model.store_transition_in_replay_memory(state, action, reward, next_state)
			if total_steps % config.rl_update_frequency == 0:
				sum_loss += model.replay_experience()
				model.decrease_exploration_rate()
			if episode_ends is True:
				break
		if episode % dump_freq == 0:
			print "episode:", episode, "reward:", sum_reward / float(dump_freq), "loss:", sum_loss / float(dump_freq)
			sum_reward = 0
			sum_loss = 0

		if episode % save_freq == 0:
			model.save()

		if episode_rewards == 10:
			num_optimal_episodes += 1

print "num_optimal_episodes:", num_optimal_episodes