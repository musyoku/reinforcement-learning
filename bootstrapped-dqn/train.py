# -*- coding: utf-8 -*-
import os, sys, time
import numpy as np
from config import config
import model
from env import Environment

#############################################
config.use_gpu = True
#############################################

config.rl_model = "double_dqn"
config.rl_model = "bootstrapped_double_dqn"
config.rl_final_exploration_step = 20000
config.apply_batchnorm = False

model = model.load()
env = Environment()

max_episode = 2000
total_steps = 0
exploration_rate = config.rl_initial_exploration

dump_freq = 10
episode_rewards = 0
num_optimal_episodes = 0
sum_reward = 0
sum_loss = 0.0
save_freq = 100

bootstrapped = False
if config.rl_model in ["bootstrapped_double_dqn"]:
	bootstrapped = True

start_time = time.time()
print config.rl_model
for episode in xrange(max_episode):
	episode_rewards = 0
	env.init()
	k = np.random.randint(0, config.q_k_heads)
	# print "episode:", episode, "k:", k
	while True:
		state = env.get_current_state().copy()
		if bootstrapped:
			action, q = model.explore(state, k)
			next_state, reward, episode_ends = env.agent_step(action)
			mask = np.random.binomial(1, config.q_p_mask_sampling, (config.q_k_heads,))
		else:
			action, q = model.eps_greedy(state, exploration_rate=exploration_rate)
			next_state, reward, episode_ends = env.agent_step(action)
		# print np.array_str(state, max_line_width=1000000)
		total_steps += 1
		sum_reward += reward
		episode_rewards += reward
		if reward == 1.0:
			print "*"
		if bootstrapped:
			model.store_transition_in_replay_memory(state, action, reward, next_state, mask, episode_ends)
		else:
			model.store_transition_in_replay_memory(state, action, reward, next_state, episode_ends)
		loss = model.replay_experience()
		if loss == -1.0:
			pass # burn out
		else:
			sum_loss += loss
		if bootstrapped is False:
			exploration_rate = model.decrease_exploration_rate()
		if episode_ends is True:
			break
	if episode % dump_freq == 0 and episode != 0:
		if bootstrapped:
			print "episode:", episode, "reward:", sum_reward / float(dump_freq * env.max_episodes), "loss:", sum_loss / float(dump_freq), "optimal:", num_optimal_episodes
		else:
			print "episode:", episode, "reward:", sum_reward / float(dump_freq * env.max_episodes), "loss:", sum_loss / float(dump_freq), "eps:", "%.4f" % exploration_rate, "optimal:", num_optimal_episodes

		sum_reward = 0
		sum_loss = 0

	if episode % save_freq == 0 and episode != 0:
		model.save()

	if episode_rewards == 10:
		num_optimal_episodes += 1

	if num_optimal_episodes == 100:
		break

print "num_optimal_episodes:", num_optimal_episodes
print "total_time:", time.time() - start_time