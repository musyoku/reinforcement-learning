# -*- coding: utf-8 -*-
import numpy as np
import math, pylab
import seaborn as sns

# 現在の状態から取れる行動を返す
## ここでは掛け金
def get_actions(state):
	actions = np.arange(1, min(state, 100 - state) + 1)
	return actions

# 価値関数
def get_value(state):
	return V[state]

# 方策
def get_policy(state):
	actions = get_actions(state)
	if len(actions) == 0:
		return None
	evaluations = eval(state)
	if evaluations is None:
		return None
	# 最大値を選択
	max_eval = 0.0
	max_action = -1
	for i in range((len(actions))):
		if evaluations[i] > max_eval:
			max_eval = evaluations[i]
			max_action = actions[i]
	return None if max_action == -1 else max_action

# 価値関数の更新
def update_value(state):
	evaluations = eval(state)
	if evaluations is None:
		return
	# 最大値を選択
	V[state] = np.amax(evaluations)

def eval(state):
	actions = get_actions(state)
	if len(actions) == 0:
		return None
	evaluations = np.zeros((len(actions),), dtype=np.float32)
	for i in range(len(actions)):
		action = actions[i]
		for new_state in range(0, 101):
			# 確率pでstateはstate + actionになり、1 - pでstate - actionになる
			## 従ってそれ以外のnew_stateが起こる確率は0なのでスキップ
			if new_state != state + action and new_state != state - action:
				continue
			if new_state == state + action:
				p_s_a_new_s = p
			else:
				p_s_a_new_s = 1.0 - p

			reward = 0.0
			if new_state == 100:
				reward = 1.0
			reward_expectation = reward / 101.0

			evaluations[i] += p_s_a_new_s * (reward_expectation + discount_rate * get_value(new_state))
	return evaluations

# 所持金
## 0と100はダミー
states = np.arange(0, 101)

# 状態価値関数
## ここではテーブル
V = np.zeros((101,), dtype=np.float32)
V[100] = 1.0

# コインの表が出る確率
p = 0.4

# 割引率
discount_rate = 1

# Δのしきい値
delta_threshold = 0.00001

# Plot
sns.set_style("ticks")
pylab.clf()
fig = pylab.gcf()
fig.set_size_inches(10.0, 6.0)

iteration = 1
while 1:
	# Δ
	delta = 0.0
	for state in range(1, 100):
		v = get_value(state)
		update_value(state)
		delta = max(delta, abs(v - get_value(state)))
	if delta < delta_threshold:
		break
	print iteration, delta
	if iteration == 1:
		pylab.plot(V, label="sweep 1")
	if iteration == 2:
		pylab.plot(V, label="sweep 2")
	if iteration == 3:
		pylab.plot(V, label="sweep 3")
	iteration += 1


for state in range(1, 100):
	print state, "->", V[state]

final_policy = np.zeros((100,), dtype=np.float32)
for state in range(1, 100):
	policy = get_policy(state)
	if policy is not None:
		final_policy[state] = policy
	print state, "->", get_policy(state)

pylab.plot(V, label=("sweep %d" % iteration))
ax = pylab.subplot(111)
ax.set_xticks(np.array([1, 25, 50, 75, 99]))
pylab.xlim([1, 99])
pylab.legend(loc="best")
pylab.xlabel("Capital")
pylab.ylabel("Value Estimates")
pylab.savefig("value_estimates.png")

pylab.clf()
pylab.step(np.arange(0, 100), final_policy)
ax = pylab.subplot(111)
ax.set_xticks(np.array([1, 25, 50, 75, 99]))
pylab.xlim([1, 99])
pylab.xlabel("Capital")
pylab.ylabel("Final Policy")
pylab.savefig("final_policy.png")