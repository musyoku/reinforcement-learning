# -*- coding: utf-8 -*-
import os, time, collections, six
import numpy as np
import chainer, math, copy, os
from chainer import cuda, Variable, optimizers, serializers, function, link
from chainer.utils import type_check
from chainer import functions as F
from chainer import links as L
from activations import activations
from config import config

try:
	os.mkdir("model")
except:
	pass

def load():
	available_models = ["dqn", "double_dqn", "dueling_double_dqn", "bootstrapped_double_dqn"]
	if config.rl_model not in available_models:
		raise Exception("specified model is not available.")
	if config.rl_model == "dqn":
		return DQN()
	elif config.rl_model == "double_dqn":
		return DoubleDQN()
	elif config.rl_model == "dueling_double_dqn":
		return DuelingDoubleDQN()
	elif config.rl_model == "bootstrapped_double_dqn":
		return BootstrappedDoubleDQN()
	raise Exception("specified model is not available.")

def _as_mat(x):
	if x.ndim == 2:
		return x
	return x.reshape(len(x), -1)

class LinearHeadFunction(function.Function):

	def __init__(self, W_mask):
		self.W_mask = W_mask

	def check_type_forward(self, in_types):
		n_in = in_types.size()
		type_check.expect(2 <= n_in, n_in <= 3)
		x_type, w_type = in_types[:2]

		type_check.expect(
			x_type.dtype == np.float32,
			w_type.dtype == np.float32,
			x_type.ndim >= 2,
			w_type.ndim == 2,
			type_check.prod(x_type.shape[1:]) == w_type.shape[1],
		)
		if n_in.eval() == 3:
			b_type = in_types[2]
			type_check.expect(
				b_type.dtype == np.float32,
				b_type.ndim == 1,
				b_type.shape[0] == w_type.shape[0],
			)

	def forward(self, inputs):
		x = _as_mat(inputs[0])
		W = inputs[1] * self.W_mask
		y = x.dot(W.T)
		if len(inputs) == 3:
			b = inputs[2]
			y += b
		return y,

	def backward(self, inputs, grad_outputs):
		x = _as_mat(inputs[0])
		W = inputs[1] * self.W_mask
		gy = grad_outputs[0]

		gx = gy.dot(W).reshape(inputs[0].shape)
		gW = gy.T.dot(x)
		if len(inputs) == 3:
			gb = gy.sum(0)
			return gx, gW, gb
		else:
			return gx, gW


def linear_head(x, W, W_mask, b=None):
	if b is None:
		return LinearHeadFunction(W_mask)(x, W)
	else:
		return LinearHeadFunction(W_mask)(x, W, b)

class LinearHead(link.Link):

	def __init__(self, head_in_size, head_out_size, num_heads, wscale=1, bias=0, nobias=False, initialW=None, initial_bias=None):
		in_size = head_in_size * num_heads
		out_size = head_out_size * num_heads
		super(LinearHead, self).__init__(W=(out_size, in_size))
		if initialW is None:
			initialW = np.random.normal(0, wscale * np.sqrt(1. / in_size), (out_size, in_size))
		self.W.data[...] = initialW

		W_mask = np.zeros((out_size, in_size), dtype=np.float32)
		for i in xrange(num_heads):
			W_mask[i * head_out_size:(i + 1) * head_out_size, i * head_in_size:(i + 1) * head_in_size] = 1.0

		self.add_persistent("W_mask", W_mask)
		if nobias:
			self.b = None
		else:
			self.add_param('b', out_size)
			if initial_bias is None:
				initial_bias = bias
			self.b.data[...] = initial_bias
	
	def __call__(self, x):
		return linear_head(x, self.W, self.W_mask, self.b)


class BatchNormalization(L.BatchNormalization):
	def __init__(self, size, decay=0.9, eps=1e-5, dtype=np.float32):
		super(L.BatchNormalization, self).__init__()
		self.add_param('gamma', size, dtype=dtype)
		self.gamma.data.fill(1)
		self.add_param('beta', size, dtype=dtype)
		self.beta.data.fill(0)
		self.add_persistent('avg_mean', np.zeros(size, dtype=dtype))
		self.add_persistent('avg_var', np.ones(size, dtype=dtype))
		self.add_persistent('N', 0)
		self.decay = decay
		self.eps = eps

class FullyConnectedNetwork(chainer.Chain):
	def __init__(self, **layers):
		super(FullyConnectedNetwork, self).__init__(**layers)
		self.n_hidden_layers = 0
		self.activation_function = "elu"
		self.apply_batchnorm_to_input = False
		self.apply_batchnorm_to_output = False
		self.apply_batchnorm = False

	def forward_one_step(self, x, test):
		f = activations[self.activation_function]
		chain = [x]

		# Hidden layers
		for i in range(self.n_hidden_layers):
			u = getattr(self, "layer_%i" % i)(chain[-1])
			if self.apply_batchnorm:
				if i == 0 and self.apply_batchnorm_to_input is False:
					pass
				else:
					u = getattr(self, "batchnorm_%i" % i)(u, test=test)
			output = f(u)
			if self.apply_dropout:
				output = F.dropout(output, train=not test)
			chain.append(output)

		# Output
		u = getattr(self, "layer_%i" % self.n_hidden_layers)(chain[-1])
		if self.apply_batchnorm_to_output:
			u = getattr(self, "batchnorm_%i" % self.n_hidden_layers)(u, test=test)
		chain.append(f(u))

		return chain[-1]

	def __call__(self, x, test=False):
		return self.forward_one_step(x, test=test)


def _sum_sqnorm(arr):
	sq_sum = collections.defaultdict(float)
	for x in arr:
		with cuda.get_device(x) as dev:
			x = x.ravel()
			s = x.dot(x)
			sq_sum[int(dev)] += s
	return sum([float(i) for i in six.itervalues(sq_sum)])
	
class GradientClipping(object):
	name = 'GradientClipping'

	def __init__(self, threshold):
		self.threshold = threshold

	def __call__(self, opt):
		norm = np.sqrt(_sum_sqnorm([p.grad for p in opt.target.params()]))
		if norm == 0:
			norm = 1
		rate = self.threshold / norm
		if rate < 1:
			for param in opt.target.params():
				grad = param.grad
				with cuda.get_device(grad):
					grad *= rate

class Model:
	def __init__(self):
		self.exploration_rate = config.rl_initial_exploration

		# Replay Memory
		## (state, action, reward, next_state, episode_ends)
		shape_state = (config.rl_replay_memory_size, config.rl_chain_length)
		shape_action = (config.rl_replay_memory_size,)
		self.replay_memory = [
			np.zeros(shape_state, dtype=np.float32),
			np.zeros(shape_action, dtype=np.uint8),
			np.zeros(shape_action, dtype=np.float32),
			np.zeros(shape_state, dtype=np.float32),
			np.zeros(shape_action, dtype=np.bool)
		]
		self.total_replay_memory = 0
		
	def store_transition_in_replay_memory(self, state, action, reward, next_state, episode_ends):
		index = self.total_replay_memory % config.rl_replay_memory_size
		if self.replay_memory[0][index].shape != state.shape:
			raise Exception()
		if self.replay_memory[3][index].shape != next_state.shape:
			raise Exception()
		self.replay_memory[0][index] = state
		self.replay_memory[1][index] = action
		self.replay_memory[2][index] = reward
		self.replay_memory[3][index] = next_state
		self.replay_memory[4][index] = episode_ends
		self.total_replay_memory += 1

	def get_replay_memory_size(self):
		return min(self.total_replay_memory, config.rl_replay_memory_size)

	def get_action_for_index(self, i):
		return config.actions[i]

	def get_index_for_action(self, action):
		return config.actions.index(action)

	def decrease_exploration_rate(self):
		self.exploration_rate = max(self.exploration_rate - 1.0 / config.rl_final_exploration_step, config.rl_final_exploration)
		return self.exploration_rate

class DQN(Model):
	def __init__(self):
		Model.__init__(self)

		self.fc = self.build_network()

		self.optimizer_fc = optimizers.Adam(alpha=config.rl_learning_rate, beta1=config.rl_gradient_momentum)
		self.optimizer_fc.setup(self.fc)
		self.optimizer_fc.add_hook(GradientClipping(10.0))

		self.update_target()

	def build_network(self):
		config.check()
		wscale = config.q_wscale

		# Fully connected part of Q-Network
		fc_attributes = {}
		fc_units = zip(config.q_fc_units[:-1], config.q_fc_units[1:])

		for i, (n_in, n_out) in enumerate(fc_units):
			fc_attributes["layer_%i" % i] = L.Linear(n_in, n_out, wscale=wscale)
			fc_attributes["batchnorm_%i" % i] = BatchNormalization(n_out)

		fc = FullyConnectedNetwork(**fc_attributes)
		fc.n_hidden_layers = len(fc_units) - 1
		fc.activation_function = config.q_fc_activation_function
		fc.apply_batchnorm = config.apply_batchnorm
		fc.apply_dropout = config.q_fc_apply_dropout
		fc.apply_batchnorm_to_input = config.q_fc_apply_batchnorm_to_input
		if config.use_gpu:
			fc.to_gpu()
		return fc

	def eps_greedy(self, state_batch, exploration_rate):
		if state_batch.ndim == 1:
			state_batch = state_batch.reshape(1, -1)
		elif state_batch.ndim == 3:
			state_batch = state_batch.reshape(-1, 34 * config.rl_history_length)
		prop = np.random.uniform()
		if prop < exploration_rate:
			action_batch = np.random.randint(0, len(config.actions), (state_batch.shape[0],))
			q = None
		else:
			state_batch = Variable(state_batch)
			if config.use_gpu:
				state_batch.to_gpu()
			q = self.compute_q_variable(state_batch, test=True)
			if config.use_gpu:
				q.to_cpu()
			q = q.data
			action_batch = np.argmax(q, axis=1)
		for i in xrange(action_batch.shape[0]):
			action_batch[i] = self.get_action_for_index(action_batch[i])
		return action_batch, q

	def forward_one_step(self, state, action, reward, next_state, test=False):
		xp = cuda.cupy if config.use_gpu else np
		n_batch = state.shape[0]
		state = Variable(state)
		next_state = Variable(next_state)
		if config.use_gpu:
			state.to_gpu()
			next_state.to_gpu()
		q = self.compute_q_variable(state, test=test)

		max_target_q = self.compute_target_q_variable(next_state, test=test)
		max_target_q = xp.amax(max_target_q.data, axis=1)

		target = q.data.copy()

		for i in xrange(n_batch):
			if episode_ends[i] is True:
				target_value = np.sign(reward[i])
			else:
				target_value = np.sign(reward[i]) + config.rl_discount_factor * max_target_q[i]
			action_index = self.get_index_with_action(action[i])
			old_value = target[i, action_index]
			diff = target_value - old_value
			if diff > 1.0:
				target_value = 1.0 + old_value	
			elif diff < -1.0:
				target_value = -1.0 + old_value	
			target[i, action_index] = target_value

		target = Variable(target)

		loss = F.mean_squared_error(target, q)
		return loss, q

	def replay_experience(self):
		if self.total_replay_memory == 0:
			return
		if self.total_replay_memory < config.rl_replay_memory_size:
			replay_index = np.random.randint(0, self.total_replay_memory, (config.rl_minibatch_size, 1))
		else:
			replay_index = np.random.randint(0, config.rl_replay_memory_size, (config.rl_minibatch_size, 1))

		shape_state = (config.rl_minibatch_size, config.rl_chain_length)
		shape_action = (config.rl_minibatch_size,)

		state = np.empty(shape_state, dtype=np.float32)
		action = np.empty(shape_action, dtype=np.uint8)
		reward = np.empty(shape_action, dtype=np.float32)
		next_state = np.empty(shape_state, dtype=np.float32)
		episode_ends = np.empty(shape_action, dtype=np.bool)
		for i in xrange(config.rl_minibatch_size):
			state[i] = self.replay_memory[0][replay_index[i]]
			action[i] = self.replay_memory[1][replay_index[i]]
			reward[i] = self.replay_memory[2][replay_index[i]]
			next_state[i] = self.replay_memory[3][replay_index[i]]
			episode_ends[i] = self.replay_memory[4][replay_index[i]]

		self.optimizer_zero_grads()
		loss, _ = self.forward_one_step(state, action, reward, next_state, episode_ends, test=False)
		loss.backward()
		self.optimizer_update()
		return loss.data

	def optimizer_zero_grads(self):
		self.optimizer_fc.zero_grads()

	def optimizer_update(self):
		self.optimizer_fc.update()

	def compute_q_variable(self, state, test=False):
		return self.fc(state, test=test)

	def compute_target_q_variable(self, state, test=True):
		return self.target_fc(state, test=test)

	def update_target(self):
		self.target_fc = copy.deepcopy(self.fc)

	def load(self):
		dir = "model"
		filename = dir + "/dqn_fc.model"
		if os.path.isfile(filename):
			serializers.load_hdf5(filename, self.fc)
			print "model loaded successfully."
		filename = dir + "/dqn_fc.optimizer"
		if os.path.isfile(filename):
			serializers.load_hdf5(filename, self.optimizer_fc)
			print "optimizer loaded successfully."

	def save(self):
		dir = "model"
		serializers.save_hdf5(dir + "/dqn_fc.model", self.fc)
		print "model saved."
		serializers.save_hdf5(dir + "/dqn_fc.optimizer", self.optimizer_fc)
		print "optimizer saved."

class DoubleDQN(DQN):
	
	def forward_one_step(self, state, action, reward, next_state, episode_ends, test=False):
		xp = cuda.cupy if config.use_gpu else np
		n_batch = state.shape[0]
		state = Variable(state)
		next_state = Variable(next_state)
		if config.use_gpu:
			state.to_gpu()
			next_state.to_gpu()
		q = self.compute_q_variable(state, test=test)
		q_ = self.compute_q_variable(next_state, test=test)
		max_action_indices = xp.argmax(q_.data, axis=1)
		if config.use_gpu:
			max_action_indices = cuda.to_cpu(max_action_indices)

		target_q = self.compute_target_q_variable(next_state, test=test)

		target = q.data.copy()

		for i in xrange(n_batch):
			max_action_index = max_action_indices[i]
			if episode_ends[i] is True:
				target_value = reward[i]
			else:
				target_value = reward[i] + config.rl_discount_factor * target_q.data[i][max_action_indices[i]]
			action_index = self.get_index_for_action(action[i])
			old_value = target[i, action_index]
			diff = target_value - old_value
			if diff > 1.0:
				target_value = 1.0 + old_value	
			elif diff < -1.0:
				target_value = -1.0 + old_value	
			target[i, action_index] = target_value

		target = Variable(target)
		loss = F.mean_squared_error(target, q)
		return loss, q

		
class GradientNormalizing(object):
	name = 'GradientNormalizing'

	def __init__(self, threshold):
		self.threshold = threshold

	def __call__(self, opt):
		for param in opt.target.params():
			grad = param.grad
			with cuda.get_device(grad):
				grad *= self.threshold

class BootstrappedDoubleDQN(DQN):

	def __init__(self):
		self.exploration_rate = config.rl_initial_exploration

		# Replay Memory
		## (state, action, reward, next_state, episode_ends)
		shape_state = (config.rl_replay_memory_size, config.rl_chain_length)
		shape_action = (config.rl_replay_memory_size,)
		shape_mask = (config.rl_replay_memory_size, config.q_k_heads)
		self.replay_memory = [
			np.zeros(shape_state, dtype=np.float32),
			np.zeros(shape_action, dtype=np.uint8),
			np.zeros(shape_action, dtype=np.float32),
			np.zeros(shape_state, dtype=np.float32),
			np.zeros(shape_mask, dtype=np.uint8),
			np.zeros(shape_action, dtype=np.bool)
		]
		self.total_replay_memory = 0

		self.shared_fc = self.build_network(config.q_bootstrapped_shared_fc_units)
		self.target_shared_fc = copy.deepcopy(self.shared_fc)
		self.optimizer_shared_fc = optimizers.Adam(alpha=config.rl_learning_rate, beta1=config.rl_gradient_momentum)
		self.optimizer_shared_fc.setup(self.shared_fc)
		self.optimizer_shared_fc.add_hook(GradientClipping(10.0))
		self.optimizer_shared_fc.add_hook(GradientNormalizing(1.0 / config.q_k_heads))

		self.head_fc = self.build_head(config.q_k_heads, config.q_bootstrapped_head_fc_units)
		self.target_head_fc = copy.deepcopy(self.head_fc)
		self.optimizer_head_fc = optimizers.Adam(alpha=config.rl_learning_rate, beta1=config.rl_gradient_momentum)
		self.optimizer_head_fc.setup(self.head_fc)
		self.optimizer_head_fc.add_hook(GradientClipping(10.0))

	def store_transition_in_replay_memory(self, state, action, reward, next_state, mask, episode_ends):
		index = self.total_replay_memory % config.rl_replay_memory_size
		if self.replay_memory[0][index].shape != state.shape:
			raise Exception()
		if self.replay_memory[3][index].shape != next_state.shape:
			raise Exception()
		if self.replay_memory[4][index].shape != mask.shape:
			raise Exception()
		self.replay_memory[0][index] = state
		self.replay_memory[1][index] = action
		self.replay_memory[2][index] = reward
		self.replay_memory[3][index] = next_state
		self.replay_memory[4][index] = mask
		self.replay_memory[5][index] = episode_ends
		self.total_replay_memory += 1

	def build_network(self, units=None):
		if units is None:
			raise Exception()
		config.check()
		wscale = config.q_wscale

		# Fully connected part of Q-Network
		fc_attributes = {}
		units[-1] *= config.q_k_heads
		fc_units = zip(units[:-1], units[1:])

		for i, (n_in, n_out) in enumerate(fc_units):
			fc_attributes["layer_%i" % i] = L.Linear(n_in, n_out, wscale=wscale)
			fc_attributes["batchnorm_%i" % i] = BatchNormalization(n_out)

		fc = FullyConnectedNetwork(**fc_attributes)
		fc.n_hidden_layers = len(fc_units) - 1
		fc.activation_function = config.q_fc_activation_function
		fc.apply_batchnorm = config.apply_batchnorm
		fc.apply_dropout = config.q_fc_apply_dropout
		fc.apply_batchnorm_to_input = config.q_fc_apply_batchnorm_to_input
		if config.use_gpu:
			fc.to_gpu()
		return fc

	def build_head(self, k=0, units=None):
		if units is None:
			raise Exception()
		config.check()
		wscale = config.q_wscale

		# Fully connected part of Q-Network
		fc_attributes = {}
		fc_units = zip(units[:-1], units[1:])

		for i, (n_in, n_out) in enumerate(fc_units):
			fc_attributes["layer_%i" % i] = LinearHead(n_in, n_out, config.q_k_heads, wscale=wscale)
			fc_attributes["batchnorm_%i" % i] = BatchNormalization(n_out)

		fc = FullyConnectedNetwork(**fc_attributes)
		fc.n_hidden_layers = len(fc_units) - 1
		fc.activation_function = config.q_fc_activation_function
		fc.apply_dropout = config.q_fc_apply_dropout
		if config.use_gpu:
			fc.to_gpu()
		return fc

	def explore(self, state, k=0, test=True):
		if state.ndim == 1:
			state = state.reshape(1, -1)
		action_batch, q_batch = self.explore_batch(state, k, test=test)
		return action_batch[0], q_batch[0]

	def explore_batch(self, state_batch, k=0, test=True):
		if state_batch.ndim == 1:
			state_batch = state_batch.reshape(1, -1)
		state_batch = Variable(state_batch)
		if config.use_gpu:
			state_batch.to_gpu()
		q_batch = self.compute_q(state_batch, k, test=test)
		if config.use_gpu:
			q_batch = cuda.to_cpu(q_batch)
		action_batch = np.argmax(q_batch, axis=1)
		for i in xrange(action_batch.shape[0]):
			action_batch[i] = self.get_action_for_index(action_batch[i])
		return action_batch, q_batch
	
	def forward_one_step(self, state, action, reward, next_state, mask, episode_ends, test=False):
		start_time = time.time()
		xp = cuda.cupy if config.use_gpu else np
		n_batch = state.shape[0]
		state = Variable(state)
		next_state = Variable(next_state)
		if config.use_gpu:
			state.to_gpu()
			next_state.to_gpu()
		q_combined = self.compute_q_variable_of_all_head(state, to_cpu=False, test=test)
		next_q_combined = self.compute_q_variable_of_all_head(next_state, to_cpu=False, test=test)
		max_action_indices_array = self.argmax_q_variable_of_all_head(next_q_combined, cpu=True)

		target_q_combined = self.compute_target_q_variable_of_all_head(next_state, test=test)
		target_combined = q_combined.data.copy()
		if config.use_gpu:
			target_combined = cuda.to_cpu(target_combined)
			target_q_combined.to_cpu()

		length = config.q_bootstrapped_head_fc_units[-1]

		ks = np.arange(config.q_k_heads, dtype=np.uint8)
		for i in xrange(n_batch):
			if episode_ends[i] is True:
				target_values = np.full((config.q_k_heads,), reward[i], dtype=np.float32)
			else:
				pos = ks*length+max_action_indices_array[i,ks]
				target_values = reward[i] + config.rl_discount_factor * target_q_combined.data[i, pos]
			action_index = self.get_index_for_action(action[i])
			old_values = target_combined[i, ks*length+action_index]
			diff = target_values - old_values
			diff = np.clip(diff, -1.0, 1.0)
			target_values = diff + old_values
			target_values *= mask[i]
			target_combined[i, ks*length+action_index] = target_values

		target_combined = Variable(target_combined)
		if config.use_gpu:
			target_combined.to_gpu()
		loss = F.mean_squared_error(target_combined, q_combined)
		return loss, q_combined

	def replay_experience(self):
		if self.total_replay_memory == 0:
			return -1.0
		if self.total_replay_memory < config.rl_minibatch_size:
			return -1.0
		minibatch_size = config.rl_minibatch_size
		if self.total_replay_memory < config.rl_replay_memory_size:
			replay_index = np.random.randint(0, self.total_replay_memory, (minibatch_size, 1))
		else:
			replay_index = np.random.randint(0, config.rl_replay_memory_size, (minibatch_size, 1))

		shape_state = (minibatch_size, config.rl_chain_length)
		shape_action = (minibatch_size,)
		shape_mask = (minibatch_size, config.q_k_heads)

		state = np.empty(shape_state, dtype=np.float32)
		action = np.empty(shape_action, dtype=np.uint8)
		reward = np.empty(shape_action, dtype=np.float32)
		next_state = np.empty(shape_state, dtype=np.float32)
		mask = np.empty(shape_mask, dtype=np.float32)
		episode_ends = np.empty(shape_action, dtype=np.float32)
		for i in xrange(len(replay_index)):
			state[i] = self.replay_memory[0][replay_index[i]]
			action[i] = self.replay_memory[1][replay_index[i]]
			reward[i] = self.replay_memory[2][replay_index[i]]
			next_state[i] = self.replay_memory[3][replay_index[i]]
			mask[i] = self.replay_memory[4][replay_index[i]]
			episode_ends[i] = self.replay_memory[5][replay_index[i]]

		self.optimizer_zero_grads()
		loss, _ = self.forward_one_step(state, action, reward, next_state, mask, episode_ends, test=False)
		loss.backward()
		self.optimizer_update()
		return loss.data

	def optimizer_zero_grads(self):
		self.optimizer_shared_fc.zero_grads()
		self.optimizer_head_fc.zero_grads()

	def optimizer_update(self):
		self.optimizer_shared_fc.update()
		self.optimizer_head_fc.update()

	def compute_q(self, state, k=0, test=False):
		if state.data.ndim != 2:
			raise Exception()
		shared_output = self.shared_fc(state, test=test)
		output = self.head_fc(shared_output, test=test)
		length = config.q_bootstrapped_head_fc_units[-1]
		output = output.data[:,k*length:(k+1)*length]
		if output.shape[1] != length:
			raise Exception()
		return output

	def compute_target_q(self, state, k=0, test=True):
		if state.data.ndim != 2:
			raise Exception()
		shared_output = self.target_shared_fc(state, test=test)
		output = self.target_head_fc(shared_output, test=test)
		length = config.q_bootstrapped_head_fc_units[-1]
		output = output.data[:,k*length:(k+1)*length]
		if output.shape[1] != length:
			raise Exception()
		return output

	def argmax_q_variable_of_all_head(self, q, cpu=False):
		xp = cuda.get_array_module(q.data)
		length = config.q_bootstrapped_head_fc_units[-1]
		result = xp.empty((q.data.shape[0], config.q_k_heads), dtype=xp.uint8)
		for i in xrange(config.q_k_heads):
			argmax = xp.argmax(q.data[:,i*length:(i+1)*length], axis=1)
			result[:,i] = argmax
		if cpu:
			result = cuda.to_cpu(result)
		return result

	def compute_q_variable_of_all_head(self, state, to_cpu=True, test=False):
		shared_output = self.shared_fc(state, test=test)
		return self.head_fc(shared_output)

	def compute_target_q_variable_of_all_head(self, state, to_cpu=True, test=True):
		shared_output = self.target_shared_fc(state, test=test)
		return self.target_head_fc(shared_output)

	def update_target(self):
		self.target_shared_fc = copy.deepcopy(self.shared_fc)
		self.target_head_fc = copy.deepcopy(self.head_fc)

	def load(self):
		dir = "model"
		filename = dir + "/bddqn_shared_fc.model"
		if os.path.isfile(filename):
			serializers.load_hdf5(filename, self.shared_fc)
			print "model shared_fc loaded successfully."

		filename = dir + "/bddqn_head_fc.model"
		if os.path.isfile(filename):
			serializers.load_hdf5(filename, self.head_fc)
			print "model head_fc loaded successfully."

		filename = dir + "/bddqn_shared_fc.optimizer"
		if os.path.isfile(filename):
			serializers.load_hdf5(filename, self.optimizer_shared_fc)
			print "optimizer shared_fc loaded successfully."

		filename = dir + "/bddqn_head_fc.optimizer"
		if os.path.isfile(filename):
			serializers.load_hdf5(filename, self.optimizer_head_fc)
			print "optimizer head_fc loaded successfully."


	def save(self):
		dir = "model"
		serializers.save_hdf5(dir + "/bddqn_shared_fc.model", self.shared_fc)
		serializers.save_hdf5(dir + "/bddqn_shared_fc.optimizer", self.optimizer_shared_fc)
		serializers.save_hdf5(dir + "/bddqn_head_fc.model", self.head_fc)
		serializers.save_hdf5(dir + "/bddqn_head_fc.optimizer", self.optimizer_head_fc)
