# -*- coding: utf-8 -*-
import os
import numpy as np
import chainer, math, copy, os
from chainer import cuda, Variable, optimizers, serializers, function
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
		if self.apply_batchnorm:
			u = getattr(self, "batchnorm_%i" % self.n_hidden_layers)(u, test=test)
		chain.append(f(u))

		return chain[-1]

	def __call__(self, x, test=False):
		return self.forward_one_step(x, test=test)

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
			np.zeros(shape_state, dtype=np.float32)
		]
		self.total_replay_memory = 0
		
	def store_transition_in_replay_memory(self, state, action, reward, next_state):
		index = self.total_replay_memory % config.rl_replay_memory_size
		if self.replay_memory[0][index].shape != state.shape:
			raise Exception()
		if self.replay_memory[3][index].shape != next_state.shape:
			raise Exception()
		self.replay_memory[0][index] = state
		self.replay_memory[1][index] = action
		self.replay_memory[2][index] = reward
		self.replay_memory[3][index] = next_state
		self.total_replay_memory += 1

	def get_replay_memory_size(self):
		return min(self.total_replay_memory, config.rl_replay_memory_size)

	def get_action_for_index(self, i):
		return config.actions[i]

	def get_index_for_action(self, action):
		return config.actions.index(action)

	def decrease_exploration_rate(self):
		self.exploration_rate = max(self.exploration_rate - 1.0 / config.rl_final_exploration_step, config.rl_final_exploration)

class DQN(Model):
	def __init__(self):
		Model.__init__(self)

		self.fc = self.build_network(output_dim=len(config.actions))

		self.optimizer_fc = optimizers.Adam(alpha=config.rl_learning_rate, beta1=config.rl_gradient_momentum)
		self.optimizer_fc.setup(self.fc)
		self.optimizer_fc.add_hook(optimizer.GradientClipping(10.0))

		self.load()
		self.update_target()

	def build_network(self, input_ndim=None, fc_hidden_units=None, output_ndim=1):
		config.check()
		if fc_hidden_units is None:
			fc_hidden_units = config.q_fc_hidden_units
		if input_ndim is None:
			input_ndim = config.rl_chain_length

		wscale = config.q_wscale

		# Fully connected part of Q-Network
		fc_attributes = {}
		fc_units = [(input_ndim, fc_hidden_units[0])]
		fc_units += zip(fc_hidden_units[:-1], fc_hidden_units[1:])
		fc_units += [(fc_hidden_units[-1], output_ndim)]

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
		reward = np.empty(shape_action, dtype=np.int8)
		next_state = np.empty(shape_state, dtype=np.float32)
		for i in xrange(config.rl_minibatch_size):
			state[i] = self.replay_memory[0][replay_index[i]]
			action[i] = self.replay_memory[1][replay_index[i]]
			reward[i] = self.replay_memory[2][replay_index[i]]
			next_state[i] = self.replay_memory[3][replay_index[i]]

		self.optimizer_zero_grads()
		loss, _ = self.forward_one_step(state, action, reward, next_state, test=False)
		loss.backward()
		self.optimizer_update()
		return loss

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
		filename = "fc.model"
		if os.path.isfile(filename):
			serializers.load_hdf5(filename, self.fc)
			print "model loaded successfully."
		filename = "fc.optimizer"
		if os.path.isfile(filename):
			serializers.load_hdf5(filename, self.optimizer_fc)
			print "optimizer loaded successfully."

	def save(self):
		serializers.save_hdf5("fc.model", self.fc)
		print "model saved."
		serializers.save_hdf5("fc.optimizer", self.optimizer_fc)
		print "optimizer saved."

class DoubleDQN(DQN):
	
	def forward_one_step(self, state, action, reward, next_state, test=False):
		xp = cuda.cupy if config.use_gpu else np
		n_batch = state.shape[0]
		state = Variable(state.reshape((n_batch, config.rl_history_length * 34)))
		next_state = Variable(next_state.reshape((n_batch, config.rl_history_length * 34)))
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
	name = 'GradientClipping'

	def __init__(self, threshold):
		self.threshold = threshold

	def __call__(self, opt):
		norm = numpy.sqrt(_sum_sqnorm([p.grad for p in opt.target.params()]))
		rate = self.threshold / norm
		if rate < 1:
			for param in opt.target.params():
				grad = param.grad
				with cuda.get_device(grad):
					grad *= rate

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
		]
		self.total_replay_memory = 0


		self.shared_fc = self.build_network(units=config.q_bootstrapped_shared_fc_units)
		self.target_shared_fc = copy.deepcopy(self.shared_fc)
		self.optimizer_shared_fc = optimizers.Adam(alpha=config.rl_learning_rate, beta1=config.rl_gradient_momentum)
		self.optimizer_shared_fc.setup(self.shared_fc)
		self.optimizer_shared_fc.add_hook(chainer.optimizer.GradientClipping(10.0))

		self.head_fc_array = []
		self.target_head_fc_array = []
		self.optimizer_head_fc_array = []
		for n in xrange(config.q_k_heads):
			network = self.build_network(units=config.q_bootstrapped_head_fc_units)
			optimizer = optimizers.Adam(alpha=config.rl_learning_rate, beta1=config.rl_gradient_momentum)
			optimizer.setup(network)
			optimizer.add_hook(chainer.optimizer.GradientClipping(10.0))
			self.head_fc_array.append(network)
			self.target_head_fc_array.append(copy.deepcopy(network))
			self.optimizer_head_fc_array.append(optimizer)

		self.load()
		
	def store_transition_in_replay_memory(self, state, action, reward, next_state, mask):
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
		self.total_replay_memory += 1

	def build_network(self, units=None):
		if units is None:
			raise Exception()
		config.check()
		wscale = config.q_wscale

		# Fully connected part of Q-Network
		fc_attributes = {}
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
		q_batch = self.compute_q_variable(state_batch, k, test=test)
		if config.use_gpu:
			q_batch.to_cpu()
		q_batch = q_batch.data
		action_batch = np.argmax(q_batch, axis=1)
		for i in xrange(action_batch.shape[0]):
			action_batch[i] = self.get_action_for_index(action_batch[i])
		return action_batch, q_batch
	
	def forward_one_step(self, state, action, reward, next_state, mask, test=False):
		xp = cuda.cupy if config.use_gpu else np
		n_batch = state.shape[0]
		state = Variable(state)
		next_state = Variable(next_state)
		if config.use_gpu:
			state.to_gpu()
			next_state.to_gpu()
		q_array = self.compute_q_variable_of_all_head(state, test=test)
		next_q_array = self.compute_q_variable_of_all_head(next_state, test=test)
		max_action_indices_array = self.compute_q_variable_of_all_head(next_q_array)

		target_q_array = self.compute_target_q_variable_of_all_head(next_state, test=test)

		print mask

		target = q.data.copy()

		for i in xrange(n_batch):
			max_action_index = max_action_indices[i]
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

	def replay_experience(self):
		if self.total_replay_memory == 0:
			return
		if self.total_replay_memory < config.rl_replay_memory_size:
			replay_index = np.random.randint(0, self.total_replay_memory, (config.rl_minibatch_size, 1))
		else:
			replay_index = np.random.randint(0, config.rl_replay_memory_size, (config.rl_minibatch_size, 1))

		shape_state = (config.rl_minibatch_size, config.rl_chain_length)
		shape_action = (config.rl_minibatch_size,)
		shape_mask = (config.rl_replay_memory_size, config.q_k_heads)

		state = np.empty(shape_state, dtype=np.float32)
		action = np.empty(shape_action, dtype=np.uint8)
		reward = np.empty(shape_action, dtype=np.int8)
		next_state = np.empty(shape_state, dtype=np.float32)
		mask = np.empty(shape_mask, dtype=np.int8)
		for i in xrange(config.rl_minibatch_size):
			state[i] = self.replay_memory[0][replay_index[i]]
			action[i] = self.replay_memory[1][replay_index[i]]
			reward[i] = self.replay_memory[2][replay_index[i]]
			next_state[i] = self.replay_memory[3][replay_index[i]]
			mask[i] = self.replay_memory[4][replay_index[i]]

		self.optimizer_zero_grads()
		loss, _ = self.forward_one_step(state, action, reward, next_state, mask, test=False)
		loss.backward()
		self.optimizer_update()
		return loss

	def optimizer_zero_grads(self):
		self.optimizer_shared_fc.zero_grads()
		for i, optimizer in enumerate(self.optimizer_head_fc_array):
			optimizer.zero_grads()

	def optimizer_update(self):
		self.optimizer_fc_value.update()
		self.optimizer_fc_advantage.update()

	def compute_q_variable(self, state, k=0, test=False):
		shared_output = self.shared_fc(state, test=test)
		head = self.head_fc_array[k]
		output = head(shared_output, test=test)
		return output

	def compute_target_q_variable(self, state, k=0, test=True):
		shared_output = self.target_shared_fc(state, test=test)
		head = self.target_head_fc_array[k]
		output = head(shared_output, test=test)
		return output

	def argmax_q_variable_of_all_head(self, q_array):
		argmax_array = []
		for i, q in enumerate(q_array):
			argmax = np.argmax(q.data, axis=1)
			argmax_array.push(argmax)
		return argmax_array

	def compute_q_variable_of_all_head(self, state, to_cpu=True, test=False):
		shared_output = self.shared_fc(state, test=test)
		q_array = []
		for i, head in enumerate(self.head_fc_array):
			output = head(shared_output)
			if to_cpu:
				output.to_cpu()
			q_array.append(output)
		return q_array

	def compute_target_q_variable_of_all_head(self, state, to_cpu=True, test=True):
		shared_output = self.target_shared_fc(state, test=test)
		q_array = []
		for i, head in enumerate(self.target_head_fc_array):
			output = head(shared_output)
			if to_cpu:
				output.to_cpu()
			q_array.append(output)
		return q_array

	def update_target(self):
		self.target_shared_fc = copy.deepcopy(self.shared_fc)
		for i, network in enumerate(self.head_fc_array):
			self.head_fc_array[i] = copy.deepcopy(self.head_fc_array[i])

	def load(self):
		dir = "model"
		filename = dir + "/shared_fc.model"
		if os.path.isfile(filename):
			serializers.load_hdf5(filename, self.shared_fc)
			print "model shared_fc loaded successfully."

		for i, network in enumerate(self.head_fc_array):
			filename = dir + "/head_fc_%d.model" % i
			if os.path.isfile(filename):
				serializers.load_hdf5(filename, self.head_fc_array[i])
				print "model %s loaded successfully." % filename

		filename = dir + "/shared_fc.optimizer"
		if os.path.isfile(filename):
			serializers.load_hdf5(filename, self.optimizer_shared_fc)
			print "optimizer shared_fc loaded successfully."

		for i, network in enumerate(self.optimizer_head_fc_array):
			filename = dir + "/head_fc_%d.optimizer" % i
			if os.path.isfile(filename):
				serializers.load_hdf5(filename, self.optimizer_head_fc_array[i])
				print "optimizer %s loaded successfully." % filename

	def save(self):
		dir = "model"
		serializers.save_hdf5(dir + "/shared_fc.model", self.shared_fc)
		for i, network in enumerate(self.head_fc_array):
			filename = dir + "/head_fc_%d.model" % i
			serializers.save_hdf5(filename, self.head_fc_array[i])
		print "model saved."
		serializers.save_hdf5(dir + "/shared_fc.optimizer", self.optimizer_shared_fc)
		for i, network in enumerate(self.optimizer_head_fc_array):
			filename = dir + "/head_fc_%d.optimizer" % i
			serializers.save_hdf5(filename, self.optimizer_head_fc_array[i])
		print "optimizer saved."