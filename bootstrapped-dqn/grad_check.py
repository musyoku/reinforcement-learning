# -*- coding: utf-8 -*-
import sys, os
import numpy as np
from chainer import cuda, optimizers, gradient_check, Variable
from chainer import links as L
sys.path.append(os.path.split(os.getcwd())[0])
import model
from config import config

# Override config
config.use_gpu = True
config.q_k_heads = 3


def forward_check():
	xp = cuda.cupy if config.use_gpu else np
	out_head = 2
	in_head = 3
	n_x = 100
	state = xp.ones((2, n_x)).astype(xp.float32)
	state = Variable(state)
	initial_weight = np.ones((config.q_k_heads * in_head, n_x))
	shared = L.Linear(n_x, config.q_k_heads * in_head, initialW=initial_weight)
	initial_weight = np.ones((out_head * config.q_k_heads, in_head * config.q_k_heads))
	link1 = model.LinearHead(in_head, out_head, config.q_k_heads, initialW=initial_weight)
	initial_weight = np.ones((in_head * config.q_k_heads, out_head * config.q_k_heads))
	link2 = model.LinearHead(out_head, in_head, config.q_k_heads, initialW=initial_weight)
	if config.use_gpu:
		link1.to_gpu()
		link2.to_gpu()
		shared.to_gpu()
	output = link2(link1(shared(state)))
	print output.data


def grad_check():
	xp = cuda.cupy if config.use_gpu else np
	out_head = 2
	in_head = 3
	state = xp.ones((2, in_head * config.q_k_heads)).astype(xp.float32)
	state = Variable(state)
	initial_weight = np.ones((out_head * config.q_k_heads, in_head * config.q_k_heads))
	link = model.LinearHead(in_head, out_head, config.q_k_heads, initialW=initial_weight)
	if config.use_gpu:
		link.to_gpu()
	y_grad = xp.random.uniform(-1.0, 1.0, (2, 2 * config.q_k_heads)).astype(xp.float32)
	gradient_check.check_backward(link, (state.data,), y_grad, eps=1e-2)

# backprop_check()
forward_check()
# grad_check()
	
