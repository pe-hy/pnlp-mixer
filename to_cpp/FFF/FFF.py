import time
import random
import numpy as np
import os
import torch
import torch.nn as nn
torch.set_num_threads(1)
from fff_extension import fff_l1

os.environ["MKL_NUM_THREADS"] = "1"

class FFFInference(nn.Module):
	def __init__(self, fff_layer):
		# keep parallel size 1.
		super().__init__()
		self.W1 = fff_layer.linear_in.weight
		self.W2 = fff_layer.linear_out.weight.T.contiguous() # the transpose is important!

		self.output_width = fff_layer.output_width
		self.input_width = fff_layer.input_width
		self.depth = 4
		self.n_nodes = 2 ** (self.depth + 1) - 1

	def forward(self, x):
		print("INFERENCE")
		print(x.shape)
		x = x.reshape(-1, self.input_width)
		batch_size = x.shape[0]

		OUT = torch.zeros(batch_size, self.output_width)

		fff_l1(x, self.W1, self.W2, OUT, batch_size, self.input_width, self.output_width, 4)
		print("OUT")
		print(OUT.shape)
	
		return OUT

class FFFTrain(nn.Module):
	def __init__(self, input_width, output_width, depth, activation=nn.GELU):
		super().__init__()

		self.input_width = input_width
		self.output_width = output_width
		self.depth = depth # this functions as the max depth
		self.parallel_size = 1 # parallel trees are not currently suppported.
		self.n_nodes = 2 ** (self.depth + 1) - 1
		self.bias = False
		self.switch_decisions_regularization = False
		if self.bias: # easier to do bias like this in c++
			self.linear_in = nn.Linear(input_width + 1, self.parallel_size * self.n_nodes, bias=False)

		else:
			self.linear_in = nn.Linear(input_width, self.parallel_size * self.n_nodes, bias=False)
		self.linear_out = nn.Linear(self.parallel_size * self.n_nodes, output_width, bias=False)

		self.activation = activation()

	def forward(self, oldx: torch.Tensor, force_depth, stoch=False) -> torch.Tensor:

		if not stoch:
			depth = force_depth
		else:
			depth = random.randint(0, force_depth)

		x = oldx.reshape(-1, self.input_width)

		if self.bias:
			biastensor = torch.ones(x.shape[0], 1)
			x = torch.cat((x, biastensor), dim=1)

		batch_size = x.shape[0]

		logits = self.linear_in(x) # (batch_size, parallel_size * n_nodes)

		logit_decisions = (logits > 0).long() # (batch_size, parallel_size * n_nodes)



		if self.switch_decisions_regularization:
			flips = torch.tensor(np.random.choice([0, 1], logit_decisions.shape, p=[0.9, 0.1]))

			logit_decisions = logit_decisions + flips

			logit_decisions[logit_decisions == 2] = 0


		activations = self.activation(logits) # (batch_size, parallel_size * n_nodes)

		activations = activations.view(batch_size, self.parallel_size, self.n_nodes) # (batch_size, parallel_size, n_nodes)

		decisions = logit_decisions.view(batch_size, self.parallel_size, self.n_nodes) # (batch_size, parallel_size, n_nodes)

		with torch.no_grad():
			current_nodes = torch.zeros((batch_size, self.parallel_size), dtype=torch.long, device=x.device)

			decision_map = torch.zeros_like(decisions, dtype=torch.float) # (batch_size, parallel_size, n_nodes)

			decision_map.scatter_(dim=2, index=current_nodes.unsqueeze(-1), value=1.0)

			for d in range(depth):

				current_platform = 2 ** d - 1

				next_platform = 2 ** (d + 1) - 1

				moves = torch.gather(decisions, 2, current_nodes.unsqueeze(2)).squeeze(2)

				next_nodes = (current_nodes - current_platform) * 2 + moves + next_platform

				decision_map.scatter_(2, next_nodes.unsqueeze(-1), 1.0)


				current_nodes = next_nodes

		activations = activations * decision_map # (batch_size, parallel_size, n_nodes)

		new_logits = self.linear_out(activations.flatten(1, 2)) # (batch_size, output_width)

		return new_logits


if __name__ == "__main__":

	batch_size = 16
	hidden_dim = 128
	depth = 6
	layer_size = 2
	n_nodes = (1 << depth) - 1;
	N = 250

	# =============================================== WORKING =========================================
	torch.torch.manual_seed(17)
	# simple_fff = FFF(hidden_dim, hidden_dim, depth, 1)

	input_width = 13
	output_width = 10
	simple_fff = FFFTrain(input_width, output_width, depth, 1)
	# print(fff_python_layer)

	W1 = simple_fff.linear_in.weight
	W2 = simple_fff.linear_out.weight
	print(W1.shape)
	print(W2.shape)
	IN = torch.randn(batch_size, input_width)
	OUT = torch.zeros(batch_size, output_width)

	# OUT_T = torch.zeros(batch_size, hidden_dim).T.contiguous()
	W2T = W2.T.contiguous()
	times = []
	for k in range(1):
		python_start = time.time()
		ret = simple_fff.forward(IN, depth, False)
		print("RET")
		print(ret)
		python_time = time.time() - python_start
		print("Python Time: ", python_time)
		# print(W1.shape)
		fff_start = time.time()

		fff_l1(IN, W1, W2T, OUT, batch_size, input_width, output_width, depth)
		fff_time = time.time() - fff_start
		print("FFF Time: ", fff_time)
		print(OUT)

		print("Speedup")
		print(python_time / float(fff_time))
		times.append(python_time / float(fff_time))

	print("Mean speedup: ",  float(sum(times))/ len(times))
	# =============================================== WORKING =========================================

	FFF_PT_MKL = FFFInference(simple_fff)
	print("MKLFFF")
	print(FFF_PT_MKL.forward(IN, depth))

