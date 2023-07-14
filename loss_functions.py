from torch.autograd import Variable
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def BSL(labels, logits, sample_per_class):
	"""Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
	Args:
	  labels: A int tensor of size [batch].
	  logits: A float tensor of size [batch, no_of_classes].
	  sample_per_class: A int tensor of size [no of classes].
	  reduction: string. One of "none", "mean", "sum"
	Returns:
	  loss: A float tensor. Balanced Softmax Loss.
	"""
	for i in range(len(sample_per_class)):
		if sample_per_class[i] <= 0:
			sample_per_class[i] = 1
	spc = torch.tensor(sample_per_class).type_as(logits)
	spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
	logits = logits + spc.log()
	loss = F.cross_entropy(input=logits, target=labels)
	return loss

def RBL(labels, logits, sample_pre_class, at_pre_class):
	beta = np.zeros(len(sample_pre_class)).astype(np.float32)
	E = np.zeros(len(sample_pre_class)).astype(np.float32)
	for i in range(len(sample_pre_class)):
		beta[i] = (sample_pre_class[i] - 1.) / sample_pre_class[i]
		E[i] = (1. - beta[i]**at_pre_class[i]) / (1. - beta[i])
	weights = 1. / (E + 1e-5)
	weights = weights / np.sum(weights) * len(sample_pre_class)
	loss = F.cross_entropy(logits, labels, weight=torch.from_numpy(weights.astype(np.float32)).cuda())
	return loss

def REAT(model, x, y, optimizer, sample_per_class, at_per_class, args):
	kl = nn.KLDivLoss(size_average='none').cuda()
	spc = torch.tensor(sample_per_class).type_as(x)
	weights = torch.sqrt(1. / (spc / spc.sum()))
	tail_class = [i for i in range(len(sample_per_class)//3 * 2 + 1, len(sample_per_class))]
	model.eval()
	epsilon = args.eps
	num_steps = args.ns
	step_size = args.ss
	x_adv = x.detach() + torch.FloatTensor(*x.shape).uniform_(-epsilon, epsilon).cuda()
	for _ in range(num_steps):
		x_adv.requires_grad_()
		with torch.enable_grad():
			f_adv, logits_adv = model(x_adv, True)
			loss = RBL(y, logits_adv, sample_per_class, at_per_class)
		grad = torch.autograd.grad(loss, [x_adv])[0]
		x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
		x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
		x_adv = torch.clamp(x_adv, 0.0, 1.0)
	model.train()
	x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
	# zero gradient
	optimizer.zero_grad()
	# calculate robust loss
	f_adv, logits = model(x_adv, True)
	TAIL = 0.0
	counter = 0.0
	for bi in range(y.size(0)):
		if y[bi].item() in tail_class:
			idt = torch.tensor([-1. if y[bi].item()==y[bj].item() else 1. for bj in range(y.size(0))]).cuda()
			W = torch.tensor([weights[y[bi].item()] + weights[y[bj].item()] for bj in range(y.size(0))]).cuda()
			TAIL += kl(F.log_softmax(f_adv, 1), F.softmax(f_adv[bi, :].clone().detach().view(1, -1).tile(y.size(0), ).view(y.size(0), -1), 1)) * idt * W
			counter += 1
	TAIL = TAIL.mean() / counter if counter>0. else 0.0
	loss = BSL(y, logits, sample_per_class) + TAIL
	return logits, loss

