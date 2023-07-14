import copy
import torch
import torch.utils.data
import resnet
from torch import nn
import torchattacks
from tqdm import tqdm
import logging
import random
import os
import numpy as np
import argparse
import wrn
import loss_functions
from torch.optim.lr_scheduler import MultiStepLR
import time
from datetime import timedelta
from logging import getLogger
from torch.nn import functional as F
import build_loader, builder

class LogFormatter:
	def __init__(self):
		self.start_time = time.time()

	def format(self, record):
		elapsed_seconds = round(record.created - self.start_time)

		prefix = "%s - %s - %s" % (
			record.levelname,
			time.strftime("%x %X"),
			timedelta(seconds=elapsed_seconds),
		)
		message = record.getMessage()
		message = message.replace("\n", "\n" + " " * (len(prefix) + 3))
		return "%s - %s" % (prefix, message) if message else ""
def create_logger(filepath, rank):
	# create log formatter
	log_formatter = LogFormatter()

	# create file handler and set level to debug
	if filepath is not None:
		if rank > 0:
			filepath = "%s-%i" % (filepath, rank)
		file_handler = logging.FileHandler(filepath, "a")
		file_handler.setLevel(logging.DEBUG)
		file_handler.setFormatter(log_formatter)

	# create console handler and set level to info
	console_handler = logging.StreamHandler()
	console_handler.setLevel(logging.INFO)
	console_handler.setFormatter(log_formatter)

	# create logger and set level to debug
	logger = logging.getLogger()
	logger.handlers = []
	logger.setLevel(logging.DEBUG)
	logger.propagate = False
	if filepath is not None:
		logger.addHandler(file_handler)
	logger.addHandler(console_handler)

	# reset logger elapsed time
	def reset_time():
		log_formatter.start_time = time.time()

	logger.reset_time = reset_time

	return logger
def setup_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True
	np.random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet',
					help='model architecture')

##Dataset Settings
parser.add_argument('--dataset', default='cifar10', type=str,
					help='which dataset used to train')
parser.add_argument('--imb', type=float, default=0.02,
					help='imbalance ratio for dataset')
parser.add_argument('--ext', type=float, default=1.0,
					help='existing ratio for dataset')
parser.add_argument('--num_classes', default=10, type=int, metavar='N',
					help='number of classes')

##Training Settings
parser.add_argument('--epochs', default=80, type=int, metavar='N',
					help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=128, type=int,
					metavar='N',
					help='mini-batch size (default: 256), this is the total '
						 'batch size of all GPUs on the current node when '
						 'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
					metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
					help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
					metavar='W', help='weight decay (default: 1e-4)',
					dest='wd')
parser.add_argument('--gamma', type=float, default=0.1,
					help='LR is multiplied by gamma on schedule.')
parser.add_argument('--seed', type=int,
					default=0, help='random seed')


##AT Settings
parser.add_argument('--eps', type=float, default=8./255., help='perturbation bound')
parser.add_argument('--ns', type=int, default=10, help='maximum perturbation step K')
parser.add_argument('--ss', type=float, default=2./255., help='step size')
parser.add_argument('--beta', type=float, default=6.0)


##Save Settings
parser.add_argument('--save', default='M2.pkl', type=str,
					help='model save name')
parser.add_argument('--exp', default='exp_test', type=str,
					help='exp name')

args = parser.parse_args()

if args.dataset == 'cifar10':
	args.num_classes = 10
else:
	args.num_classes = 100


os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
setup_seed(args.seed)


logger = getLogger()
if not os.path.exists(args.dataset+'/' + str(args.imb)+'/'+ args.arch +'/'+args.exp):
	os.makedirs(args.dataset+'/'+ str(args.imb)+'/'+ args.arch +'/'+args.exp)
logger = create_logger(
	os.path.join(args.dataset+'/'+ str(args.imb)+'/'+ args.arch +'/'+args.exp + '/', args.exp + ".log"), rank=0
)
logger.info("============ Initialized logger ============")
logger.info(
	"\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
)
args.save = args.dataset+'/'+ str(args.imb)+'/'+ args.arch +'/'+args.exp + '/' +  args.save



torch.backends.cudnn.benchmark = True

args.num_classes= 10 if args.dataset == 'cifar10' else 100
trainset, samples_per_cls = builder.build_datasets(name=args.dataset, mode='train',
										   num_classes=args.num_classes,
										   imbalance_ratio=0.0 if args.ext < 1.0 else args.imb,
										   existing_ratio=args.ext,
										   root='./data')
testset, _ = builder.build_datasets(name=args.dataset, mode='test',
							num_classes=args.num_classes, root='./data')

train_loader = build_loader.build_dataloader(trainset, imgs_per_gpu=args.batch_size,
											 dist=False, sampler=None, shuffle=True)
test_loader = build_loader.build_dataloader(testset, imgs_per_gpu=args.batch_size,
											dist=False, shuffle=False)

if args.arch == 'resnet':
	n = resnet.resnet18(args.dataset).cuda()
elif args.arch == 'wrn':
	n = wrn.WideResNet(num_classes=args.num_classes).cuda()

optimizer = torch.optim.SGD(n.parameters(),momentum=args.momentum,
							lr=args.lr,weight_decay=args.wd)

milestones = [60, 75]
scheduler = MultiStepLR(optimizer,milestones=milestones,gamma=args.gamma)


train_clean_acc = []
train_adv_acc = []
test_clean_acc = []
test_adv_acc = []

train_clean_loss = []
train_adv_loss = []
test_clean_loss = []
test_adv_loss = []

best_eval_acc = 0.0

overall = []
overall_error = []
pre_at_sample = copy.deepcopy(samples_per_cls)
next_at_sample = [0 for i in range(len(samples_per_cls))]
overall.append(samples_per_cls)
overall_error.append(samples_per_cls)
piror_p = [0 for i in range(len(samples_per_cls))]
post_p = copy.deepcopy(samples_per_cls)
print(pre_at_sample)

for epoch in range(args.epochs):
	loadertrain = tqdm(train_loader, desc='{} E{:03d}'.format('train', epoch), ncols=0)
	epoch_loss = 0.0
	epoch_loss_clean = 0.0
	total = 0.0
	clean_acc = 0.0
	adv_acc = 0.0
	for (input, target, index) in loadertrain:
		n.eval()
		x_train, y_train = input.cuda(), target.cuda()
		y_pre = n(x_train)
		loss_clean = F.cross_entropy(y_pre, y_train)
		epoch_loss_clean += loss_clean.data.item()
		logits_adv, loss = loss_functions.REAT(n, x_train, y_train, optimizer,
													  samples_per_cls, pre_at_sample, args)
		loss.backward()
		optimizer.step()
		epoch_loss += loss.data.item()
		_, predicted = torch.max(y_pre.data, 1)
		_, predictedadv = torch.max(logits_adv.data, 1)
		if (epoch + 1) % 1 == 0:
			for j in range(predictedadv.size(0)):
				next_at_sample[predictedadv[j].item()] += 1
				if predictedadv[j].item() != y_train[j].item():
					piror_p[y_train[j].item()] += 1
		total += y_train.size(0)
		clean_acc += predicted.eq(y_train.data).cuda().sum()
		adv_acc += predictedadv.eq(y_train.data).cuda().sum()
		fmt = '{:.4f}'.format
		loadertrain.set_postfix(loss=fmt(epoch_loss / total * args.batch_size),
								acc_cl=fmt(clean_acc.item() / total * 100),
								acc_adv=fmt(adv_acc.item() / total * 100))

	if (epoch + 1) % 1 == 0:
		pre_at_sample = copy.deepcopy(next_at_sample)
		next_at_sample = [0 for i in range(len(samples_per_cls))]
		post_p = copy.deepcopy(piror_p)
		piror_p = [0 for i in range(len(samples_per_cls))]
	overall.append(copy.deepcopy(pre_at_sample))
	overall_error.append(copy.deepcopy(post_p))
	print(pre_at_sample)
	print(post_p)
	train_clean_acc.append(clean_acc.item() / total * 100)
	train_adv_acc.append(adv_acc.item() / total * 100)
	train_clean_loss.append(epoch_loss_clean / total * args.batch_size)
	train_adv_loss.append(epoch_loss / total * args.batch_size)
	scheduler.step()
	if (epoch) % 1 == 0:
		Loss_test = nn.CrossEntropyLoss().cuda()
		test_loss_cl = 0.0
		test_loss_adv = 0.0
		correct_cl = 0.0
		correct_adv = 0.0
		total = 0.0
		n.eval()
		pgd_eval = torchattacks.PGD(n, eps=8.0/255.0, steps=20, alpha=2./255.)
		loadertest = tqdm(test_loader, desc='{} E{:03d}'.format('test', epoch), ncols=0)
		with torch.enable_grad():
			for (x_test, y_test) in loadertest:
				x_test, y_test = x_test.cuda(), y_test.cuda()
				x_adv = pgd_eval(x_test, y_test)
				n.eval()
				y_pre = n(x_test)
				y_adv = n(x_adv)
				loss_cl = Loss_test(y_pre, y_test)
				loss_adv = Loss_test(y_adv, y_test)
				test_loss_cl += loss_cl.data.item()
				test_loss_adv += loss_adv.data.item()
				_, predicted = torch.max(y_pre.data, 1)
				_, predicted_adv = torch.max(y_adv.data, 1)
				total += y_test.size(0)
				correct_cl += predicted.eq(y_test.data).cuda().sum()
				correct_adv += predicted_adv.eq(y_test.data).cuda().sum()
				fmt = '{:.4f}'.format
				loadertest.set_postfix(loss_cl=fmt(test_loss_cl / total * args.batch_size),
									   loss_adv=fmt(test_loss_cl / total * args.batch_size),
									   acc_cl=fmt(correct_cl.item() / total * 100),
									   acc_adv=fmt(correct_adv.item() / total * 100))
			test_clean_acc.append(correct_cl.item() / total * 100)
			test_adv_acc.append(correct_adv.item() / total * 100)
			test_clean_loss.append(test_loss_cl / total * args.batch_size)
			test_adv_loss.append(test_loss_adv / total * args.batch_size)
		if correct_adv.item() / total * 100 >= best_eval_acc:
			best_eval_acc = correct_adv.item() / total * 100
			checkpoint = {
					'state_dict': n.state_dict(),
					'epoch': epoch
				}
			torch.save(checkpoint, args.save+ 'best.pkl')

checkpoint = {
			'state_dict': n.state_dict(),
			'epoch': epoch
		}
torch.save(checkpoint, args.save + 'last.pkl')
np.save(args.save+'_train_acc_cl.npy', train_clean_acc)
np.save(args.save+'_train_acc_adv.npy', train_adv_acc)
np.save(args.save+'_test_acc_cl.npy', test_clean_acc)
np.save(args.save+'_test_acc_adv.npy', test_adv_acc)

np.save(args.save+'_train_loss_cl.npy', train_clean_loss)
np.save(args.save+'_train_loss_adv.npy', train_adv_loss)
np.save(args.save+'_test_loss_cl.npy', test_clean_loss)
np.save(args.save+'_test_loss_adv.npy', test_adv_loss)

overall = np.array(overall).reshape((-1, args.num_classes))
np.save(args.save+'label_d.npy', overall)

overall_error = np.array(overall_error).reshape((-1, args.num_classes))
np.save(args.save+'error_d.npy', overall_error)
