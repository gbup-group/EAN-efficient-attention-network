from __future__ import print_function

import os
import shutil
import time
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import imagenet_network as models
from torch.utils.data.sampler import SubsetRandomSampler
from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig
import numpy as np
from train_imagenet_ensemble_subset import train, test

import argparse
import tools
import math
import copy
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='NAS')
# Datasets
parser.add_argument('-data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=160, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--validate_size', type=int, default=0, help='')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--checkpoint', default='', type=str, help='checkpoint')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
print(state)


use_cuda = torch.cuda.is_available()
# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)

np.random.seed(args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

hyperparams = {}
hyperparams['controller_max_step'] = 300
hyperparams['ema_baseline_decay'] = 0.95
hyperparams['controller_lr'] = 5e-2
hyperparams['controller_grad_clip'] = 0
hyperparams['checkpoint'] = args.checkpoint

if args.arch == 'forward_config_share_sge_resnet50' or 'forward_dia_fbresnet50':
    config_limit = (3, 4, 6, 3)
    total_num_blocks = sum(config_limit)
else:
    num_blocks = None
    total_num_blocks = None

print(hyperparams)

np.random.seed(args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed_all(args.manualSeed)

## build the ckpt directory
if not os.path.isdir(args.checkpoint):
    mkdir_p(args.checkpoint)

## save the hyperparams
with open(args.checkpoint + "/Config.txt", 'w+') as f:
    for (k, v) in args._get_kwargs():
        f.write(k + ' : ' + str(v) + '\n')
    for v in hyperparams:
        f.write(v + ' : ' + str(hyperparams[v]) + '\n')

#replaybuffer for ppo
class ReplayBuffer(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.memory_actions = []
        self.memory_rewards = []
        self.memory_probs = []
        self.memory_steps = []

    def num_buffers(self):
        return len(self.memory_actions)

    def add_new(self, action, reward, prob, step):
        self.memory_actions.append(action)
        self.memory_rewards.append(reward)
        self.memory_probs.append(prob)
        self.memory_steps.append(step)
        if len(self.memory_rewards)>self.max_size:
            self.memory_actions.pop(0)
            self.memory_rewards.pop(0)
            self.memory_probs.pop(0)
            self.memory_steps.pop(0)

    def sample(self, num):
        rnd_choice = np.random.choice(np.arange(len(self.memory_steps)), size=num, replace=False) 
        sampled_actions = [self.memory_actions[i] for i in rnd_choice.tolist()]
        sampled_rewards = [self.memory_rewards[i] for i in rnd_choice.tolist()]
        sampled_probs = [self.memory_probs[i] for i in rnd_choice.tolist()]
        sampled_steps = [self.memory_steps[i] for i in rnd_choice.tolist()]
        return sampled_actions, sampled_rewards, sampled_probs, sampled_steps

#rnd
class RND_fix(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)

        self.num_blocks = total_num_blocks
        self.input_size = self.num_blocks
        self.hidden_size = 32
        self.output_scale = 4
        self.output_size = self.output_scale * self.num_blocks

        self.rnd_fix = nn.Sequential(
            nn.Linear(self.input_size,self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size,self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size,self.output_size)
        )

    def forward(self,x):
        logits = self.rnd_fix(x)
        return logits

class RND_learn(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)

        self.num_blocks = total_num_blocks
        self.input_size = self.num_blocks
        self.hidden_size = 32
        self.output_scale = 4
        self.output_size = self.output_scale * self.num_blocks

        self.rnd_fix = nn.Sequential(
            nn.Linear(self.input_size,self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size,self.output_size)
        )

    def forward(self,x):
        logits = self.rnd_fix(x)
        return logits

class Controller(torch.nn.Module):
    """Based on
    https://github.com/pytorch/examples/blob/master/word_language_model/model.py
    Base the controller RNN on the GRU from:
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
    """
    def __init__(self):
        torch.nn.Module.__init__(self)

        self.num_blocks = total_num_blocks
        self.softmax_temperature = 5.0
        self.tanh_c = 2.5
        self.mode = True


        self.input_size = 20
        self.hidden_size = 50
        self.output_size = self.num_blocks * 2

        self._fc_controller = nn.Sequential(
            nn.Linear(self.input_size,self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size,self.output_size)
        )


    def forward(self,x):
        logits = self._fc_controller(x)

        logits /= self.softmax_temperature

        # exploration # ??
        if self.mode == 'train':
            logits = (self.tanh_c*F.tanh(logits))

        return logits

    def sample(self, batch_size=1, replay = None):
        """Samples a set of `args.num_blocks` many computational nodes from the
        controller, where each node is made up of an activation function, and
        each node except the last also includes a previous node.
        """

        # [B, L, H]
        inputs = torch.zeros(batch_size, self.input_size).cuda()
        log_probs = []
        actions = []


        total_logits = self.forward(inputs)


        if replay == None:
            for block_idx in range(total_num_blocks):
                logits = total_logits[:,(2*block_idx):(2*block_idx+1+1)]
            
                # print(logits.size()) # batch size * 2
                probs = F.softmax(logits, dim=-1) # batch size * 2
                log_prob = F.log_softmax(logits, dim=-1)

                action = probs.multinomial(num_samples=1).data
                # print(action.size()) # batch size * 1
                selected_log_prob = log_prob.gather(
                    1, tools.get_variable(action, requires_grad=False))
                # print(selected_log_prob.size()) # batch size * 1

                log_probs.append(selected_log_prob[:, 0:1])
                inputs = tools.get_variable(action[:, 0], requires_grad=False)
                actions.append(action[:, 0])


            return actions, torch.cat(log_probs, 1)



        else:

            r_actions, r_rewards, r_probs, r_steps = replay
            replay_actions, replay_rewards, replay_probs, replay_steps = copy.deepcopy(r_actions[0]),\
                                                                            copy.deepcopy(r_rewards[0]), \
                                                                                copy.deepcopy(r_probs[0]), \
                                                                                    copy.deepcopy(r_steps[0])
            ratio = []
            log_probs = []
            for block_idx in range(total_num_blocks):
                logits = total_logits[:,(2*block_idx):(2*block_idx+1+1)]

                probs = F.softmax(logits, dim=-1)
                log_prob = F.log_softmax(logits, dim=-1)

                action = probs.multinomial(num_samples=1).data

                val_size = args.validate_size + 1


                selected_log_prob = log_prob.gather(1, tools.get_variable(replay_actions[:,block_idx].view(1,val_size), requires_grad=False))
                temp_prob = replay_probs[:,block_idx].view(1,val_size)
                prob_ratio = torch.exp(selected_log_prob)/temp_prob
                

                ratio.append(prob_ratio)
                log_probs.append(selected_log_prob)
                inputs = tools.get_variable(action[:, 0], requires_grad=False)
            return torch.cat(log_probs, 0), replay_rewards, torch.cat(ratio,0)
            # torch.cat(log_probs, 0)  54*batch
            # torch.cat(ratio,0) 54*batch


    def init_hidden(self, batch_size):
        zeros = torch.zeros(batch_size, self.controller_hid)
        return (tools.get_variable(zeros, True, requires_grad=False),
                tools.get_variable(zeros.clone(), True, requires_grad=False))
# reward
def get_reward(trainloader, testloader, model, optimizer, criterion, actions, step, hyperparams):
    """Computes the perplexity of a single sampled model on a minibatch of
    validation data.
    """
    # get the action code
    rewards_list, val_acc_list, val_loss_list, sparse_portion_list = [], [], [], []
    for i in range(actions[0].size(0)):
        binary_code = ''
        for action in actions:
            binary_code = binary_code + str(action[i].item())
        seg = np.cumsum(config_limit)
        code1 = int(binary_code[0:seg[0]],2)
        code2 = int(binary_code[seg[0]:seg[1]],2)
        code3 = int(binary_code[seg[1]:seg[2]],2)
        code4 = int(binary_code[seg[2]:seg[3]],2)
        config = (code1, code2, code3, code4)

        # update the model
        if trainloader is not None:
            train_loss, train_acc = train(config, trainloader, model, criterion, optimizer, step)
        val_loss, val_acc = test(config, testloader, model, criterion)
        sparse_portion = sum([i == '0' for i in binary_code])/len(binary_code)

        base = 54.0
        R = val_acc + sparse_portion*2 - base
        rewards = R
        rewards_list.append(rewards)
        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss)
        sparse_portion_list.append(sparse_portion)
    return np.row_stack(rewards_list), np.row_stack(val_acc_list), np.row_stack(val_loss_list), np.row_stack(sparse_portion_list)

def get_action_code(actions):
        # get the action code (binary to decimal)
    binary_code = ''
    for action in actions:
        binary_code = binary_code + str(action.item())
    actions_code = int(binary_code, 2)
    return actions_code, binary_code

def train_controller(Controller, Controller_optim, rnd_fix, rnd_learn, rnd_learn_optim, trainloader, testloader, model, optimizer, criterion, hyperparams):
    """
    The controller is updated with a score function gradient estimator
    (i.e., REINFORCE), with the reward being c/valid_ppl, where valid_ppl
    is computed on a minibatch of validation data.

    A moving average baseline is used.

    The controller is trained for 2000 steps per epoch.
    """
    logger = Logger(os.path.join(hyperparams['checkpoint'], 'search_log.txt'), title='')
    logger.set_names(['Loss', 'Baseline', 'Reward', 'Action', 'Binary', 'Valid Loss', 'Valid Acc.', 'Sparse','rnd_loss','p'])

    logger_all = Logger(os.path.join(hyperparams['checkpoint'], 'search_log_all_sampled.txt'), title='')
    logger_all.set_names(['Loss', 'Baseline', 'Reward', 'Action', 'Binary', 'Valid Loss', 'Valid Acc.', 'Sparse'])
    
    model_fc = Controller
    model_fc.train()

    model_rnd_fix = rnd_fix
    model_rnd_fix.eval()
    model_rnd_learn = rnd_learn
    model_rnd_learn.train()

    baseline = None
    total_loss = 0
    buffer = ReplayBuffer(30)
    buffer_sparse = ReplayBuffer(30)
    update_mode = 'online'

    for step in range(hyperparams['controller_max_step']):
        print('************************* ('+str(step+1)+'/'+str(hyperparams['controller_max_step'])+')******')
        adjust_learning_rate(optimizer, step, args, hyperparams)
        actions, log_probs = model_fc.sample(replay = None)

        #sample N connection for val (for updating theta)
        actions_validate, log_probs_validate = model_fc.sample(batch_size=args.validate_size)

        decimal_code_all = []
        binary_code_all = []

        # get the action code (binary to decimal)
        actions_code_1, binary_code_1 = get_action_code(actions)
        decimal_code_all.append(actions_code_1)
        binary_code_all.append(binary_code_1)

        for i in range(actions_validate[0].size(0)):
            binary_code = ''
            for action in actions_validate:
                binary_code = binary_code + str(action[i].item())
            decimal_code = int(binary_code, 2)
            decimal_code_all.append(decimal_code)
            binary_code_all.append(binary_code)

        #get reward (train one "step")
        rewards_org, val_acc, val_loss, sparse_portion = get_reward(None, testloader, model, optimizer, criterion, actions, step, hyperparams)
        rewards_validate, val_acc_validate, val_loss_validate, sparse_portion_validate = get_reward(None, testloader, model, optimizer, criterion, actions_validate, step, hyperparams)

        val_acc = np.row_stack((val_acc, val_acc_validate))
        val_loss = np.row_stack((val_loss, val_loss_validate))
        #buf_rewards
        rewards = np.row_stack((rewards_org,rewards_validate)) 
        
        # #buf_action
        temp_action = torch.cat(actions).view(1,-1)
        temp_actions_validate = torch.cat(actions_validate).view(args.validate_size,-1)
        buf_action = torch.cat((temp_action,temp_actions_validate),0)

        # #buf_prob
        temp_prob = torch.exp(log_probs).detach()
        temp_prob_val = torch.exp(log_probs_validate).detach()
        buf_prob = torch.cat((temp_prob,temp_prob_val),0)


        # #store - buffer
        buffer.add_new(buf_action, rewards, buf_prob, step)

        #cal rnd
        val_fix = model_rnd_fix(buf_action.float())
        val_learn = model_rnd_learn(buf_action.float())
        reward_rnd = ((val_fix - val_learn)**2).sum()
        rewards = rewards + reward_rnd.detach().cpu().data.numpy()
        # moving average baseline
        if baseline is None:
            baseline = rewards.mean()
        else:
            decay = hyperparams['ema_baseline_decay']
            baseline = decay * baseline + (1 - decay) * rewards.mean()


        
        adv = rewards - baseline
        adv = adv 
        log_probs = torch.cat((log_probs, log_probs_validate), 0)
        loss = -log_probs * tools.get_variable(adv,True,requires_grad=False) 

        loss = loss.mean(dim = 0, keepdim = True)
        
        
        loss = loss.sum()

        # update
        Controller_optim.zero_grad()
        rnd_learn_optim.zero_grad()
        loss.backward()
        reward_rnd.mean().backward()
        if hyperparams['controller_grad_clip'] > 0:
            torch.nn.utils.clip_grad_norm(model_fc.parameters(),
                                          hyperparams['controller_grad_clip'])
        rnd_learn_optim.step()
        Controller_optim.step()



        log = 'Step: {step}| Loss: {loss:.4f}| Action: {act} |Baseline: {base:.4f}| ' \
              'Reward {re:.4f}| Valid Acc {acc:.4f}'.format(loss=loss.item(), base=baseline, act = binary_code,
                                                            re=rewards[0].item(), acc=val_acc[0].item(), step=step)
        print(log)
        logger.append([loss.item(), baseline, rewards[0].item(), actions_code_1, binary_code_1, \
                            val_loss[0].item(), val_acc[0].item(), (binary_code.count('0')/len(binary_code)), reward_rnd.item(), torch.exp(log_probs).mean().item()])
        for i in range(len(binary_code_all)):
            logger_all.append([loss.item(), baseline, rewards[i].item(), decimal_code_all[i], binary_code_all[i],
                           val_loss[i].item(), val_acc[i].item(), (binary_code_all[i].count('0') / len(binary_code_all[i]))])

        save_checkpoint({
                'iters': step + 1,
                'state_dict': model_fc.state_dict(),
                'optimizer' : Controller_optim.state_dict(),
                        }, checkpoint=hyperparams['checkpoint'])
        save_checkpoint({'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict()},
                        checkpoint=hyperparams['checkpoint'],
                        filename='model.pth.tar')


        if step >= 10:

            # use old data (buff)
            replay = buffer.sample(num = 1)
            log_probs, replay_rewards, prob_ratio = model_fc.sample(replay = replay)
            adv = replay_rewards - baseline
            eps_clip = 0.1

            temp_surr1 = (prob_ratio.detach() * log_probs).mm(tools.get_variable(adv,True,requires_grad=False))
            temp_surr2 = (torch.clamp(prob_ratio.detach(), 1-eps_clip, 1+eps_clip) * log_probs).mm(tools.get_variable(adv,True,requires_grad=False))
            surr1 = temp_surr1.sum()/(args.validate_size + 1)
            surr2 = temp_surr2.sum()/(args.validate_size + 1)

            loss = 0.1 * -torch.min(surr1, surr2) 


            # update
            Controller_optim.zero_grad()
            loss.backward()

            if hyperparams['controller_grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm(model_fc.parameters(),
                                            hyperparams['controller_grad_clip'])
            Controller_optim.step()






def save_checkpoint(state, checkpoint='checkpoint', filename='controller.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

def adjust_learning_rate(optimizer, epoch, args, hyperparams):
    lr = 0.5 * args.lr * (math.cos(math.pi * epoch / hyperparams['controller_max_step']) + 1)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
def main():


    # Data loading code
    traindir = os.path.join(args.data, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset_from_train = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    print('train_datatset_len', len(train_dataset))

    train_subset = []
    val_subset = []
    # for class_idx in range(1, 1001):
    for class_idx in range(1000):
        class_sample_size = len([e for e, i in enumerate(train_dataset.targets) if i == class_idx])
        class_sample_index = [e for e, i in enumerate(train_dataset.targets) if i == class_idx]
        split = 100
        val_subset.append(class_sample_index[:split])
        train_subset.append(class_sample_index[split:])

    flattened_val_subset = [val for sublist in val_subset for val in sublist]
    flattened_train_subset = [val for sublist in train_subset for val in sublist]

    print(len(flattened_train_subset))
    print(len(flattened_val_subset))
    print('sum:', len(flattened_val_subset) + len(flattened_train_subset))

    train_sampler = SubsetRandomSampler(flattened_train_subset)
    valid_sampler = SubsetRandomSampler(flattened_val_subset)


    valid_from_train_loader = torch.utils.data.DataLoader(
        val_dataset_from_train, batch_size=args.batch_size, shuffle=(valid_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=valid_sampler)

    model = models.__dict__[args.arch](pretrained=False)
    model = torch.nn.DataParallel(model).cuda()

    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['state_dict'])

    controller = Controller().cuda()
    rnd_fix = RND_fix().cuda()
    rnd_learn = RND_learn().cuda()

    rnd_learn_optim = torch.optim.Adam(rnd_learn.parameters(), lr=hyperparams['controller_lr'])
    controller_optim = torch.optim.Adam(controller.parameters(), lr=hyperparams['controller_lr'])
    train_controller(controller, controller_optim, rnd_fix, rnd_learn, rnd_learn_optim,  None, valid_from_train_loader, model, optimizer, criterion, hyperparams)

if __name__ == '__main__':
    main()
