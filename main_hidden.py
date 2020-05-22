"""
    Adv train for free CIFAR10 with PyTorch.
    with loss on hidden layer
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from tqdm import tqdm
import random
import numpy as np
from models.wideresnet import *
from inference import adv_test
from models.iterative_projected_gradient import LinfPGDAttack


def train(epoch, net, trainloader, device, m, delta, optimizer, epsilon, args):
    print('Epoch: {}/{}'.format(epoch, args.epoch))
    net.train()
    # train_loss = 0
    correct = 0
    total = 0
    iterator = tqdm(trainloader, ncols=0, leave=False)
    kl_criterion = nn.KLDivLoss(reduction='batchmean')  # kl_criterion(h_adv.log(), h)
    mse = nn.MSELoss()
    cos = nn.CosineSimilarity()

    for batch_idx, (inputs, targets) in enumerate(iterator):
        inputs, targets = inputs.to(device), targets.to(device)
        for i in range(m):
            optimizer.zero_grad()
            x_adv = (inputs + delta).detach()
            x_adv.requires_grad_()
            h_adv = net(x_adv)
            adv_outputs = net.h_to_logits(h_adv)
            xent_loss = F.cross_entropy(adv_outputs, targets)
            h = net(inputs)
            h_loss = mse(h_adv, h)
            loss = h_loss * 1 + xent_loss
            loss.backward()
            optimizer.step()
            grad = x_adv.grad.data
            delta = delta.detach() + epsilon * torch.sign(grad.detach())
            delta = torch.clamp(delta, -epsilon, epsilon)

            _, adv_predicted = adv_outputs.max(1)
            total += targets.size(0)
            correct += adv_predicted.eq(targets).sum().item()
            desc = 'h_loss:' + "{:10.4f}".format(h_loss.item()) + ' xent_loss:' + "{:10.4f}".format(xent_loss.item())
            iterator.set_description(desc=desc)

    acc = 100. * correct / total
    print('Train acc:', acc)


def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--seed', default=9527, type=int)
    parser.add_argument('--epoch', default=41, type=int)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-3, type=float)
    parser.add_argument('--epsilon', default=8.0 / 255, type=float)
    parser.add_argument('--m', default=10, type=int)
    parser.add_argument('--iteration', default=20, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--step_size', default=2. / 255, type=float)
    parser.add_argument('--resume', '-r', default=None, type=int, help='resume from checkpoint')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # Normalization messes with l-inf bounds.
    ])

    trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])
    testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    net = wide_resnet_34_10()
    epsilon = args.epsilon
    m = args.m
    delta = torch.zeros(args.batch_size, 3, 32, 32)
    delta = delta.to(device)
    net = net.to(device)

    if args.resume is not None:
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/adv_hloss_ckpt.{}'.format(args.resume))
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch'] + 1

    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=args.momentum, weight_decay=args.weight_decay,
                          nesterov=True)

    adversary = LinfPGDAttack(
        net, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=args.epsilon,
        nb_iter=args.iteration, eps_iter=args.step_size, rand_init=True, clip_min=0.0, clip_max=1.0,
        targeted=False)

    for epoch in range(start_epoch, args.epoch):
        adjust_learning_rate(optimizer, epoch)
        train(epoch, net, trainloader, device, m, delta, optimizer, epsilon, args)

    adv_test(args.epoch, net, testloader, device, adversary)
    # if not os.path.isdir('checkpoint'):
    #     os.mkdir('checkpoint')
    state = {
        'net': net.state_dict(),
        'epoch': args.epoch
    }
    torch.save(state, './checkpoint/adv_hloss_ckpt.{}'.format(args.epoch))


if __name__ == '__main__':
    main()
