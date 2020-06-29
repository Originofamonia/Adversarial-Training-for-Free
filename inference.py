"""Train CIFAR10 with PyTorch."""
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
import numpy as np
from models.wideresnet import *
from models.utils import *
from models.iterative_projected_gradient import LinfPGDAttack


def adv_test(epoch, net, testloader, device, adversary):
    net.eval()
    adv_correct = 0
    clean_correct = 0
    total = 0
    with torch.no_grad():
        iterator = tqdm(testloader, ncols=0, leave=False)
        for batch_idx, (inputs, targets) in enumerate(iterator):
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.enable_grad():
                adv = adversary.perturb(inputs, targets)
            h_adv = net(adv)
            adv_outputs = net.h_to_logits(h_adv)
            _, adv_predicted = adv_outputs.max(1)
            h = net(inputs)
            outputs = net.h_to_logits(h)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            adv_correct += adv_predicted.eq(targets).sum().item()
            clean_correct += predicted.eq(targets).sum().item()
            iterator.set_description(str(adv_predicted.eq(targets).sum().item() / targets.size(0)))

    adv_acc = 100. * adv_correct / total
    clean_acc = 100. * clean_correct / total
    print('Adv test Acc of ckpt.{}: {}'.format(epoch, adv_acc))
    print('Clean test Acc of ckpt.{}: {}'.format(epoch, clean_acc))


def clean_test(epoch, net, testloader, device):
    net.eval()
    clean_correct = 0
    total = 0
    with torch.no_grad():
        iterator = tqdm(testloader, ncols=0, leave=False)
        for batch_idx, (inputs, targets) in enumerate(iterator):
            inputs, targets = inputs.to(device), targets.to(device)
            h = net(inputs)
            outputs = net.h_to_logits(h)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            clean_correct += predicted.eq(targets).sum().item()
            iterator.set_description(str(predicted.eq(targets).sum().item() / targets.size(0)))

    clean_acc = 100. * clean_correct / total
    print('Clean test Acc of ckpt.{}: {}'.format(epoch, clean_acc))


def kl_div_loss(logits_q, logits_p, T):
    assert logits_p.size() == logits_q.size()
    b, c = logits_p.size()
    p = nn.Softmax(dim=1)(logits_p / T)
    q = nn.Softmax(dim=1)(logits_q / T)
    epsilon = 1e-8
    _p = (p + epsilon * torch.ones(b, c).cuda()) / (1.0 + c * epsilon)
    _q = (q + epsilon * torch.ones(b, c).cuda()) / (1.0 + c * epsilon)
    return (T ** 2) * torch.mean(torch.sum(_p * torch.log(_p / _q), dim=1))


def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Testing')
    parser.add_argument('--seed', default=11111, type=int, help='seed')
    parser.add_argument('--epoch', default=30, type=int, help='load checkpoint from that epoch')
    parser.add_argument('--model', default='wideresnet', type=str)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--iteration', default=20, type=int)
    parser.add_argument('--epsilon', default=8. / 255, type=float)
    parser.add_argument('--step_size', default=2. / 255, type=float)

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    testset = torchvision.datasets.CIFAR10(root='data', train=True, download=True,
                                           transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    if args.model == 'wideresnet':
        net = wide_resnet_34_10()
    else:
        raise ValueError('No such model.')

    checkpoint = torch.load('./checkpoint/adv_ckpt_41.pt')
    net.load_state_dict(checkpoint['net'])
    net = net.to(device)
    net.eval()

    adversary = LinfPGDAttack(
        net, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=args.epsilon,
        nb_iter=args.iteration, eps_iter=args.step_size, rand_init=True, clip_min=0.0, clip_max=1.0,
        targeted=False)

    adv_test(args.epoch, net, testloader, device, adversary)


if __name__ == '__main__':
    main()
