"""
Use MINE to estimate the MI(x, y) and MI(h, y)
mine_abdu
https://github.com/abdulfatir/MINE-PyTorch/blob/master/Neural-MI-Estimation.ipynb
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

from models.resnet import ResNet34, ResNet2
from models.wideresnet import wide_resnet_34_10, adjust_learning_rate
from mutual_info.mine_sung import draw, get_cifar10, ma
from mutual_info.mine_abdu import train_mine, train_mine_f_div, train_mine_density_ratio


def calc_mi():
    parser = argparse.ArgumentParser(description='MINE robust model')
    parser.add_argument('--seed', default=9527, type=int)
    parser.add_argument('--epochs', default=11, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--h', default=True, type=bool)  # whether I(x, y): False or I(h, y): True
    parser.add_argument('--batch_size', default=5, type=int)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    robust_model = 'checkpoint/adv_ckpt_41.pt'
    robust_net = wide_resnet_34_10()
    checkpoint = torch.load(robust_model)
    robust_net.load_state_dict(checkpoint['net'])
    robust_net.to(device)

    if args.h:
        in_channels = 640  # 3 for MI(x, y), 640 for MI(h, y)
    else:
        in_channels = 3

    mine_net = ResNet2(in_channels, args.h).cuda()
    # mine_net = ResNet34(in_channels, args.h).cuda()
    trainloader, testloader = get_cifar10(args.batch_size)
    optimizer = optim.Adam(mine_net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    mi_lb_list = train_mine_f_div(trainloader, testloader, robust_net, mine_net, optimizer, device, args.epochs,
                                  args.h)
    result_cor_ma = ma(mi_lb_list)
    print('h: {}, last MI: {}'.format(args.h, result_cor_ma[-1]))
    if args.h:
        with open('mi_hy_sung.txt', 'w') as filehandle:
            for listitem in result_cor_ma:
                filehandle.write('%s\n' % listitem)
    else:
        with open('mi_xy_sung.txt', 'w') as filehandle:
            for listitem in result_cor_ma:
                filehandle.write('%s\n' % listitem)


if __name__ == '__main__':
    calc_mi()
    # draw()
