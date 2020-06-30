"""
https://github.com/abdulfatir/MINE-PyTorch/blob/master/Neural-MI-Estimation.ipynb
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def sample_batch(data, device, robust_net, h):
    x_sample, y_sample = data
    x_sample = x_sample.to(device)
    y_sample = y_sample.type(torch.float).to(device)
    y_sample = (y_sample - 0.45) / 9  # normalize y
    if h:
        x_sample = robust_net(x_sample)
        dims = 1, 640, 8, 8
    else:
        dims = 1, 3, 32, 32

    index = torch.randperm(y_sample.shape[0])
    y_shuffle = y_sample[index]
    y_sample = y_sample.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    y_sample = y_sample.repeat(dims).type(torch.float)

    y_shuffle = y_shuffle.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    y_shuffle = y_shuffle.repeat(dims).type(torch.float)

    return x_sample, y_sample, y_shuffle


def train_mine(trainloader, testloader, robust_net, mine_net, mine_net_optim, device, epochs, h):
    #
    iterator = tqdm(trainloader, ncols=0, leave=False)
    mi_mine = list()

    for i in range(1, epochs + 1):
        mine_net.train()

        for j, data in enumerate(iterator):
            x_sample, y_sample, y_shuffle = sample_batch(data, device, robust_net, h)
            pred_xy = mine_net(x_sample, y_sample)
            pred_x_y = mine_net(x_sample, y_shuffle)
            mi = torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y)))
            loss = -mi
            mine_net.zero_grad()
            loss.backward()
            mine_net_optim.step()
            mi_mine.append(mi.detach().cpu().numpy())
            desc = 'MI: ' + "{:10.4f}".format(mi.item())
            iterator.set_description(desc)

        print('Epoch: {}; MI: {}'.format(i, mi_mine[-1]))

    return mi_mine


def train_mine_f_div(trainloader, testloader, robust_net, mine_net, mine_net_optim, device, epochs, h):
    #
    iterator = tqdm(trainloader, ncols=0, leave=False)
    mi_f = list()

    for i in range(1, epochs + 1):
        mine_net.train()

        for j, data in enumerate(iterator):
            x_sample, y_sample, y_shuffle = sample_batch(data, device, robust_net, h)
            pred_xy = mine_net(x_sample, y_sample)
            pred_x_y = mine_net(x_sample, y_shuffle)
            mi_lb = torch.mean(pred_xy) - torch.mean(torch.exp(pred_x_y))
            loss = -mi_lb
            mine_net.zero_grad()
            loss.backward()
            mine_net_optim.step()
            mi_f.append(mi_lb.detach().cpu().numpy())
            desc = 'MI_f: ' + "{:10.4f}".format(mi_lb.item())
            iterator.set_description(desc)

        print('Epoch: {}; MI_f: {}'.format(i, mi_f[-1]))

    return mi_f


def train_mine_density_ratio(trainloader, testloader, robust_net, mine_net, mine_net_optim, device, epochs, h):
    #
    iterator = tqdm(trainloader, ncols=0, leave=False)
    crit = torch.nn.BCEWithLogitsLoss()
    mi_dr = list()

    for i in range(1, epochs + 1):
        mine_net.train()

        for j, data in enumerate(iterator):
            x_sample, y_sample, y_shuffle = sample_batch(data, device, robust_net, h)
            p_xy = torch.cat([x_sample, y_sample], dim=1)
            p_x_y = torch.cat([x_sample, y_shuffle], dim=1)
            labels = torch.cat([torch.ones(p_xy.size(0)), torch.zeros(p_x_y.size(0))]).to(device)

            data_x = torch.cat([p_xy, p_x_y])
            logits = mine_net(data_x)
            loss = crit(logits, labels.view(-1, 1))
            z = mine_net(p_xy)
            mine_net.zero_grad()
            loss.backward()
            mine_net_optim.step()

            mi = (torch.sigmoid(z)/(1 - torch.sigmoid(z))).log().mean()
            desc = 'MI_dr: ' + "{:10.4f}".format(mi.item())
            iterator.set_description(desc)

        print('Epoch: {}; MI_dr: {}'.format(i, mi_dr[-1]))

    return mi_dr
