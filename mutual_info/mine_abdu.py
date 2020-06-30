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
    mi_lb_list = list()

    for i in range(1, epochs + 1):
        mine_net.train()

        for j, data in enumerate(iterator):
            x_sample, y_sample, y_shuffle = sample_batch(data, device, robust_net, h)
            pred_xy = mine_net(x_sample, y_sample)
            pred_x_y = mine_net(x_sample, y_shuffle)
            mi_lb = torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y)))
            loss = -mi_lb
            mine_net.zero_grad()
            loss.backward()
            mine_net_optim.step()
            mi_lb_list.append(mi_lb.detach().cpu().numpy())
            desc = 'mi_lb: ' + "{:10.4f}".format(mi_lb.item())
            iterator.set_description(desc)

        print('Epoch: {}; mi_lb: {}'.format(i, mi_lb_list[-1]))

    return mi_lb_list
