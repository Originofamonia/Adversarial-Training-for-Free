"""
Use MINE to estimate the MI(x, y) and MI(h, y)
https://github.com/sungyubkim/MINE-Mutual-Information-Neural-Estimation-/blob/master/MINE.ipynb
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

from models.resnet import ResNet34
from models.wideresnet import wide_resnet_34_10, adjust_learning_rate


def mutual_information(joint, marginal, mine_net):
    t = mine_net(joint)
    et = torch.exp(mine_net(marginal))
    mi_lb = torch.mean(t) - torch.log(torch.mean(et - 1))
    return mi_lb, t, et


def learn_mine(batch, mine_net, mine_net_optim, ma_et, ma_rate=0.01):
    # batch is a tuple of (joint, marginal)
    joint, marginal = batch
    # joint = torch.autograd.Variable(torch.FloatTensor(joint)).cuda()
    # marginal = torch.autograd.Variable(torch.FloatTensor(marginal)).cuda()
    mi_lb, t, et = mutual_information(joint, marginal, mine_net)
    ma_et = (1 - ma_rate) * ma_et + ma_rate * torch.mean(et)

    # unbiasing use moving average
    loss = -(torch.mean(t) - (1 / ma_et.mean()).detach() * torch.mean(et))
    # use biased estimator
    #     loss = - mi_lb

    mine_net_optim.zero_grad()
    autograd.backward(loss)
    mine_net_optim.step()
    return mi_lb, ma_et


def sample_batch(data, device, robust_net, h, sample_mode='joint'):
    x, y = data
    x = x.to(device)
    y = y.type(torch.float).to(device)
    y = (y - 0.45) / 9  # normalize y
    if h:
        x = robust_net(x)
        dims = 1, 640, 8, 8
    else:
        dims = 1, 3, 32, 32
    if sample_mode == 'joint':
        y = y.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        y = y.repeat(dims).type(torch.float)
        batch = torch.cat((x, y), 1)
    else:
        index = torch.randperm(y.shape[0])
        y = y[index]
        y = y.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        y = y.repeat(dims).type(torch.float)
        batch = torch.cat((x, y), 1)
    return batch


def ma(a, window_size=100):
    return [np.mean(a[i:i+window_size]) for i in range(0, len(a)-window_size)]


def train(trainloader, testloader, robust_net, mine_net, mine_net_optim, device, epochs, h):
    #
    iterator = tqdm(trainloader, ncols=0, leave=False)
    mi_lb_list = list()
    ma_et = 1.
    for i in range(1, epochs + 1):
        mine_net.train()
        # adjust_learning_rate(mine_net_optim, i)
        for j, data in enumerate(iterator):
            batch = sample_batch(data, device, robust_net, h), \
                 sample_batch(data, device, robust_net, h, sample_mode='marginal')
            mi_lb, ma_et = learn_mine(batch, mine_net, mine_net_optim, ma_et)
            mi_lb_list.append(mi_lb.detach().cpu().numpy())
            desc = 'mi_lb: ' + "{:10.4f}".format(mi_lb.item())
            iterator.set_description(desc)

        print('Epoch: {}; mi_lb: {}'.format(i, mi_lb_list[-1]))

    return mi_lb_list


def get_cifar10(batch_size):
    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # Normalization messes with l-inf bounds.
    ])

    trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])
    testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader


def draw():
    plt.figure(figsize=[8, 5])
    mi_hy = 'mutual_info/mi_hy.txt'
    mi_xy = 'mutual_info/mi_xy.txt'
    hy = np.loadtxt(mi_hy)
    xy = np.loadtxt(mi_xy)
    # hy = ma(hy)
    # xy = ma(xy)
    plot_x = np.arange(len(hy))
    plt.plot(plot_x, hy, label='I(h, y)')
    plt.plot(plot_x, xy, label='I(x, y)')

    plt.xlabel('Iteration')
    plt.ylabel('Mutual Information')
    plt.legend()
    plt.savefig('mine_fig/hy_xy.png', dpi=300)


def calc_mi():
    parser = argparse.ArgumentParser(description='MINE robust model')
    parser.add_argument('--seed', default=9527, type=int)
    parser.add_argument('--epochs', default=21, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-3, type=float)
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
        in_channels = 1280  # 6 = 3 * 2 for MI(x, y), 1280 = 640 * 2 for MI(h, y)
    else:
        in_channels = 6

    resnet34 = ResNet34(in_channels, args.h).cuda()
    trainloader, testloader = get_cifar10(args.batch_size)
    optimizer = optim.Adam(resnet34.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    mi_lb_list = train(trainloader, testloader, robust_net, resnet34, optimizer, device, args.epochs, args.h)
    result_cor_ma = ma(mi_lb_list)
    print('h: {}, last MI: {}'.format(args.h, result_cor_ma[-1]))
    if args.h:
        with open('mi_hy.txt', 'w') as filehandle:
            for listitem in result_cor_ma:
                filehandle.write('%s\n' % listitem)
    else:
        with open('mi_xy.txt', 'w') as filehandle:
            for listitem in result_cor_ma:
                filehandle.write('%s\n' % listitem)


if __name__ == '__main__':
    # calc_mi()
    draw()
