"""
Use MINE to estimate the MI(x, y) and MI(h, y)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import torchvision
import torchvision.transforms as transforms
import numpy as np
from models.resnet import ResNet34


def mutual_information(joint, marginal, mine_net):
    t = mine_net(joint)
    et = torch.exp(mine_net(marginal))
    mi_lb = torch.mean(t) - torch.log(torch.mean(et))
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


def sample_batch(data, device, sample_mode='joint'):
    """
    TODO: change this function to CIFAR10
    :param data:
    :param batch_size:
    :param sample_mode:
    :return:
    """
    x, y = data
    x = x.to(device)
    y = y.to(device)
    if sample_mode == 'joint':
        y = y.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        y = y.repeat(1, 3, 32, 32).type(torch.float)  # this dimension is only for I(x, y)
        batch = torch.cat((x, y), 1)
    else:
        index = torch.randperm(y.shape[0])
        y = y[index]
        y = y.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        y = y.repeat(1, 3, 32, 32).type(torch.float)
        batch = torch.cat((x, y), 1)
    return batch


def ma(a, window_size=100):
    return [np.mean(a[i:i+window_size]) for i in range(0, len(a)-window_size)]


def train(trainloader, testloader, model, mine_net, mine_net_optim, device, epochs=int(1e+1),
          log_freq=int(1e+3)):
    # TODO: adjust for CIFAR10 and ResNet34
    mi_lb_list = list()
    ma_et = 1.
    for i in range(1, epochs + 1):
        mine_net.train()
        for j, data in enumerate(trainloader, 0):
            batch = sample_batch(data, device), \
                 sample_batch(data, device, sample_mode='marginal')
            mi_lb, ma_et = learn_mine(batch, mine_net, mine_net_optim, ma_et)
        mi_lb_list.append(mi_lb.detach().cpu().numpy())
        if (i + 1) % log_freq == 0:
            print(mi_lb_list[-1])
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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, drop_last=True)
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])
    testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    robust_model = '../checkpoint/adv_ckpt_41.pt'
    in_channels = 6  # 6 = 3 * 2 for MI(x, y), 1280 = 640 * 2 for MI(h, y)
    learning_rate = 1e-3
    batch_size = 100
    resnet34 = ResNet34(in_channels).cuda()
    trainloader, testloader = get_cifar10(batch_size)
    optimizer = optim.Adam(resnet34.parameters(), lr=learning_rate)
    mi_lb_list = train(trainloader, testloader, robust_model, resnet34, optimizer, device)
    result_cor_ma = ma(mi_lb_list)
    print(result_cor_ma[-1])
    with open('mi_lb_list.txt', 'w') as filehandle:
        for listitem in result_cor_ma:
            filehandle.write('%s\n' % listitem)


if __name__ == '__main__':
    main()
