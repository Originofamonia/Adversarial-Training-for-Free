"""
Use Deep-Infomax's global and local maximization to estimate I(x, y), I(h, y)
https://github.com/Originofamonia/DeepInfomaxPytorch
https://github.com/Originofamonia/Deep-INFOMAX.git
"""

import torch
from torchvision.datasets.cifar import CIFAR10
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
import statistics as stats
import argparse

from mutual_info.dim_models import GlobalDiscriminator, LocalDiscriminator, DeepInfoMaxLoss
from models.wideresnet import wide_resnet_34_10


def train_dim(args, cifar_10_train_dt, cifar_10_train_l, device, loss_fn, loss_optim, optim, wrn):
    for epoch in range(1, args.epochs):
        batch = tqdm(cifar_10_train_l, total=len(cifar_10_train_dt) // args.batch_size)
        train_loss = []
        for x, target in batch:
            x = x.to(device)
            y = target.unsqueeze(-1)
            y_onehot = torch.zeros(y.shape[0], 10).scatter_(1, y, 1).to(device)
            optim.zero_grad()
            loss_optim.zero_grad()
            M = wrn(x)

            # rotate images to create pairs for comparison
            # put the first img to the end; unsqueeze is adding another dimension of the tensor
            M_prime = torch.cat((M[1:], M[0].unsqueeze(0)), dim=0)
            loss = loss_fn(y_onehot, M, M_prime)
            train_loss.append(loss.item())
            descr = 'Epoch: {}, Loss:{:10.4f}'.format(epoch, stats.mean(train_loss[-20:]))
            batch.set_description(descr)
            loss.backward()
            optim.step()
            loss_optim.step()

    torch.save(loss_fn.state_dict(), 'checkpoint/dim.{}'.format(args.epoch))


def main():
    parser = argparse.ArgumentParser(description='Deep-Infomax pytorch')
    parser.add_argument('--batch_size', default=5, type=int, help='batch_size')
    parser.add_argument('--epochs', default=10, type=int, help='epochs')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    # parser.add_argument('--ish', default=True, type=bool, help='is h?')  # whether I(x, y): False or I(h, y): True
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # image size 3, 32, 32
    # batch size must be an even number
    # shuffle must be True
    cifar_10_train_dt = CIFAR10('data', train=True, download=True, transform=ToTensor())
    cifar_10_train_l = DataLoader(cifar_10_train_dt, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                  pin_memory=torch.cuda.is_available())

    wrn = wide_resnet_34_10().to(device)
    loss_fn = DeepInfoMaxLoss().to(device)
    optim = Adam(wrn.parameters(), lr=args.lr)
    loss_optim = Adam(loss_fn.parameters(), lr=args.lr)

    train_dim(args, cifar_10_train_dt, cifar_10_train_l, device, loss_fn, loss_optim, optim, wrn)
    # TODO: add an inference function to get local and global MI.


if __name__ == '__main__':
    main()
