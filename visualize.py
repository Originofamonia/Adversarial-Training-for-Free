"""
    Visualize hidden layer
    https://github.com/gurjaspalbedi/deep-learning-pytorch/blob/master/mnist_data_and_analysis_fully_connected.ipynb
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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from tqdm import tqdm
import random
import numpy as np
from models.wideresnet import *
from inference import adv_test
from models.iterative_projected_gradient import LinfPGDAttack


def get_hidden_layer(net, filename, x, device):
    checkpoint = torch.load(filename)
    net = net.to(device)
    net.load_state_dict(checkpoint['net'])

    net.eval()
    h1 = net.get_h1(x)
    h2 = net.get_h2(x)
    h3 = net.get_h3(x)
    h4 = net.get_h4(x)
    h = net(x)
    logits = net.h_to_logits(h)

    return h1, h2, h3, h4, logits


def draw_pca_plot(x, y):
    plt.figure(figsize=(15, 10))
    pca = PCA(2)
    principal_components = pca.fit_transform(x)
    dataframe = pd.DataFrame(data=y, columns=['digit'])
    principalDf = pd.DataFrame(data=principal_components, columns=['principal component 1', 'principal component 2'])
    finalDf = pd.concat([principalDf, dataframe], axis=1)
    # digits = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    colors = ['red', 'green', 'blue', 'purple', 'yellow', 'black', 'orange', 'brown', 'grey', 'navy']
    mean = finalDf.groupby('digit').mean()
    for digit, color in zip(digits, colors):
        indicesToKeep = finalDf['digit'] == digit
        plt.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
                    finalDf.loc[indicesToKeep, 'principal component 2'],
                    c=color)
        # plt.text(mean.loc[digit, 'principal component 1'], mean.loc[digit, 'principal component 2'], digit,
        #          fontsize=14)
    plt.title("PCA logits")
    plt.legend(digits)
    plt.grid()
    plt.show()


def draw_tsne_plot(x, y):
    plt.figure(figsize=(15, 10))
    tsne = TSNE(2)
    tsne = tsne.fit_transform(x)
    dataframe = pd.DataFrame(data=y, columns=['digit'])
    principalDf = pd.DataFrame(data=tsne, columns=['principal component 1', 'principal component 2'])
    finalDf = pd.concat([principalDf, dataframe], axis=1)
    digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    colors = ['red', 'green', 'blue', 'purple', 'yellow', 'black', 'orange', 'brown', 'grey', 'navy']
    mean = finalDf.groupby('digit').mean()
    for digit, color in zip(digits, colors):
        indicesToKeep = finalDf['digit'] == digit
        plt.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
                    finalDf.loc[indicesToKeep, 'principal component 2'],
                    c=color)
        # plt.text(mean.loc[digit, 'principal component 1'], mean.loc[digit, 'principal component 2'], digit,
        # fontsize=14)
    plt.title("tSNE")
    plt.legend(digits)
    plt.grid()
    plt.show()


def main():
    clean_model = './checkpoint/clean_ckpt41.pt'
    device = 'cpu'  # 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 50
    net = wide_resnet_34_10()
    adversary = LinfPGDAttack(
        net, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=8. / 255,
        nb_iter=20, eps_iter=2. / 255, rand_init=True, clip_min=0.0, clip_max=1.0,
        targeted=False)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # Normalization messes with l-inf bounds.
    ])
    trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, drop_last=True)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            indices = torch.tensor([0, 1, 3, 4, 6, 7, 8, 9, 27, 29])
            inputs = torch.index_select(inputs, 0, indices)
            targets = torch.index_select(targets, 0, indices)
            h1, h2, h3, h4, logits = get_hidden_layer(net, clean_model, inputs, device)
            h_orig = logits.view(10, -1)
            with torch.enable_grad():
                adv = adversary.perturb(inputs, targets)
            ah1, ah2, ah3, ah4, alogits = get_hidden_layer(net, clean_model, adv, device)
            ah = alogits.view(10, -1)
            h = torch.cat((h_orig, ah), 0)
            y = torch.cat((targets, targets), 0)
            draw_pca_plot(h.cpu().detach().numpy(), y)
            # draw_tsne_plot(h.cpu().detach().numpy(), y)


if __name__ == '__main__':
    main()
