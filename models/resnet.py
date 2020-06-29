"""ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
    https://github.com/kuangliu/pytorch-cifar
"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, dropout, stride=1):
        super(BasicBlock, self).__init__()
        self.dropout = dropout
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.dropout > 0:
            out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# Used to calculate MI lower bound
class ResNet(nn.Module):
    def __init__(self, block, in_channels, h, num_blocks, dropout=0.3):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.in_channels = in_channels
        self.h = h
        self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], dropout, stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], dropout, stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], dropout, stride=2)
        # self.layer4 = self._make_layer(block, 512, num_blocks[3], dropout, stride=2)
        self.pool = nn.AvgPool2d(4)
        self.linear = nn.Linear(1024, 1)

    def _make_layer(self, block, planes, num_blocks, dropout, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = self.layer4(out)
        if not self.h:
            out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet34(in_channels, h):
    return ResNet(BasicBlock, in_channels, h, [3, 4, 6])


def get_cifar10():
    training_data = datasets.CIFAR10(root="data", train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
                                     ]))

    test_data = datasets.CIFAR10(root="data", train=False, download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
                                 ]))

    return training_data, test_data


def eval_accuracy(model, validation_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in validation_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_xent_losses = []
    in_channels = 6
    net = ResNet34(in_channels).to(device)
    training_data, test_data = get_cifar10()
    learning_rate = 1e-3
    batch_size = 64
    num_epochs = 30
    training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    validation_loader = DataLoader(test_data, batch_size=100, shuffle=True, pin_memory=True)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)  # lr decay
    net.param.require_grad = False
    for j in range(1, num_epochs + 1):
        start_time = time.time()
        net.train()
        for i, data in enumerate(training_loader, 0):
            train_x, train_y = data
            train_x = train_x.to(device)
            train_y = train_y.to(device)
            optimizer.zero_grad()

            logits = net(train_x)
            criterion = nn.CrossEntropyLoss()
            xent_loss = criterion(logits, train_y)
            xent_loss.backward()
            optimizer.step()

            train_xent_losses.append(xent_loss.item())

        scheduler.step()
        end_time = time.time()
        acc = eval_accuracy(net, validation_loader, device)
        print('Epoch: {}, xent_loss: {}, time elapse for current epoch: {}, '
              'test accuracy: {}'.format(j, xent_loss, end_time - start_time, acc))


if __name__ == '__main__':
    main()
