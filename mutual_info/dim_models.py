
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn as nn


class GlobalDiscriminator(nn.Module):
    def __init__(self):
        super(GlobalDiscriminator, self).__init__()
        # self.ish = ish

        self.c0 = nn.Conv2d(640, 64, kernel_size=3)
        self.l0 = nn.Linear(522, 512)

        self.c1 = nn.Conv2d(64, 32, kernel_size=3)
        self.l1 = nn.Linear(512, 512)
        self.l2 = nn.Linear(512, 1)

    def forward(self, y, M):
        hidden = F.relu(self.c0(M))
        hidden = self.c1(hidden)
        hidden = hidden.view(y.shape[0], -1)
        hidden = torch.cat((y, hidden), dim=1)
        hidden = F.relu(self.l0(hidden))
        hidden = F.relu(self.l1(hidden))
        hidden = self.l2(hidden)
        return hidden


class LocalDiscriminator(nn.Module):
    def __init__(self):
        super(LocalDiscriminator, self).__init__()
        # self.ish = ish
        in_channel = 650
        self.c0 = nn.Conv2d(in_channel, 512, kernel_size=1)
        self.c1 = nn.Conv2d(512, 512, kernel_size=1)
        self.c2 = nn.Conv2d(512, 1, kernel_size=1)

    def forward(self, x):
        hidden = F.relu(self.c0(x))
        hidden = F.relu(self.c1(hidden))
        return self.c2(hidden)


class DeepInfoMaxLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(DeepInfoMaxLoss, self).__init__()
        self.global_d = GlobalDiscriminator()
        self.local_d = LocalDiscriminator()
        self.alpha = alpha
        self.beta = beta
        # self.ish = ish

    def forward(self, y, M, M_prime):
        # see appendix 1A of https://arxiv.org/pdf/1808.06670.pdf
        y_exp = y.unsqueeze(-1).unsqueeze(-1)
        y_exp = y_exp.expand(-1, -1, 8, 8)  # to match h: [640, 8, 8]

        y_M = torch.cat((M, y_exp), dim=1)
        y_M_prime = torch.cat((M_prime, y_exp), dim=1)

        Ej = -F.softplus(-self.local_d(y_M)).mean()
        Em = F.softplus(self.local_d(y_M_prime)).mean()
        LOCAL = (Em - Ej) * self.beta

        Ej = -F.softplus(-self.global_d(y, M)).mean()
        Em = F.softplus(self.global_d(y, M_prime)).mean()
        GLOBAL = (Em - Ej) * self.alpha

        return LOCAL + GLOBAL

    def get_local(self, y, M, M_prime):
        y_exp = y.unsqueeze(-1).unsqueeze(-1)
        y_exp = y_exp.expand(-1, -1, 8, 8)  # to match h: [640, 8, 8]

        y_M = torch.cat((M, y_exp), dim=1)
        y_M_prime = torch.cat((M_prime, y_exp), dim=1)

        Ej = -F.softplus(-self.local_d(y_M)).mean()
        Em = F.softplus(self.local_d(y_M_prime)).mean()
        local = (Em - Ej) * self.beta
        return local

    def get_global(self, y, M, M_prime):
        Ej = -F.softplus(-self.global_d(y, M)).mean()
        Em = F.softplus(self.global_d(y, M_prime)).mean()
        GLOBAL = (Em - Ej) * self.alpha
        return GLOBAL
