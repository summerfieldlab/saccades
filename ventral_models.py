import numpy as np
import math
import torch
from torch import nn
from scipy.stats import special_ortho_group
from models import RNN, RNN2, MultRNN, MultiplicativeLayer


class MLP(nn.Module):
    def __init__(self, input_size, layer_width, n_layers, output_size, drop=0.5):
        super().__init__()
        self.n_layers = n_layers
        self.layer0 = nn.Linear(input_size, layer_width)
        self.BN0 = torch.nn.BatchNorm1d(layer_width)
        self.layer1 = nn.Linear(layer_width, layer_width)
        self.BN1 = torch.nn.BatchNorm1d(layer_width)
        self.layer2 = nn.Linear(layer_width, layer_width)
        self.BN2 = torch.nn.BatchNorm1d(layer_width)
        self.drop_layer = nn.Dropout(p=drop)
        self.layer3 = nn.Linear(layer_width, 100)
        self.BN3 = torch.nn.BatchNorm1d(100)
        # self.layers = [self.layer1, self.layer2, self.layer3]
        self.out = nn.Linear(100, output_size)
        self.LReLU = nn.LeakyReLU(0.1)


    def forward(self, x):
        x = self.LReLU(self.BN0(self.layer0(x)))
        x = self.LReLU(self.BN1(self.layer1(x)))
        x = self.LReLU(self.BN2(self.layer2(x)))
        x = self.drop_layer(x)
        x = self.LReLU(self.BN3(self.layer3(x)))
        # for layerid in range(self.n_layers):
        #     x = self.LReLU(self.layers[layerid](x))
        pred = self.out(x)
        pred = torch.clamp(pred, -1e6, 1e6)
        return pred, x
