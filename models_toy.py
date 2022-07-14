import numpy as np
import math
import torch
from torch import nn
from scipy.stats import special_ortho_group
from models import RNN, RNN2, MultRNN, MultiplicativeLayer


class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        super().__init__()
        map_size = 9
        self.act = kwargs['act'] if 'act' in kwargs.keys() else None
        self.detach = kwargs['detach'] if 'detach' in kwargs.keys() else True
        self.embedding = nn.Linear(input_size, hidden_size)
        self.rnn = RNN2(hidden_size, hidden_size, hidden_size, self.act)
        self.map_readout = nn.Linear(hidden_size, map_size)
        self.num_readout = nn.Linear(map_size, output_size)
        self.initHidden = self.rnn.initHidden
        self.sigmoid = nn.Sigmoid()
        self.LReLU = nn.LeakyReLU(0.1)

    def forward(self, x, hidden):
        x = self.LReLU(self.embedding(x))
        x, hidden = self.rnn(x, hidden)
        map = self.map_readout(x)
        sig = self.sigmoid(map)
        if self.detach:
            map_to_pass_on = sig.detach().clone()
        else:
            map_to_pass_on = sig
        num = self.num_readout(map_to_pass_on)
        return num, map, hidden


class RNNClassifier_nosymbol(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        super().__init__()
        map_size = 9
        self.act = kwargs['act'] if 'act' in kwargs.keys() else None
        self.detach = kwargs['detach'] if 'detach' in kwargs.keys() else True
        self.n_shapes = 9
        self.embedding = nn.Linear(input_size, hidden_size)
        self.rnn = RNN2(hidden_size, hidden_size, hidden_size, self.act)
        self.baby_rnn = RNN2(3, 9, 9, self.act)
        self.map_readout = nn.Linear(hidden_size, map_size)
        self.num_readout = nn.Linear(map_size, output_size)
        self.initHidden = self.rnn.initHidden
        self.sigmoid = nn.Sigmoid()
        self.LReLU = nn.LeakyReLU(0.1)

    def forward(self, xy, shape, hidden):
        """shape here will be batchsize x n_kinds_of_shapes x 3."""
        batch_size = xy.shape[0]
        baby_hidden = self.baby_rnn.initHidden(batch_size)
        baby_hidden = baby_hidden.to(shape.device)
        for i in range(self.n_shapes):
            shape_emb, baby_hidden = self.baby_rnn(shape[:, i, :], baby_hidden)
        combined = torch.cat((xy, shape_emb), 1)
        x = self.LReLU(self.embedding(combined))
        x, hidden = self.rnn(x, hidden)
        map = self.map_readout(x)
        sig = self.sigmoid(map)
        if self.detach:
            map_to_pass_on = sig.detach().clone()
        else:
            map_to_pass_on = sig
        num = self.num_readout(map_to_pass_on)
        return num, map, hidden

class NumAsMapsum(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        super().__init__()
        map_size = 9
        self.act = kwargs['act'] if 'act' in kwargs.keys() else None
        self.embedding = nn.Linear(input_size, hidden_size)
        self.rnn = RNN2(hidden_size, hidden_size, hidden_size, self.act)
        self.readout = nn.Linear(hidden_size, map_size)
        self.initHidden = self.rnn.initHidden
        self.sigmoid = nn.Sigmoid()
        self.LReLU = nn.LeakyReLU(0.1)

    def forward(self, x, hidden):
        x = self.LReLU(self.embedding(x))
        x, hidden = self.rnn(x, hidden)
        map = self.readout(x)
        sig = self.sigmoid(map)

        num = torch.sum(sig, 1)
        # num = torch.round(torch.sum(x, 1))
        # num_onehot = nn.functional.one_hot(num, 9)
        return num, map, hidden

    def init_small(self):
        pass

class NumAsMapsum_nosymbol(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, kwargs):
        super().__init__()
        map_size = 9
        self.act = kwargs['act'] if 'act' in kwargs.keys() else None
        self.n_shapes = 9
        self.embedding = nn.Linear(input_size, hidden_size)
        self.baby_rnn = RNN2(3, 9, 9, self.act)
        self.rnn = RNN2(hidden_size, hidden_size, hidden_size, self.act)
        self.readout = nn.Linear(hidden_size, map_size)
        self.initHidden = self.rnn.initHidden
        self.sigmoid = nn.Sigmoid()
        self.LReLU = nn.LeakyReLU(0.1)

    def forward(self, xy, shape, hidden):
        """shape here will be batchsize x n_kinds_of_shapes x 3."""
        batch_size = xy.shape[0]
        baby_hidden = self.baby_rnn.initHidden(batch_size)
        baby_hidden = baby_hidden.to(shape.device)
        for i in range(self.n_shapes):
            shape_emb, baby_hidden = self.baby_rnn(shape[:, i, :], baby_hidden)
        combined = torch.cat((xy, shape_emb), 1)
        x = self.LReLU(self.embedding(combined))
        x, hidden = self.rnn(x, hidden)
        map = self.readout(x)
        sig = self.sigmoid(map)

        num = torch.sum(sig, 1)
        # num = torch.round(torch.sum(x, 1))
        # num_onehot = nn.functional.one_hot(num, 9)
        return num, map, hidden

#
# class RNNClassifier_nosymbol(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, act=None):
#         super().__init__()
#         map_size = 9
#         self.n_shapes = 9
#         self.embedding = nn.Linear(input_size, hidden_size)
#         self.baby_rnn = RNN2(3, 9, 9, act)
#         self.rnn = RNN2(hidden_size, hidden_size, hidden_size, act)
#         self.readout = nn.Linear(hidden_size, output_size)
#         self.initHidden = self.rnn.initHidden
#         self.LReLU = nn.LeakyReLU(0.1)
#         # self.sigmoid = nn.Sigmoid()
#
#     def forward(self, xy, shape, hidden):
#         """shape here will be batchsize x n_kinds_of_shapes x 3."""
#         batch_size = xy.shape[0]
#         baby_hidden = self.baby_rnn.initHidden(batch_size)
#         baby_hidden = baby_hidden.to(shape.device)
#         for i in range(self.n_shapes):
#             shape_emb, baby_hidden = self.baby_rnn(shape[:, i, :], baby_hidden)
#         combined = torch.cat((xy, shape_emb), 1)
#         x = self.LReLU(self.embedding(combined))
#         x, hidden = self.rnn(x, hidden)
#         num = self.readout(x)
#         return num, shape_emb, hidden
#
#     def init_small(self):
#         pass
#
#
# class RNNClassifier(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, act=None):
#         super().__init__()
#         map_size = 9
#         self.embedding = nn.Linear(input_size, hidden_size)
#         self.rnn = RNN2(hidden_size, hidden_size, hidden_size, act)
#         self.readout = nn.Linear(hidden_size, output_size)
#         self.initHidden = self.rnn.initHidden
#         self.LReLU = nn.LeakyReLU(0.1)
#         # self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x, hidden):
#         x = self.LReLU(self.embedding(x))
#         x, hidden = self.rnn(x, hidden)
#         num = self.readout(x)
#         return num, None, hidden
#
#     def init_small(self):
#         pass


class RNNRegression(RNNClassifier):
    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        super().__init__(input_size, hidden_size, output_size, **kwargs)
        map_size = 9
        self.act = kwargs['act'] if 'act' in kwargs.keys() else None
        self.embedding = nn.Linear(input_size, hidden_size)
        self.rnn = RNN2(hidden_size, hidden_size, hidden_size, self.act)
        self.readout = nn.Linear(hidden_size, 1)
        self.initHidden = self.rnn.initHidden
        self.LReLU = nn.LeakyReLU(0.1)
        # self.sigmoid = nn.Sigmoid()


class MultiplicativeModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, small_weights):
        super().__init__()
        shape_size = 9
        xy_size = 2
        embedding_size = 25
        factor_size = 26
        map_size = 9
        if input_size == xy_size:
            self.embedding = nn.Linear(input_size, embedding_size)
        else:
            self.embedding = MultiplicativeLayer(shape_size, xy_size, embedding_size, small_weights)
        # self.embedding = nn.Linear(input_size, hidden_size)
        self.rnn = MultRNN(embedding_size, hidden_size, factor_size, hidden_size, small_weights)
        self.readout = nn.Linear(hidden_size, map_size)
        if small_weights:
            nn.init.normal_(self.readout.weight, mean=0, std=0.1)
        self.initHidden = self.rnn.initHidden
        self.sigmoid = nn.Sigmoid()
        self.LReLU = nn.LeakyReLU(0.1)

    def forward(self, x, hidden):
        if x.shape[1] > 2:
            xy = x[:, :2]
            shape = x[:, 2:]
            x = self.LReLU(self.embedding(xy, shape))
        else:
            x = self.LReLU(self.embedding(x))
        x, hidden = self.rnn(x, hidden)
        map = self.readout(x)
        sig = self.sigmoid(map)
        num = torch.sum(sig, 1)

        return num, map, hidden


class HyperModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        shape_size = 9
        xy_size = 2
        embedding_size = 25
        factor_size = 26
        n_z = 24
        map_size = 9
        # self.embedding = MultiplicativeLayer(shape_size, xy_size, embedding_size)
        self.embedding = nn.Linear(input_size, hidden_size)
        # (self, input_size: int, hidden_size: int, hyper_size: int, n_z: int, n_layers: int)
        self.rnn = HyperLSTM(embedding_size, hidden_size, factor_size, n_z, 2)
        self.readout = nn.Linear(hidden_size, map_size)
        # self.initHidden = self.rnn.initHidden
        self.sigmoid = nn.Sigmoid()
        self.LReLU = nn.LeakyReLU(0.1)

    def forward(self, x, hidden):
        # xy = x[:, :2]
        # shape = x[:, 2:]
        # x = self.LReLU(self.embedding(xy, shape))

        x = self.LReLU(self.embedding(x))
        x, hidden = self.rnn(x, hidden)
        map = self.readout(x)
        sig = self.sigmoid(map)
        num = torch.sum(sig, 1)

        return num, map, hidden

    def initHidden(self, batch_size):
        return None
