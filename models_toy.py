import numpy as np
import math
import torch
from torch import nn
from scipy.stats import special_ortho_group
from models import RNN, RNN2, MultRNN, MultiplicativeLayer


class MapGated2RNN(nn.Module):
    def __init__(self, sh_size, hidden_size, map_size, output_size, **kwargs):
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        emb_size=100
        # n_shapes = kwargs['n_shapes'] if 'n_shapes' in kwargs.keys() else 20
        self.act = kwargs['act'] if 'act' in kwargs.keys() else None
        self.detach = kwargs['detach'] if 'detach' in kwargs.keys() else False
        drop = kwargs['dropout'] if 'dropout' in kwargs.keys() else 0
        # self.par = kwargs['parallel'] if 'parallel' in kwargs.keys() else False
        self.pix_embedding = nn.Linear(sh_size, hidden_size//2)
        # self.shape_readout = nn.Linear(hidden_size//2, n_shapes)
        self.xy_embedding = nn.Linear(2, hidden_size//2)
        self.joint_embedding = nn.Linear(hidden_size, hidden_size)
        self.rnn_map = RNN2(hidden_size, hidden_size, hidden_size, self.act)
        self.rnn_gate = RNN2(hidden_size, hidden_size, hidden_size, self.act)
        self.drop_layer = nn.Dropout(p=drop)

        self.map_readout = nn.Linear(hidden_size, map_size)
        self.gate = nn.Linear(hidden_size, map_size)
        self.after_map = nn.Linear(map_size, map_size)
        self.num_readout = nn.Linear(map_size, output_size)

        self.sigmoid = nn.Sigmoid()
        self.LReLU = nn.LeakyReLU(0.1)

    def initHidden(self, batch_size):
        hidden_m = self.rnn_map.initHidden(batch_size)
        hidden_g = self.rnn_gate.initHidden(batch_size)
        hidden = torch.cat((hidden_m, hidden_g), dim=1)
        return hidden

    def forward(self, x, hidden):
        hidden_map = hidden[:, :self.hidden_size]
        hidden_gate = hidden[:, self.hidden_size:]
        xy = x[:,:2]  # xy coords are first two input features
        shape = x[:,2:]
        # gate = self.sigmoid(self.gate(shape))
        # gated = torch.mul(x, gate)
        # xy = gated[:,:2]  # xy coords are first two input features
        # shape = gated[:,2:]
        shape_emb = self.LReLU(self.pix_embedding(shape))
        xy_emb = self.LReLU(self.xy_embedding(xy))
        combined = torch.cat((shape_emb, xy_emb), dim=-1)
        # gated_combined = torch.mul(combined, gate)
        x = self.LReLU(self.joint_embedding(combined))
        x_map, hidden_map = self.rnn_map(x, hidden_map)
        x_gate, hidden_gate = self.rnn_gate(x, hidden_gate)
        x_map = self.drop_layer(x_map)

        full_map = self.map_readout(x_map)
        # sig_full = self.sigmoid(full_map)
        gate = self.sigmoid(self.gate(x_gate))
        if self.detach:
            # map_to_pass_on = sig_full.detach().clone()
            map_to_pass_on = full_map.detach().clone()
        else:
            # map_to_pass_on = sig_full
            map_to_pass_on = full_map
        gated_map = torch.mul(map_to_pass_on, gate)
        if self.detach:
            gated_map_to_pass_on = gated_map.detach().clone()
        else:
            gated_map_to_pass_on = gated_map
        penult = self.LReLU(self.after_map(gated_map_to_pass_on))
        num = self.num_readout(penult)
        hidden = torch.cat((hidden_map, hidden_gate), dim=1)
        return num, shape, (full_map, gated_map), hidden


class MapGatedSymbolicRNN(nn.Module):
    def __init__(self, sh_size, hidden_size, map_size, output_size, **kwargs):
        super().__init__()
        self.output_size = output_size
        emb_size=100
        # n_shapes = kwargs['n_shapes'] if 'n_shapes' in kwargs.keys() else 20
        self.act = kwargs['act'] if 'act' in kwargs.keys() else None
        self.detach = kwargs['detach'] if 'detach' in kwargs.keys() else False
        drop = kwargs['dropout'] if 'dropout' in kwargs.keys() else 0
        # self.par = kwargs['parallel'] if 'parallel' in kwargs.keys() else False
        self.pix_embedding = nn.Linear(sh_size, emb_size//2)
        # self.shape_readout = nn.Linear(hidden_size//2, n_shapes)
        self.xy_embedding = nn.Linear(2, emb_size//2)
        self.joint_embedding = nn.Linear(emb_size, hidden_size)
        self.rnn = RNN2(hidden_size, hidden_size, hidden_size, self.act)
        self.drop_layer = nn.Dropout(p=drop)

        self.map_readout = nn.Linear(hidden_size, map_size)
        self.gate = nn.Linear(hidden_size, map_size)
        self.after_map = nn.Linear(map_size, map_size)
        # if self.par:
        #     self.notmap = nn.Linear(hidden_size, map_size)
        #     self.num_readout = nn.Linear(map_size * 2, output_size)
        # else:
        self.num_readout = nn.Linear(map_size, output_size)

        self.initHidden = self.rnn.initHidden
        self.sigmoid = nn.Sigmoid()
        self.LReLU = nn.LeakyReLU(0.1)

    def forward(self, x, hidden):
        xy = x[:,:2]  # xy coords are first two input features
        shape = x[:,2:]
        # gate = self.sigmoid(self.gate(shape))
        # gated = torch.mul(x, gate)
        # xy = gated[:,:2]  # xy coords are first two input features
        # shape = gated[:,2:]
        shape_emb = self.LReLU(self.pix_embedding(shape))
        xy_emb = self.LReLU(self.xy_embedding(xy))
        combined = torch.cat((shape_emb, xy_emb), dim=-1)
        # gated_combined = torch.mul(combined, gate)
        x = self.LReLU(self.joint_embedding(combined))
        x, hidden = self.rnn(x, hidden)
        x = self.drop_layer(x)

        full_map = self.map_readout(x)
        # sig_full = self.sigmoid(full_map)
        gate = self.sigmoid(self.gate(x))
        if self.detach:
            # map_to_pass_on = sig_full.detach().clone()
            map_to_pass_on = full_map.detach().clone()
        else:
            # map_to_pass_on = sig_full
            map_to_pass_on = full_map
        gated_map = torch.mul(map_to_pass_on, gate)
        penult = self.LReLU(self.after_map(gated_map))
        num = self.num_readout(penult)
        return num, shape, (full_map, gated_map), hidden


class GatedSymbolicRNN(nn.Module):
    def __init__(self, sh_size, hidden_size, map_size, output_size, **kwargs):
        super().__init__()
        self.output_size = output_size
        # n_shapes = kwargs['n_shapes'] if 'n_shapes' in kwargs.keys() else 20
        self.act = kwargs['act'] if 'act' in kwargs.keys() else None
        self.detach = kwargs['detach'] if 'detach' in kwargs.keys() else False
        drop = kwargs['dropout'] if 'dropout' in kwargs.keys() else 0
        # self.par = kwargs['parallel'] if 'parallel' in kwargs.keys() else False
        self.pix_embedding = nn.Linear(sh_size, hidden_size//2)
        # self.shape_readout = nn.Linear(hidden_size//2, n_shapes)
        self.xy_embedding = nn.Linear(2, hidden_size//2)
        self.gate = nn.Linear(sh_size, sh_size+2)
        self.joint_embedding = nn.Linear(hidden_size, hidden_size)
        self.rnn = RNN2(hidden_size, hidden_size, hidden_size, self.act)
        self.drop_layer = nn.Dropout(p=drop)
        self.map_readout = nn.Linear(hidden_size, map_size)
        self.after_map = nn.Linear(map_size, map_size)
        # if self.par:
        #     self.notmap = nn.Linear(hidden_size, map_size)
        #     self.num_readout = nn.Linear(map_size * 2, output_size)
        # else:
        self.num_readout = nn.Linear(map_size, output_size)

        self.initHidden = self.rnn.initHidden
        self.sigmoid = nn.Sigmoid()
        self.LReLU = nn.LeakyReLU(0.1)

    def forward(self, x, hidden):
        xy = x[:,:2]  # xy coords are first two input features
        shape = x[:,2:]
        gate = self.sigmoid(self.gate(shape))
        gated = torch.mul(x, gate)
        xy = gated[:,:2]  # xy coords are first two input features
        shape = gated[:,2:]
        shape_emb = self.LReLU(self.pix_embedding(shape))
        xy_emb = self.LReLU(self.xy_embedding(xy))
        combined = torch.cat((shape_emb, xy_emb), dim=-1)
        # gated_combined = torch.mul(combined, gate)
        x = self.LReLU(self.joint_embedding(combined))
        x, hidden = self.rnn(x, hidden)
        x = self.drop_layer(x)

        map_ = self.map_readout(x)
        sig = self.sigmoid(map_)
        if self.detach:
            map_to_pass_on = sig.detach().clone()
        else:
            map_to_pass_on = sig
        penult = self.LReLU(self.after_map(map_to_pass_on))
        num = self.num_readout(penult)
        return num, shape, map_, hidden

class RNNClassifier2stream(nn.Module):
    def __init__(self, pix_size, hidden_size, map_size, output_size, **kwargs):
        super().__init__()
        # map_size = 9
        self.output_size = output_size
        n_shapes = kwargs['n_shapes'] if 'n_shapes' in kwargs.keys() else 20
        self.act = kwargs['act'] if 'act' in kwargs.keys() else None
        self.detach = kwargs['detach'] if 'detach' in kwargs.keys() else False
        drop = kwargs['dropout'] if 'dropout' in kwargs.keys() else 0
        self.par = kwargs['parallel'] if 'parallel' in kwargs.keys() else False
        self.pix_embedding = nn.Linear(pix_size, hidden_size//2)
        self.shape_readout = nn.Linear(hidden_size//2, n_shapes)
        self.xy_embedding = nn.Linear(2, hidden_size//2)
        self.joint_embedding = nn.Linear(hidden_size + n_shapes, hidden_size)
        self.rnn = RNN2(hidden_size, hidden_size, hidden_size, self.act)
        self.drop_layer = nn.Dropout(p=drop)
        self.map_readout = nn.Linear(hidden_size, map_size)
        self.after_map = nn.Linear(map_size, map_size)
        if self.par:
            self.notmap = nn.Linear(hidden_size, map_size)
            self.num_readout = nn.Linear(map_size * 2, output_size)
        else:
            self.num_readout = nn.Linear(map_size, output_size)

        self.initHidden = self.rnn.initHidden
        self.sigmoid = nn.Sigmoid()
        self.LReLU = nn.LeakyReLU(0.1)

    def forward(self, x, hidden):
        xy = x[:,:2]  # xy coords are first two input features
        pix = x[:,2:]
        xy = self.LReLU(self.xy_embedding(xy))
        pix = self.LReLU(self.pix_embedding(pix))
        shape = self.shape_readout(pix)
        combined = torch.cat((shape, xy, pix), dim=-1)
        x = self.LReLU(self.joint_embedding(combined))
        x, hidden = self.rnn(x, hidden)
        x = self.drop_layer(x)

        map_ = self.map_readout(x)
        sig = self.sigmoid(map_)
        if self.detach:
            map_to_pass_on = sig.detach().clone()
        else:
            map_to_pass_on = sig

        penult = self.LReLU(self.after_map(map_to_pass_on))
        if self.par:
            # Two parallel layers, one to be a map, the other not
            notmap = self.notmap(x)
            penult = torch.cat((penult, notmap), dim=1)
        # num = self.num_readout(map_to_pass_on)
        num = self.num_readout(penult)
        return num, shape, map_, hidden


class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, map_size, output_size, **kwargs):
        super().__init__()
        # map_size = 9
        self.output_size = output_size
        self.act = kwargs['act'] if 'act' in kwargs.keys() else None
        self.detach = kwargs['detach'] if 'detach' in kwargs.keys() else False
        drop = kwargs['dropout'] if 'dropout' in kwargs.keys() else 0
        self.par = kwargs['parallel'] if 'parallel' in kwargs.keys() else False
        self.embedding = nn.Linear(input_size, hidden_size)
        self.rnn = RNN2(hidden_size, hidden_size, hidden_size, self.act)
        self.drop_layer = nn.Dropout(p=drop)
        self.map_readout = nn.Linear(hidden_size, map_size)
        if self.par:
            self.notmap = nn.Linear(hidden_size, map_size)
            self.num_readout = nn.Linear(map_size * 2, output_size)
        else:
            self.num_readout = nn.Linear(map_size, output_size)

        self.initHidden = self.rnn.initHidden
        self.sigmoid = nn.Sigmoid()
        self.LReLU = nn.LeakyReLU(0.1)

    def forward(self, x, hidden):
        x = self.LReLU(self.embedding(x))
        x, hidden = self.rnn(x, hidden)
        x = self.drop_layer(x)

        map = self.map_readout(x)
        sig = self.sigmoid(map)
        if self.detach:
            map_to_pass_on = sig.detach().clone()
        else:
            map_to_pass_on = sig
        if self.par:
            # Two parallel layers, one to be a map, the other not
            notmap = self.notmap(x)
            penult = torch.cat((map_to_pass_on, notmap), dim=1)
        else:
            penult = map_to_pass_on
        # num = self.num_readout(map_to_pass_on)
        num = self.num_readout(penult)
        return num, map, hidden


class RNNClassifier_nosymbol(nn.Module):
    def __init__(self, input_size, hidden_size, map_size, output_size, **kwargs):
        super().__init__()
        # map_size = 9
        self.act = kwargs['act'] if 'act' in kwargs.keys() else None
        self.detach = kwargs['detach'] if 'detach' in kwargs.keys() else False
        self.n_shapes = 9
        self.embedding = nn.Linear(input_size, hidden_size)
        self.rnn = RNN2(hidden_size, hidden_size, hidden_size, self.act)
        self.baby_rnn = RNN2(3, 27, 27, self.act)
        self.shape_readout = nn.Linear(27, 9)
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
        shape_emb = self.shape_readout(shape_emb)
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
        return num, map, shape_emb, hidden

class NumAsMapsum2stream(nn.Module):
    def __init__(self, pix_size, hidden_size, map_size, output_size, **kwargs):
        super().__init__()
        self.act = kwargs['act'] if 'act' in kwargs.keys() else None
        self.detach = kwargs['detach'] if 'detach' in kwargs.keys() else False
        drop = kwargs['dropout'] if 'dropout' in kwargs.keys() else 0
        self.output_size = output_size
        self.pix_embedding = nn.Linear(pix_size, hidden_size//2)
        self.xy_embedding = nn.Linear(2, hidden_size//2)
        self.joint_embedding = nn.Linear(hidden_size, hidden_size)
        self.rnn = RNN2(hidden_size, hidden_size, hidden_size, self.act)
        self.drop_layer = nn.Dropout(p=drop)
        self.map_readout = nn.Linear(hidden_size, map_size)
        self.num_readout = nn.Linear(1, 7)
        self.initHidden = self.rnn.initHidden
        self.sigmoid = nn.Sigmoid()
        self.LReLU = nn.LeakyReLU(0.1)

    def forward(self, x, hidden):
        xy = x[:, :2]  # xy coords are first two input features
        pix = x[:, 2:]
        xy = self.LReLU(self.xy_embedding(xy))
        pix = self.LReLU(self.pix_embedding(pix))
        combined = torch.cat((xy, pix), dim=-1)
        x, hidden = self.rnn(combined, hidden)
        x = self.drop_layer(x)
        map_ = self.map_readout(x)
        sig = self.sigmoid(map_)
        if self.detach:
            map_to_pass_on = sig.detach().clone()
        else:
            map_to_pass_on = sig
        # import pdb;pdb.set_trace()
        mapsum = torch.sum(map_to_pass_on, 1, keepdim=True)
        num = self.num_readout(mapsum)
        # num = torch.round(torch.sum(x, 1))
        # num_onehot = nn.functional.one_hot(num, 9)
        return num, map_, hidden

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
    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        super().__init__()
        map_size = 9
        self.act = kwargs['act'] if 'act' in kwargs.keys() else None
        self.n_shapes = 9
        self.embedding = nn.Linear(input_size, hidden_size)
        self.baby_rnn = RNN2(3, 27, 27, self.act)
        self.shape_readout = nn.Linear(27, 9)
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
        shape_emb = self.shape_readout(shape_emb)
        combined = torch.cat((xy, shape_emb), 1)
        x = self.LReLU(self.embedding(combined))
        x, hidden = self.rnn(x, hidden)
        map = self.readout(x)
        sig = self.sigmoid(map)

        num = torch.sum(sig, 1)
        # num = torch.round(torch.sum(x, 1))
        # num_onehot = nn.functional.one_hot(num, 9)
        return num, map, shape_emb, hidden

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


# class HyperModel(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super().__init__()
#         shape_size = 9
#         xy_size = 2
#         embedding_size = 25
#         factor_size = 26
#         n_z = 24
#         map_size = 9
#         # self.embedding = MultiplicativeLayer(shape_size, xy_size, embedding_size)
#         self.embedding = nn.Linear(input_size, hidden_size)
#         # (self, input_size: int, hidden_size: int, hyper_size: int, n_z: int, n_layers: int)
#         self.rnn = HyperLSTM(embedding_size, hidden_size, factor_size, n_z, 2)
#         self.readout = nn.Linear(hidden_size, map_size)
#         # self.initHidden = self.rnn.initHidden
#         self.sigmoid = nn.Sigmoid()
#         self.LReLU = nn.LeakyReLU(0.1)
#
#     def forward(self, x, hidden):
#         # xy = x[:, :2]
#         # shape = x[:, 2:]
#         # x = self.LReLU(self.embedding(xy, shape))
#
#         x = self.LReLU(self.embedding(x))
#         x, hidden = self.rnn(x, hidden)
#         map = self.readout(x)
#         sig = self.sigmoid(map)
#         num = torch.sum(sig, 1)
#
#         return num, map, hidden
#
#     def initHidden(self, batch_size):
#         return None
