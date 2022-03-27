import sys
import os
from datetime import datetime
import argparse
from itertools import combinations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.functional import one_hot
torch.set_num_threads(15)
torch.autograd.set_detect_anomaly(True)
import math
import random
import operator as op
from functools import reduce
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score
from matplotlib import pyplot as plt


def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom

def random_combination(iterable, r):
    "Random selection from itertools.combinations(iterable, r)"
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), r))
    return tuple(pool[i] for i in indices)

class Timer():
    """A class for timing code execution.
    Copied from HRS.
    """
    def __init__(self):
        self.start = datetime.now()
        self.end = None
        self.elapsed_time = None

    def stop_timer(self):
        self.end = datetime.now()
        self.elapsed_time = self.end - self.start
        print('Execution time: {}'.format(self.elapsed_time))

# Models for min task
class ContentGated_cheat(nn.Module):
    """Same as ContentGated except the shape label is passed directly.

    No pixel processing involved. For debugging."""
    def __init__(self, hidden_size, map_size, output_size, act=None):
        super().__init__()
        total_number_size = output_size
        min_number_size = 4
        min_shape_size = 4
        self.hidden_size = hidden_size
        self.map_size = map_size
        self.out_size = output_size

        self.dorsal = TwoStepModel(map_size, hidden_size, map_size, total_number_size, act=act)

        self.softmax = nn.LogSoftmax(dim=1)

        size_prod = min_shape_size * total_number_size
        self.min_shape_readout = nn.Linear(size_prod, min_number_size)
        self.min_num_readout = nn.Linear(size_prod, min_number_size)


    def forward(self, gaze, shape, hidden):
        # From the gaze, Dorsal stream creates a map of objects in the image
        # and counts them
        number, map_, hidden[0, :, :] = self.dorsal(gaze, hidden[0, :, :])

        # The same dorsal module (the same parameters) is applied to shape-
        # gated versions of the gaze, with separate hidden states for each
        # shape, producing shape-specific counts. No supervision is provided
        # for these shape-specific counts, so they are also free to take a
        # different form.
        bs, size = gaze.shape
        gate = [label.expand(size, bs).T for label in shape.T]
        n_hrt, map_hrt, hidden[1, :, :] = self.dorsal(gaze * gate[0], hidden[1, :, :])
        n_star, map_star, hidden[2, :, :] = self.dorsal(gaze * gate[1], hidden[2, :, :])
        n_sqre, map_sqre, hidden[3, :, :] = self.dorsal(gaze * gate[2], hidden[3, :, :])
        n_tri, map_tri, hidden[4, :, :] = self.dorsal(gaze * gate[3], hidden[4, :, :])
        counts = [number, n_hrt, n_star, n_sqre, n_tri]
        counts = [self.softmax(count) for count in counts]
        maps = [map_, map_hrt, map_star, map_sqre, map_tri]

        # These shape-specific counts are contcatenated and passed to linear
        # readout layers for min shape and min number
        combined = torch.cat((n_hrt, n_star, n_sqre, n_tri), dim=1)
        min_shape = self.min_shape_readout(combined)
        min_number = self.min_num_readout(combined)

        return min_shape, min_number, counts, maps, hidden

# Models with integrated pixel and gaze streams
class IntegratedModel(nn.Module):
    def __init__(self, input_size, hidden_size, map_size, output_size, **kwargs):
        super().__init__()
        act = kwargs['act'] if 'act' in kwargs.keys() else None
        eye_weight = kwargs['eye_weight'] if 'eye_weight' in kwargs.keys() else False
        detached = kwargs['detached'] if 'detached' in kwargs.keys() else False
        dropout = kwargs['dropout'] if 'dropout' in kwargs.keys() else 0.0
        self.hidden_size = hidden_size
        self.map_size = map_size
        self.out_size = output_size
        self.detached = detached
        # detached=False should always be false for the twostep model here,
        # self.pix_fc1 = nn.Linear(1152, 2000)
        # self.pix_fc2 = nn.Linear(2000, map_size)
        self.pix_fc1 = nn.Linear(1152, map_size)

        # even if we want to detach in the forward of this class
        self.rnn = TwoStepModel(map_size*2, hidden_size, map_size, output_size, detached=detached, dropout=0)
        # self.rnn = tanhRNN(input_size, hidden_size, map_size)
        # self.readout1 = nn.Linear(map_size, map_size, bias=True)
        self.readout = nn.Linear(map_size, output_size, bias=True)

    def forward(self, x, hidden):
        gaze = x[:, :self.map_size]
        pix = x[:, self.map_size:]
        pix = torch.relu(self.pix_fc1(pix))
        # pix = torch.relu(self.pix_fc2(pix))
        combined = torch.cat((gaze, pix), dim=1)
        number, map_, hidden = self.rnn(combined, hidden)
        return number, map_, hidden

class IntegratedSkipModel(nn.Module):
    def __init__(self, input_size, hidden_size, map_size, output_size, **kwargs):
        super().__init__()
        act = kwargs['act'] if 'act' in kwargs.keys() else None
        eye_weight = kwargs['eye_weight'] if 'eye_weight' in kwargs.keys() else False
        detached = kwargs['detached'] if 'detached' in kwargs.keys() else False
        dropout = kwargs['dropout'] if 'dropout' in kwargs.keys() else 0.0
        self.hidden_size = hidden_size
        self.map_size = map_size
        self.out_size = output_size
        self.detached = detached
        # detached=False should always be false for the twostep model here,
        self.pix_fc1 = nn.Linear(1152, map_size)

        # even if we want to detach in the forward of this class
        # self.rnn = TwoStepModel(map_size*2, hidden_size, map_size, output_size, detached=detached, dropout=0)
        self.map_rnn = RNN(map_size*2, hidden_size, map_size, **kwargs)
        self.pix_rnn = RNN(map_size, map_size * 1.1, map_size, **kwargs)
        # self.rnn = tanhRNN(input_size, hidden_size, map_size)
        # self.readout1 = nn.Linear(map_size, map_size, bias=True)
        self.readout = nn.Linear(map_size*2, output_size, bias=True)

    def forward(self, x, hidden):
        gaze = x[:, :self.map_size]
        pix = x[:, self.map_size:]
        pix = torch.relu(self.pix_fc1(pix))

        combined = torch.cat((gaze, pix), dim=1)
        map_, hidden = self.map_rnn(combined, hidden)
        if self.detached:
            map_to_pass_on = map_.detach()
        else:
            map_to_pass_on = map_
        map_to_pass_on = torch.tanh(torch.relu(map_to_pass_on))

        pix = self.pix_rnn(pix)
        map_with_skippix = torch.cat((map_to_pass_on, pix), dim=1)
        number = self.readout(map_with_skippix)
        return number, map_, hidden

# Gaze only models
class ThreeStepModel(nn.Module):
    def __init__(self, input_size, hidden_size, map_size, output_size, **kwargs):
        super().__init__()
        act = kwargs['act'] if 'act' in kwargs.keys() else None
        eye_weight = kwargs['eye_weight'] if 'eye_weight' in kwargs.keys() else False
        detached = kwargs['detached'] if 'detached' in kwargs.keys() else False
        dropout = kwargs['dropout'] if 'dropout' in kwargs.keys() else 0.0
        self.hidden_size = hidden_size
        self.map_size = map_size
        self.out_size = output_size
        self.detached = detached
        # detached=False should always be false for the twostep model here,
        # even if we want to detach in the forward of this class
        # self.rnn = TwoStepModel(input_size, hidden_size, map_size, map_size, detached=False, dropout=0)
        self.rnn = RNN(input_size, hidden_size, map_size)
        # self.rnn = tanhRNN(input_size, hidden_size, map_size)
        self.readout1 = nn.Linear(map_size, map_size//2, bias=True)
        self.readout2 = nn.Linear(map_size//2, output_size, bias=True)

    def forward(self, x, hidden):
        # map_, _, hidden = self.rnn(x, hidden)
        map_, hidden = self.rnn(x, hidden)
        if self.detached:
            map_to_pass_on = map_.detach()
        else:
            map_to_pass_on = map_
        map_to_pass_on = torch.tanh(torch.relu(map_to_pass_on))
        # number = self.readout(torch.sigmoid(map_to_pass_on))
        # intermediate = torch.relu(self.readout1(map_to_pass_on))
        number = torch.relu(self.readout1(map_to_pass_on))
        number = self.readout2(number)
        # number = torch.relu(self.fc(map))
        return number, map_, hidden


class TwoStepModel(nn.Module):
    def __init__(self, input_size, hidden_size, map_size, output_size, **kwargs):
        super().__init__()
        act = kwargs['act'] if 'act' in kwargs.keys() else None
        eye_weight = kwargs['eye_weight'] if 'eye_weight' in kwargs.keys() else False
        detached = kwargs['detached'] if 'detached' in kwargs.keys() else False
        dropout = kwargs['dropout'] if 'dropout' in kwargs.keys() else 0.0
        self.hidden_size = hidden_size
        self.map_size = map_size
        self.out_size = output_size
        self.detached = detached
        self.rnn = RNN(input_size, hidden_size, map_size, act, eye_weight)
        self.initHidden = self.rnn.initHidden
        # self.rnn = tanhRNN(input_size, hidden_size, map_size)
        self.drop_layer = nn.Dropout(p=dropout)
        self.readout = nn.Linear(map_size, output_size, bias=True)

    def forward(self, x, hidden):
        map_, hidden = self.rnn(x, hidden)
        map_ = self.drop_layer(map_)
        if self.detached:
            map_to_pass_on = map_.detach()
        else:
            map_to_pass_on = map_
        # If this model is mapping rather than counting, nonlineartities get applied later
        if self.map_size != self.out_size: # counting
            map_to_pass_on = torch.tanh(torch.relu(map_to_pass_on))
        else: # mapping
            map_to_pass_on = self.drop_layer(map_to_pass_on)

        number = self.readout(map_to_pass_on)
        # number = torch.relu(self.fc(map))
        return number, map_, hidden


class ConvReadoutMapNet(nn.Module):
    def __init__(self, input_size, hidden_size, map_size, output_size, **kwargs):
        super().__init__()
        act = kwargs['act'] if 'act' in kwargs.keys() else None
        eye_weight = kwargs['eye_weight'] if 'eye_weight' in kwargs.keys() else False
        detached = kwargs['detached'] if 'detached' in kwargs.keys() else False
        dropout = kwargs['dropout'] if 'dropout' in kwargs.keys() else 0.0
        self.hidden_size = hidden_size
        self.map_size = map_size
        self.width = int(np.sqrt(map_size))
        self.out_size = output_size
        self.detached = detached
        self.rnn = RNN(input_size, hidden_size, map_size, act, eye_weight)
        self.initHidden = self.rnn.initHidden
        self.readout = ConvNet(self.width, self.width, self.out_size, dropout)

    def forward(self, x, hidden):
        batch_size = x.shape[0]
        map_, hidden = self.rnn(x, hidden)
        # map_ = self.drop_layer(map_)
        if self.detached:
            map_to_pass_on = map_.detach()
        else:
            map_to_pass_on = map_
        # If this model is mapping rather than counting, nonlineartities get applied later
        if self.map_size != self.out_size: # counting
            map_to_pass_on = torch.tanh(torch.relu(map_to_pass_on))
        map_to_pass_on = map_to_pass_on.view((-1, 1, self.width, self.width))
        number = self.readout(map_to_pass_on)
        # number = torch.relu(self.fc(map))
        return number, map_, hidden

class ConvNet(nn.Module):
    def __init__(self, width, height, output_size, dropout):
        super().__init__()
        self.kernel1_size = 5
        self.cnn1_nchannels_out = 6
        self.poolsize = 2
        self.kernel2_size = 6
        self.cnn2_nchannels_out = 12
        self.LReLU = nn.LeakyReLU(0.1)

        # Default initialization is init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # (Also assumes leaky Relu for gain)
        # which is appropriate for all layers here
        self.conv1 = nn.Conv2d(1, self.cnn1_nchannels_out, self.kernel1_size)    # (NChannels_in, NChannels_out, kernelsize)
        # torch.nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        self.pool = nn.MaxPool2d(self.poolsize, self.poolsize)     # kernel height, kernel width
        self.conv2 = nn.Conv2d(self.cnn1_nchannels_out, self.cnn2_nchannels_out, self.kernel2_size)   # (NChannels_in, NChannels_out, kernelsize)
        # torch.nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        # track the size of the cnn transformations
        self.cnn2_width_out = ((width - self.kernel1_size+1) - self.kernel2_size + 1) // self.poolsize
        self.cnn2_height_out = ((height - self.kernel1_size+1) - self.kernel2_size + 1) // self.poolsize

        # pass through FC layers
        self.fc1 = nn.Linear(int(self.cnn2_nchannels_out * self.cnn2_width_out * self.cnn2_height_out), 120)  # size input, size output

        self.fc2 = nn.Linear(120, output_size)

        # Dropout
        self.drop_layer = nn.Dropout(p=dropout)  # 20% chance of each neuron being dropped out / zeroed during each forward pass

    def forward(self, x):
        x = self.LReLU(self.conv1(x))
        x = self.pool(self.LReLU(self.conv2(x)))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3]) # this reshapes the tensor to be a 1D vector, from whatever the final convolutional layer output
        # add dropout before fully connected layers, widest part of network
        x = self.drop_layer(x)
        x = self.LReLU(self.fc1(x))
        x = self.fc2(x)
        return x

# class TwoStepModel_hist(nn.Module):
    # def __init__(self, input_size, hidden_size, map_size, output_size):
    #     super(TwoStepModel_hist, self).__init__()
    #     fc_size = 64
    #     self.rnn = linearRNN_hist(input_size, hidden_size, map_size)
    #     # self.rnn = tanhRNN(input_size, hidden_size, map_size)
    #     self.fc1 = nn.Linear(map_size, fc_size)
    #     self.fc2 = nn.Linear(fc_size, output_size)

    # def forward(self, x, hidden):
    #     outs, hidden, hist = self.rnn(x, hidden)
    #     fc_layer = torch.relu(self.fc1(hist))
    #     number = self.fc2(fc_layer)
    #     # number = torch.relu(self.fc(map))
    #     return outs, hidden, number

class TwoStepModel_weightshare(nn.Module):
    def __init__(self, hidden_size, map_size, output_size, init, act=None, eye_weight=False):
        super(TwoStepModel_weightshare, self).__init__()
        # self.saved_model = torch.load('models/two_step_model_with_maploss_bias.pt')
        self.hidden_size = hidden_size
        self.map_size = map_size
        self.out_size = output_size
        self.rnn = RNN(map_size, hidden_size, map_size, act, eye_weight)
        # self.rnn = tanhRNN(input_size, hidden_size, map_size)

        self.readout = Readout(map_size, output_size, init)
        # self.fc = nn.Linear(map_size, output_size)
        # self.fc.weight = self.saved_model.fc.weight
        # self.fc.weight.requires_grad = False
        # self.fc.bias = self.saved_model.fc.bias
        # self.fc.bias.requires_grad = False

        # hardcode_weight = torch.linspace(-0.1, 0.1, steps=7, dtype=torch.float)
        # hardcode_bias = torch.linspace(0.3, -0.3, steps=7, dtype=torch.float)
        # with torch.no_grad():
        #     for i in range(output_size):
        #         self.fc.weight[i, :] = hardcode_weight[i]
        #         self.fc.bias[i] =  hardcode_bias[i]

    def forward(self, x, hidden):
        map_, hidden = self.rnn(x, hidden)
        # number = self.fc(torch.sigmoid(map_))
        number = self.readout(torch.sigmoid(map_))
        # number = torch.relu(self.fc(map))
        return number, map_, hidden


class TwoStepModel_weightshare_detached(TwoStepModel_weightshare):

    def forward(self, x, hidden):
        map_, hidden = self.rnn(x, hidden)
        # number = self.fc(torch.sigmoid(map_))
        map_detached = map_.detach()
        number = self.readout(torch.sigmoid(map_detached))
        # number = torch.relu(self.fc(map))
        return number, map_, hidden

    def train_rnn(self):
        for param in self.readout.parameters():
            param.requires_grad = False
        for param in self.rnn.parameters():
            param.requires_grad = True

    def train_readout(self):
        for param in self.readout.parameters():
            param.requires_grad = True
        for param in self.rnn.parameters():
            param.requires_grad = False

class Readout(nn.Module):
    def __init__(self, map_size, out_size, init):
        super(Readout, self).__init__()
        self.map_size = map_size
        self.out_size = out_size
        weight_bound = 7
        bias_bound = 4 * weight_bound
        # self.weight = torch.nn.Parameter(torch.empty((out_size,)))
        # self.bias = torch.nn.Parameter(torch.empty((out_size,)))
        if init:
            self.weight = torch.nn.Parameter(torch.linspace(-weight_bound, weight_bound, self.out_size))
            self.bias = torch.nn.Parameter(torch.linspace(bias_bound, -bias_bound, self.out_size))
        else:
            self.weight = torch.nn.Parameter(torch.empty((out_size,)))
            self.bias = torch.nn.Parameter(torch.empty((out_size,)))
            nn.init.uniform_(self.weight, -1, 1)
            nn.init.uniform_(self.bias, -1, 1)

        # self.weight_mat = self.weight.expand(map_size, out_size)
        # self.weight_mat = self.weight.repeat(map_size, out_size)
        # self.init_params()

    # def init_params(self):
    #     weight_bound = 1
    #     bias_bound = 4 * weight_bound
    #     with torch.no_grad():
    #         self.weight = torch.nn.Parameter(torch.linspace(-weight_bound, weight_bound, self.out_size))
    #         self.bias = torch.nn.Parameter(torch.linspace(bias_bound, -bias_bound, self.out_size))
        # nn.init.uniform_(self.weight, -weight_bound, weight_bound)
        # if self.bias is not None:
        #     nn.init.uniform_(self.bias, -bias_bound, bias_bound)

    def forward(self, map):
        # map = torch.sigmoid(map)
        # out = torch.matmul(map, self.weight_mat) + self.bias
        weight_mat = self.weight.repeat(self.map_size, 1)
        # out = torch.addmm(self.bias, map, self.weight_mat)
        out = torch.addmm(self.bias, map, weight_mat)
        return out


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, act=None, eye_weight=False, dropout=0):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.out_size = output_size
        self.hidden_size = hidden_size
        self.eye_weight = eye_weight
        if act == 'tanh':
            self.act_fun = torch.tanh
            gain = 5/3
        elif act == 'relu':
            self.act_fun = torch.relu
            gain = np.sqrt(2)
        elif act == 'sig':
            self.act_fun = nn.Sigmoid()
            gain = 1
        else:
            self.act_fun = None
            gain = 1

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.drop_layer = nn.Dropout(p=dropout)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.init_params(gain)

                # self.i2o.weight[:, :input_size] = (torch.eye(input_size) * 1.05) - 0.05
                # self.i2h.weight[:, input_size:] = (torch.eye(hidden_size) * 1.05) - 0.05
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        hidden = self.drop_layer(hidden)
        if self.act_fun is not None:
            hidden = self.act_fun(hidden)
        output = self.i2o(combined)
        # output = self.softmax(output)
        return output, hidden

    def init_params(self, gain):
        if self.act_fun == 'relu':
            nn.init.kaiming_uniform_(self.i2h.weight, a=math.sqrt(5), nonlinearity='relu')
            nn.init.kaiming_uniform_(self.i2o.weight, a=math.sqrt(5), nonlinearity='relu')
        elif self.act_fun == 'lrelu':
            nn.init.kaiming_uniform_(self.i2h.weight, a=math.sqrt(5), nonlinearity='leaky_relu')
            nn.init.kaiming_uniform_(self.i2o.weight, a=math.sqrt(5), nonlinearity='leaky_relu')
        elif self.act_fun is None:
            nn.init.kaiming_uniform_(self.i2h.weight, a=math.sqrt(5), nonlinearity='linear')
            nn.init.kaiming_uniform_(self.i2o.weight, a=math.sqrt(5), nonlinearity='linear')
        else:
            nn.init.xavier_uniform_(self.i2h.weight, gain=gain)
            nn.init.xavier_uniform_(self.i2o.weight, gain=gain)
        if self.eye_weight:
            with torch.no_grad():
                nn.init.eye_(self.i2o.weight[:, :self.input_size])
                nn.init.eye_(self.i2h.weight[:, self.input_size:])

    def initHidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)

class linearRNN_hist(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(linearRNN_hist, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o_1 = nn.Linear(input_size + hidden_size, output_size)
        self.i2o_2 = nn.Linear(input_size + hidden_size, output_size)
        self.i2o_3 = nn.Linear(input_size + hidden_size, output_size)
        self.i2o_4 = nn.Linear(input_size + hidden_size, output_size)
        self.i2o_5 = nn.Linear(input_size + hidden_size, output_size)
        self.i2o_6 = nn.Linear(input_size + hidden_size, output_size)
        self.i2o_7 = nn.Linear(input_size + hidden_size, output_size)
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        out1 = self.i2o_1(combined)
        out2 = self.i2o_2(combined)
        out3 = self.i2o_3(combined)
        out4 = self.i2o_4(combined)
        out5 = self.i2o_5(combined)
        out6 = self.i2o_6(combined)
        out7 = self.i2o_7(combined)
        outs = [out1, out2, out3, out4, out5, out6, out7]

        hist = torch.stack([torch.argmax(out, dim=1) for out in outs]).T.float()

        # output = self.softmax(output)
        return outs, hidden, hist

    def initHidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)

class NumberAsMapSum(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.rnn = RNN(input_size, hidden_size, input_size)

    def forward(self, input, hidden):
        map_, hidden = self.rnn(input, hidden)
        number = torch.sum(torch.sigmoid(map_), axis=1)
        # number_oh = F.one_hot(torch.relu(number).long(), self.input_size)

        return number, hidden, map_

def train_sum_model(model, optimizer, config):
    n_epochs = config.n_epochs
    device = config.device
    use_loss = config.use_loss
    model_version = config.model_version
    print('Training number-as-sum model..')
    model.train()

    print('Using MSE loss for number objective.')
    criterion = nn.MSELoss()

    batch_size = 64
    seq_len = 7
    n_classes = 7
    filename = f'results/number-as-sum_results_mapsz{model.input_size}_loss-{use_loss}.npz'
    # data, target, _, _ = get_data(seq_len, n_classes, device)
    data, target, map_, _, _ = get_data(seq_len, n_classes, model.input_size, device)
    nex = data.shape[0]
    positive = (data.sum(axis=1) > 0).sum(axis=0)
    negative = -positive + nex
    pos_weight = torch.true_divide(negative, positive)
    # Remove infs
    to_replace = torch.tensor(nex).float().to(device)
    pos_weight = torch.where(torch.isinf(pos_weight), to_replace, pos_weight)
    criterion_map = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # To store learning curves
    numb_losses = np.zeros((n_epochs,))
    numb_accs = np.zeros((n_epochs,))
    map_losses = np.zeros((n_epochs,))
    map_accs = np.zeros((n_epochs, model.input_size))
    map_auc = np.zeros((n_epochs,))

    dataset = TensorDataset(data, target, map_)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for ep in range(n_epochs):
        m_correct = np.zeros(model.input_size,)
        n_correct = 0
        auc = 0
        train_loss = 0
        map_train_loss = 0
        for batch_idx, (inputs, label, map_label) in enumerate(loader):
            input_dim = inputs.shape[0]
            hidden = model.rnn.initHidden(input_dim)
            hidden = hidden.to(device)
            model.zero_grad()
            for i in range(seq_len):
                number, hidden, map_pred = model(inputs[:, i, :], hidden)
            number_loss = criterion(number, label.float())

            map_loss = criterion_map(map_pred, map_label)
            if 'number' in use_loss:
                loss = number_loss
            if 'map' in use_loss:
                loss = map_loss
            if 'both' in use_loss:
                loss = number_loss + map_loss

            loss.backward()
            optimizer.step()
            hidden.detach_()

            # Evaluate performance
            with torch.no_grad():
                train_loss += loss.item()
                map_train_loss += map_loss.item()
                numb_local = number.round()
                label_local = label
                map_local = map_pred
                map_label_local = map_label
                sigout = torch.sigmoid(map_local).round()
                m_correct += sum(sigout == map_label_local).cpu().numpy()
                map_label_flat = map_label_local.cpu().numpy().flatten()
                auc += roc_auc_score(map_label_flat, sigout.cpu().flatten())
                f1 += f1_score(map_label_flat, sigout.cpu().flatten())
                n_correct += numb_local.eq(label_local.view_as(numb_local)).sum().item()

        train_loss /= batch_idx + 1
        map_train_loss /= batch_idx + 1

        accs = 100. * (m_correct / len(dataset))
        map_accs[ep, :] = accs
        auc /= batch_idx+1
        f1 /= batch_idx+1

        numb_acc = 100. * (n_correct / len(dataset))
        numb_accs[ep] = numb_acc

        pct_done = round(100. * (ep / n_epochs))
        map_losses[ep] = map_train_loss
        numb_losses[ep] = train_loss
        map_acc = accs.mean()
        map_auc[ep] = auc

        if not ep % 10:
            # print(f'Progress {pct_done}% trained: \t Loss (Map/Numb): {map_loss.item():.6}/{number_loss.item():.6}, \t Accuracy: {accs[0]:.3}% {accs[1]:.3}% {accs[2]:.3}% {accs[3]:.3}% {accs[4]:.3}% {accs[5]:.3}% {accs[6]:.3}% \t Number {numb_acc:.3}% \t Mean Map Acc {accs.mean():.3}%')
            print(f'Epoch {ep}, Progress {pct_done}% trained: \t Loss (Map/Numb): {map_train_loss:.6}/{train_loss:.6}, \t Accuracy: \t Number {numb_acc:.3}% \t Map {map_acc:.3}% \t Map AUC: {auc:.3}')
            np.savez(filename, numb_accs=numb_accs, map_accs=map_accs,
                     numb_losses=numb_losses, map_losses=map_losses, map_auc=map_auc)

    print(f'Final performance: \t Loss (Map/Numb): {map_train_loss:.6}/{train_loss:.6}, \t Accuracy: \t Number {numb_acc:.3}% \t Map {map_acc:.3}%')
    np.savez(filename, numb_accs=numb_accs, map_accs=map_accs,
             numb_losses=numb_losses, map_losses=map_losses, map_auc=map_auc)

def train_rnn_nocount(model, optimizer, n_epochs, device):
    print('Linear RNN...')
    rnn = model
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()
    batch_size = 64
    seq_len = 7
    n_classes = 7
    data, target, target_nocount, _ = get_data(seq_len, n_classes, device)
    for ep in range(n_epochs):
        correct = np.zeros(7,)
        # shuffle the sequence order on each epoch
        for i, row in enumerate(data):
            data[i, :, :] = row[torch.randperm(seq_len), :]
        dataset = TensorDataset(data, target_nocount)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for batch_idx, (inputs, label) in enumerate(loader):
            input_dim = inputs.shape[0]
            hidden = rnn.initHidden(input_dim)
            rnn.zero_grad()

            for i in range(seq_len):
                output, hidden = rnn(inputs[:, i, :], hidden)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            out_local = output
            # Evaluate performance
            sigout = torch.sigmoid(out_local).round()
            correct += sum(sigout == label).cpu().numpy()
        accs = 100. * (correct / len(dataset))
        pct_done = round(100. * (ep / n_epochs))
        print(f'Progress {pct_done}% trained: \t Loss: {loss.item():.6}, \t Accuracy: {accs[0]:.3}% {accs[1]:.3}% {accs[2]:.3}% {accs[3]:.3}% {accs[4]:.3}% {accs[5]:.3}% {accs[6]:.3}%')


def train_rnn_nocount_diffs(model, optimizer, n_epochs, device):
    print('Linear RNN...')
    rnn = model
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()
    batch_size = 64
    seq_len = 7
    n_classes = 7
    data, target, target_nocount, _ = get_data(seq_len, n_classes, device)
    for ep in range(n_epochs):
        correct = np.zeros(7,)
        # shuffle the sequence order on each epoch
        for i, row in enumerate(data):
            data[i, :, :] = row[torch.randperm(seq_len), :]
        dataset = TensorDataset(data, target_nocount)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for batch_idx, (inputs, label) in enumerate(loader):
            input_dim = inputs.shape[0]
            hidden = rnn.initHidden(input_dim)
            rnn.zero_grad()
            prev_input = torch.zeros((inputs.shape[0], 1), dtype=torch.long)
            for i in range(seq_len):
                unonehot = inputs[:, i, :].nonzero(as_tuple=False)[:, 1].unsqueeze(1)
                this_input = torch.true_divide(unonehot, 6) - prev_input
                output, hidden = rnn(this_input, hidden)
                prev_input = this_input
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            out_local = output
            # Evaluate performance
            sigout = torch.sigmoid(out_local).round()
            correct += sum(sigout == label).cpu().numpy()
        accs = 100. * (correct / len(dataset))
        pct_done = round(100. * (ep / n_epochs))
        print(f'Progress {pct_done}% trained: \t Loss: {loss.item():.6}, \t Accuracy: {accs[0]:.3}% {accs[1]:.3}% {accs[2]:.3}% {accs[3]:.3}% {accs[4]:.3}% {accs[5]:.3}% {accs[6]:.3}%')


def train_rnn(model, optimizer, config):
    n_epochs = config.n_epochs
    device = config.device
    use_loss = config.use_loss
    model_version = config.model_version
    preglimpsed = config.preglimpsed
    eye_weight = config.eye_weight

    print('One-step RNN on unique task...')
    rnn = model
    rnn.train()

    batch_size = 64
    seq_len = 7
    n_classes = 7
    nonlinearity = 'tanh_' if rnn.act_fun is not None else ''
    sched = '_sched' if config.use_schedule else ''
    filename = f'results/{model_version}_{nonlinearity}results_mapsz{rnn.input_size}_loss-{use_loss}{sched}'
    if preglimpsed is not None:
        filename += '_' + preglimpsed
    if eye_weight:
        filename += '_eye'
    print(f'Results will be saved in {filename}')
    # data, target, _, _ = get_data(seq_len, n_classes, device)
    data, target, map_, _, _  = get_data(seq_len, rnn.out_size, rnn.input_size, device, preglimpsed)
    nex, seq_len = data.shape[0], data.shape[1]
    print(f'Sequence length is {seq_len}')
    if use_loss == 'map':
        target = map_
        positive = map_.sum(axis=0).float()
        negative = -positive + nex
        pos_weight = torch.true_divide(negative, positive)
        # Remove infs
        to_replace = torch.tensor(nex).float().to(device)
        pos_weight = torch.where(torch.isinf(pos_weight), to_replace, pos_weight)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.CrossEntropyLoss()
    # verify_dataset(data, target, seq_len)
    numb_losses = np.zeros((n_epochs,))
    numb_accs = np.zeros((n_epochs,))
    # np.save('toy_data_shuffled.npy', data.numpy())
    # np.save('toy_target_shuffled.npy', target.numpy())
    # exit()
    # data = torch.from_numpy(np.load('../relational_saccades/all_data_ds5_randseq.npy'))
    # target = torch.from_numpy(np.load('../relational_saccades/all_target_ds5_randseq.npy'))
    # target = target.long()
    dataset = TensorDataset(data, target)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for ep in range(n_epochs):
        correct = 0
        train_loss = 0
        # for i, row in enumerate(data):
        # shuffle the sequence order on each epoch
        #     data[i, :, :] = row[torch.randperm(seq_len), :]

        for batch_idx, (inputs, label) in enumerate(loader):
            input_dim = inputs.shape[0]
            hidden = rnn.initHidden(input_dim)
            hidden = hidden.to(device)
            rnn.zero_grad()
            for i in range(seq_len):
                output, hidden = rnn(inputs[:, i, :], hidden)
            loss = criterion(output, label)
            label_local = label
            loss.backward()
            optimizer.step()
            hidden.detach_()

            with torch.no_grad():
                out_local = output
                # Evaluate performance
                if use_loss == 'map':
                    sigout = torch.sigmoid(out_local).round()
                    # correct += roc_auc_score(label_local.cpu().numpy().flatten(), sigout.cpu().flatten())
                    correct += f1_score(label_local.cpu().numpy().flatten(), sigout.cpu().flatten())
                else:
                    pred = out_local.argmax(dim=1, keepdim=True)
                    correct += pred.eq(label_local.view_as(pred)).sum().item()
                train_loss += loss.item()
        train_loss /= batch_idx + 1
        numb_losses[ep] = train_loss
        if use_loss == 'map':
            acc = 100. * (correct / (batch_idx + 1))
        else:
            acc = 100. * (correct / len(dataset))
        numb_accs[ep] = acc
        if not ep % 5:
            pct_done = round(100. * (ep / n_epochs))
            print(f'Epoch {ep}, Progress {pct_done}% trained: \t Loss: {train_loss:.6}, Accuracy or F1: {acc:.6}% ({correct}/{len(dataset)})')
            np.savez(filename, numb_accs=numb_accs, numb_losses=numb_losses)
    np.savez(filename, numb_accs=numb_accs, numb_losses=numb_losses)
    torch.save(model.state_dict(), f'models/{filename}.pt')

def train_two_step_model(model, optimizer, config, scheduler=None):
    if 'detached' in config.use_loss:
        assert isinstance(optimizer, list)
        opt_rnn, opt_readout = optimizer
        scheduler_rnn, scheduler_readout = scheduler

    print('Two step model...')
    n_epochs = config.n_epochs
    device = config.device
    differential = config.differential
    use_loss = config.use_loss
    model_version = config.model_version
    control = config.control
    preglimpsed = config.preglimpsed
    eye_weight = config.eye_weight
    rnn = model.rnn
    drop = config.dropout
    clip_grad_norm = 1

    # Synthesize or load the data
    batch_size = 64
    width = int(np.sqrt(model.map_size))
    preglimpsed_train = preglimpsed + '_train' if preglimpsed is not None else None
    data, num, _, map_, _, _ = get_min_data(preglimpsed_train, config.train_on, device)
    seq_len = data.shape[1]
    # data = get_input(gaze, pix)

    # Calculate the weight per location to trade off precision and recall
    # As map_size increases, negative examples far outweight positive ones
    # Hoping this may also help with numerical stability
    # positive = (data.sum(axis=1) > 0).sum(axis=0)
    # nex = data.shape[0]
    # positive = map_.sum(axis=0).float()
    # negative = -positive + data.shape[0]
    # pos_weight = torch.true_divide(negative, positive)
    # # Remove infs
    # to_replace = torch.tensor(nex).float().to(device)
    positive = 4.5  # average num objects
    negative = config.map_size - positive
    pos_weight = torch.true_divide(negative, positive)
    pos_weight = torch.ones_like(map_[0, :]) * pos_weight
    # pos_weight = torch.where(torch.isinf(pos_weight), to_replace, pos_weight)
    map_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion = nn.CrossEntropyLoss()

    trainset = TensorDataset(data, map_, num)
    preglimpsed_val = preglimpsed + '_valid' if preglimpsed is not None else None
    data, num, _, map_, _, _ = get_min_data(preglimpsed_val, config.train_on, device)
    validset = TensorDataset(data, map_, num)
    preglimpsed_test = preglimpsed + '_test0' if preglimpsed is not None else None
    data, num, _, map_, _, _ = get_min_data(preglimpsed_test, config.train_on, device)
    testset = TensorDataset(data, map_, num)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(validset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True)
    # with torch.no_grad():
    #     verify_dataset(data, num, seq_len)

    # To store learning curves
    optimised_losses = np.zeros((n_epochs,))
    numb_losses = np.zeros((n_epochs,))
    numb_accs = np.zeros((n_epochs,))
    map_losses = np.zeros((n_epochs,))
    map_auc = np.zeros((n_epochs,))
    map_f1 = np.zeros((n_epochs,))
    valid_optimised_losses = np.zeros((n_epochs,))
    valid_numb_losses = np.zeros((n_epochs,))
    valid_numb_accs = np.zeros((n_epochs,))
    valid_map_losses = np.zeros((n_epochs,))
    valid_map_auc = np.zeros((n_epochs,))
    valid_map_f1 = np.zeros((n_epochs,))
    test_optimised_losses = np.zeros((n_epochs,))
    test_numb_losses = np.zeros((n_epochs,))
    test_numb_accs = np.zeros((n_epochs,))
    test_map_losses = np.zeros((n_epochs,))
    test_map_auc = np.zeros((n_epochs,))
    test_map_f1 = np.zeros((n_epochs,))

    # Where to save the results
    nonlinearity = config.rnn_act + '_' if config.rnn_act is not None else ''
    sched = '_sched' if config.use_schedule else ''
    pretrained = '-pretrained' if config.pretrained_map else ''
    if control:
        filename = f'two_step_results_{control}'
    else:
        if use_loss == 'map_then_both':
            filename = f'{model_version}{pretrained}_{nonlinearity}results_mapsz{model.map_size}_loss-{use_loss}-hardthresh{sched}_lr{config.lr}_wd{config.wd}_dr{drop}_tanhrelu_rand_trainon-{config.train_on}'
        else:
            filename = f'{model_version}{pretrained}_{nonlinearity}results_mapsz{model.map_size}_loss-{use_loss}{sched}_lr{config.lr}_wd{config.wd}_dr{drop}_tanhrelu_rand_trainon-{config.train_on}'
    if preglimpsed is not None:
        filename += '_' + preglimpsed
    if eye_weight:
        filename += '_eye'

    # if control == 'non_spatial':
    #     dataset = TensorDataset(data, map_control, num)
    # elif control == 'hist':
    #     dataset = TensorDataset(data, hist, num)
    # else:
    #     dataset = TensorDataset(data, map_, num)

    print(f'**Training model with map size of {model.map_size}, hidden layer size {model.hidden_size}, use_loss={use_loss}')

    def train(loader, ep, add_number_loss):
        model.train()
        n_correct = 0
        auc = 0
        f1 = 0
        train_loss = 0
        map_train_loss = 0
        number_train_loss = 0
        for batch_idx, (inputs, map_label, numb_label) in enumerate(loader):
            model.zero_grad()
            input_dim = inputs.shape[0]

            for i, row in enumerate(inputs):
            # shuffle the sequence order
                inputs[i, :, :] = row[torch.randperm(seq_len), :]

            hidden = rnn.initHidden(input_dim)
            hidden = hidden.to(device)
            prev_input = torch.zeros((inputs.shape[0], 1), dtype=torch.long)

            # FORWARD PASS
            for i in range(seq_len):
                if differential:
                    unonehot = inputs[:, i, :].nonzero(as_tuple=False)[:, 1].unsqueeze(1)
                    this_input = torch.true_divide(unonehot, 6) - prev_input
                else:
                    this_input = inputs[:, i, :]
                number, map, hidden = model(this_input, hidden)
                prev_input = this_input

            if control == 'hist':
                map_loss = [criterion(map[j], map_label[:, j]) for j in range(len(map))]
                map_loss = sum(map_loss)/len(map_loss)
            else:
                map_loss = map_criterion(map, map_label)
            number_loss = criterion(number, numb_label)
            if use_loss == 'map':
                loss = map_loss
            elif 'number' in use_loss:
                loss = number_loss
            elif use_loss == 'both':
                loss = map_loss + number_loss
            elif use_loss == 'map_then_both':
                if ep > 0 and (map_f1[ep - 1] > 0.5 or ep > n_epochs / 2):
                    add_number_loss = True
                if add_number_loss:
                    loss = map_loss + number_loss
                else:
                    loss = map_loss

            if use_loss == 'map_then_both-detached':
                if ep > 0 and (map_f1[ep - 1] > 0.5 or ep > n_epochs / 2):
                    add_number_loss = True
                # model.train_rnn()
                map_loss.backward()
                nn.utils.clip_grad_norm_(model.rnn.parameters(), clip_grad_norm)
                opt_rnn.step()
                if add_number_loss:
                    # model.train_readout()
                    number_loss.backward(retain_graph=True)
                    nn.utils.clip_grad_norm_(model.readout.parameters(), clip_grad_norm)
                    opt_readout.step()
                    loss = map_loss + number_loss
                else:
                    loss = map_loss

            elif use_loss == 'both-detached':
                # model.train_readout()

                map_loss.backward(retain_graph=True)
                # model.train_rnn()
                # Verify that gradients are localized
                # for n, p in model.named_parameters():
                #     print(f'{n}:')
                #     if p.requires_grad and p.grad is not None:
                #         print(f'{p.grad.abs().mean()}')
                # import pdb; pdb.set_trace()
                number_loss.backward()
                # Gradient clipping
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                # Take a gradient step
                opt_readout.step()
                opt_rnn.step()
                # loss = map_loss + number_loss
                # model.train()
                loss = map_loss + number_loss
            elif use_loss == 'number-detached':
                number_loss.backward()
                # Gradient clipping
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                # Take a gradient step
                opt_readout.step()
            else:
                loss.backward()
                # print('must specify which loss to optimize')
                # Gradient clipping
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                # Take a gradient step
                optimizer.step()

            hidden.detach_()

            # Evaluate performance
            with torch.no_grad():
                train_loss += loss.item()
                number_train_loss += number_loss.item()
                map_train_loss += map_loss.item()
                numb_local = number
                map_local = map
                map_label_local = map_label
                if control == 'hist':
                    for j in range(len(map_local)):
                        pred = map_local[j].argmax(dim=1, keepdim=True)
                    # m_correct += sum(torch.isclose(map_local, map_label)).cpu().numpy()
                else:
                    sigout = torch.sigmoid(map_local).round()
                    map_label_flat = map_label_local.cpu().numpy().flatten()
                    auc += roc_auc_score(map_label_flat, sigout.cpu().flatten())
                    f1 += f1_score(map_label_flat, sigout.cpu().flatten())
                pred = numb_local.argmax(dim=1, keepdim=True)
                n_correct += pred.eq(numb_label.view_as(pred)).sum().item()

        if config.use_schedule:
            if 'detached' in use_loss:
                # decrease number loss twice as slow? not sure? basically guessing first 2000 for map next 2000 for map, don't want learning rate to be too small for number weights to learn
                if (add_number_loss or 'both' in use_loss) and ep % 2:
                    scheduler_readout.step()
                scheduler_rnn.step()
            else:
                scheduler.step()
        # Evaluate performance
        train_loss /= len(loader)
        map_train_loss /= len(loader)
        number_train_loss /= len(loader)
        auc /=len(loader)
        f1 /= len(loader)
        numb_acc = 100. * (n_correct / len(loader.dataset))

        if use_loss != 'number' and not ep % 10:
            # Make figure
            fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 10))
            im = axs[0, 0].imshow(inputs[-1, :, :config.map_size].sum(axis=0).detach().cpu().view(width, width), vmin=0)
            axs[0, 0].set_title('Input Gaze (unsequenced)')
            plt.colorbar(im, ax=axs[0,0], orientation='horizontal')

            im1 = axs[1, 0].imshow(map_local[-1, :].detach().cpu().view(width, width))
            axs[1, 0].set_title(f'Predicted Map (Predicted number: {pred[-1].item()}, Number acc: {numb_acc:.3})')
            plt.colorbar(im1, ax=axs[1,0], orientation='horizontal')
            # fig.colorbar(im1, orientation='horizontal')

            # im2 = axs[0, 1].imshow(torch.sigmoid(map_local[-1, :]).detach().cpu().view(width, width), vmin=0, vmax=1, cmap='bwr')
            # axs[0, 1].set_title(f'Sigmoid(Predicted Map) AUC={auc:.3} F1={f1:.3}')
            # plt.colorbar(im2, ax=axs[0,1], orientation='horizontal')

            im2 = axs[0, 1].imshow(torch.tanh(torch.relu(map_local[-1, :])).detach().cpu().view(width, width))
            axs[0, 1].set_title(f'Tanh(Relu(Predicted Map)) AUC={auc:.3} F1={f1:.3}')
            plt.colorbar(im2, ax=axs[0,1], orientation='horizontal')

            im3 = axs[1, 1].imshow(map_label_local[-1, :].detach().cpu().view(width, width), vmin=0, vmax=1)
            axs[1, 1].set_title(f'Actual Map (number={numb_label[-1].item()})')
            plt.colorbar(im3, ax=axs[1,1], orientation='horizontal')
            # plt.show()
            plt.savefig(f'figures/predicted_maps/{filename}_ep{ep}.png', bbox_inches='tight', dpi=200)
            plt.close()
        return train_loss, number_train_loss, map_train_loss, numb_acc, auc, f1, add_number_loss

    def test(loader, add_number_loss, **kwargs):
        which_set = kwargs['which_set'] if 'which_set' in kwargs.keys() else 'test'
        model.eval()
        n_correct = 0
        auc = 0
        f1 = 0
        train_loss = 0
        map_train_loss = 0
        number_train_loss = 0
        with torch.no_grad():
            for batch_idx, (inputs, map_label, numb_label) in enumerate(loader):
                input_dim = inputs.shape[0]
                hidden = rnn.initHidden(input_dim)
                hidden = hidden.to(device)
                prev_input = torch.zeros((inputs.shape[0], 1), dtype=torch.long)
                for i in range(seq_len):
                    if differential:
                        unonehot = inputs[:, i, :].nonzero(as_tuple=False)[:, 1].unsqueeze(1)
                        this_input = torch.true_divide(unonehot, 6) - prev_input
                    else:
                        this_input = inputs[:, i, :]
                    number, map, hidden = model(this_input, hidden)
                    prev_input = this_input

                if control == 'hist':
                    map_loss = [criterion(map[j], map_label[:, j]) for j in range(len(map))]
                    map_loss = sum(map_loss)/len(map_loss)
                else:
                    map_loss = map_criterion(map, map_label)
                number_loss = criterion(number, numb_label)
                if use_loss == 'map':
                    loss = map_loss
                elif 'number' in use_loss:
                    loss = number_loss
                elif use_loss == 'both':
                    loss = map_loss + number_loss
                elif use_loss == 'map_then_both' or use_loss == 'map_then_both-detached':
                    if ep > 0 and (map_auc[ep - 1] > 0.99 or ep > n_epochs / 2):
                        add_number_loss = True
                    if add_number_loss:
                        loss = map_loss + number_loss
                    else:
                        loss = map_loss
                elif use_loss == 'both-detached':

                    # model.train_readout()
                    # number_loss.backward(retain_graph=True)
                    # model.train_rnn()
                    # map_loss.backward()
                    loss = map_loss + number_loss
                    # model.train()

                else:
                    print('must specify which loss to optimize')

                # Evaluate performance
                train_loss += loss.item()
                number_train_loss += number_loss.item()
                map_train_loss += map_loss.item()
                numb_local = number
                map_local = map
                map_label_local = map_label
                sigout = torch.sigmoid(map_local).round()
                map_label_flat = map_label_local.cpu().numpy().flatten()
                auc += roc_auc_score(map_label_flat, sigout.cpu().flatten())
                f1 += f1_score(map_label_flat, sigout.cpu().flatten())
                pred = numb_local.argmax(dim=1, keepdim=True)
                n_correct += pred.eq(numb_label.view_as(pred)).sum().item()
            # Evaluate performance
            train_loss /= len(loader)
            map_train_loss /= len(loader)
            number_train_loss /= len(loader)
            auc /=len(loader)
            f1 /= len(loader)
            numb_acc = 100. * (n_correct / len(loader.dataset))

            if use_loss != 'number' and not ep % 10:
                # Make figure
                fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 10))
                im = axs[0, 0].imshow(inputs[-1, :, :config.map_size].sum(axis=0).detach().cpu().view(width, width), vmin=0)
                axs[0, 0].set_title('Input Gaze (unsequenced)')
                plt.colorbar(im, ax=axs[0,0], orientation='horizontal')

                im1 = axs[1, 0].imshow(map_local[-1, :].detach().cpu().view(width, width))
                axs[1, 0].set_title(f'Predicted Map (Predicted number: {pred[-1].item()}, Number acc: {numb_acc:.3})')
                plt.colorbar(im1, ax=axs[1,0], orientation='horizontal')
                # fig.colorbar(im1, orientation='horizontal')

                # im2 = axs[0, 1].imshow(torch.sigmoid(map_local[-1, :]).detach().cpu().view(width, width), vmin=0, vmax=1, cmap='bwr')
                # axs[0, 1].set_title(f'Sigmoid(Predicted Map) AUC={auc:.3} F1={f1:.3}')
                # plt.colorbar(im2, ax=axs[0,1], orientation='horizontal')

                im2 = axs[0, 1].imshow(torch.tanh(torch.relu(map_local[-1, :])).detach().cpu().view(width, width))
                axs[0, 1].set_title(f'Tanh(Relu(Predicted Map)) AUC={auc:.3} F1={f1:.3}')
                plt.colorbar(im2, ax=axs[0,1], orientation='horizontal')

                im3 = axs[1, 1].imshow(map_label_local[-1, :].detach().cpu().view(width, width), vmin=0, vmax=1)
                axs[1, 1].set_title(f'Actual Map (number={numb_label[-1].item()})')
                plt.colorbar(im3, ax=axs[1,1], orientation='horizontal')
                # plt.show()
                plt.savefig(f'figures/predicted_maps/{filename}_ep{ep}_{which_set}.png', bbox_inches='tight', dpi=200)
                plt.close()
        return train_loss, number_train_loss, map_train_loss, numb_acc, auc, f1

    add_number_loss = False
    for ep in range(n_epochs):
        if isinstance(scheduler, list):
            print(f'Epoch {ep}, Learning rates: {opt_rnn.param_groups[0]["lr"]}, {opt_readout.param_groups[0]["lr"]}')
        else:
            print(f'Epoch {ep}, Learning rate: {optimizer.param_groups[0]["lr"]}')

        train_loss, train_num_loss, train_map_loss, train_num_acc, train_auc, train_f1, add_number_loss = train(train_loader, ep, add_number_loss)
        val_loss, val_num_loss, val_map_loss, val_num_acc, val_auc, val_f1 = test(valid_loader, add_number_loss, which_set='valid')
        test_loss, test_num_loss, test_map_loss, test_num_acc, test_auc, test_f1 = test(test_loader, add_number_loss, which_set='test')

        optimised_losses[ep] = train_loss
        numb_losses[ep] = train_num_loss
        numb_accs[ep] = train_num_acc
        map_losses[ep] = train_map_loss
        map_auc[ep] = train_auc
        map_f1[ep] = train_f1
        valid_optimised_losses[ep] = val_loss
        valid_numb_losses[ep] = val_num_loss
        valid_numb_accs[ep] = val_num_acc
        valid_map_losses[ep] = val_map_loss
        valid_map_auc[ep] = val_auc
        valid_map_f1[ep] = val_f1
        test_optimised_losses[ep] = test_loss
        test_numb_losses[ep] = test_num_loss
        test_numb_accs[ep] = test_num_acc
        test_map_losses[ep] = test_map_loss
        test_map_auc[ep] = test_auc
        test_map_f1[ep] = test_f1

        pct_done = round(100. * (ep / n_epochs))
        if not ep % 5:
            # Print and save performance
            # print(f'Progress {pct_done}% trained: \t Loss (Map/Numb): {map_loss.item():.6}/{number_loss.item():.6}, \t Accuracy: {accs[0]:.3}% {accs[1]:.3}% {accs[2]:.3}% {accs[3]:.3}% {accs[4]:.3}% {accs[5]:.3}% {accs[6]:.3}% \t Number {numb_acc:.3}% \t Mean Map Acc {accs.mean():.3}%')
            print(f'Epoch {ep}, Progress {pct_done}% trained')
            print(f'Train Loss (Num/Map): {train_num_loss:.6}/{train_map_loss:.6}, \t Train Performance (Num/MapAUC/MapF1) {train_num_acc:.3}%/{train_auc:.3}/{train_f1:.3}')
            print(f'Valid Loss (Num/Map): {val_num_loss:.6}/{val_map_loss:.6}, \t Valid Performance (Num/MapAUC/MapF1) {val_num_acc:.3}%/{val_auc:.3}/{val_f1:.3}')
            print(f'Test Loss (Num/Map): {test_num_loss:.6}/{test_map_loss:.6}, \t Test Performance (Num/MapAUC/MapF1) {test_num_acc:.3}%/{test_auc:.3}/{test_f1:.3}')

            np.savez('results/'+filename,
                     train_loss=optimised_losses,
                     numb_loss=numb_losses,
                     numb_acc=numb_accs,
                     map_loss=map_losses,
                     map_auc=map_auc,
                     map_f1=map_f1,
                     valid_loss=valid_optimised_losses,
                     valid_numb_loss=valid_numb_losses,
                     valid_numb_acc=valid_numb_accs,
                     valid_map_loss=valid_map_losses,
                     valid_map_auc=valid_map_auc,
                     valid_map_f1=valid_map_f1,
                     test_loss=test_optimised_losses,
                     test_numb_loss=test_numb_losses,
                     test_numb_acc=test_numb_accs,
                     test_map_loss=test_map_losses,
                     test_map_auc=test_map_auc,
                     test_map_f1=test_map_f1)
            torch.save(model, f'models/{filename}.pt')
            # Plot performance
            kwargs_train = {'alpha': 0.8, 'color': 'blue'}
            kwargs_val = {'alpha': 0.8, 'color': 'cyan'}
            kwargs_test = {'alpha': 0.8, 'color': 'red'}
            row, col = 2, 3
            fig, ax = plt.subplots(row, col, figsize=(6*col, 6*row))
            plt.suptitle(f'{config.model_version}, nonlin={config.rnn_act}, loss={config.use_loss}, dr={config.dropout}% on intermediate, \n tanh(relu(map)), seq shuffled, pos_weight=130-4.5/4.5 \n {preglimpsed}',fontsize=20)
            ax[0,0].set_title('Number Loss')
            ax[0,0].plot(test_numb_losses[:ep+1], label='test number loss', **kwargs_test)
            ax[0,0].plot(valid_numb_losses[:ep+1], label='valid number loss', **kwargs_val)
            ax[0,0].plot(numb_losses[:ep+1], label='train number loss', **kwargs_train)
            ax[1,0].set_title('Map Loss')
            ax[1,0].plot(test_map_losses[:ep+1], label='test map loss', **kwargs_test)
            ax[1,0].plot(valid_map_losses[:ep+1], label='valid map loss', **kwargs_val)
            ax[1,0].plot(map_losses[:ep+1], label='train map loss', **kwargs_train)
            ax[0,1].set_title('Number Accuracy')
            ax[0,1].set_ylim([10, 100])
            ax[0,1].plot(test_numb_accs[:ep+1], label='test number acc', **kwargs_test)
            ax[0,1].plot(valid_numb_accs[:ep+1], label='valid number acc', **kwargs_val)
            ax[0,1].plot(numb_accs[:ep+1], label='train number acc', **kwargs_train)
            ax[1,1].set_title('Map AUC')
            ax[1,1].set_ylim([0.45, 1])
            ax[1,1].plot(test_map_auc[:ep+1], label='test map auc', **kwargs_test)
            ax[1,1].plot(valid_map_auc[:ep+1], label='valid map auc', **kwargs_val)
            ax[1,1].plot(map_auc[:ep+1], label='train map auc', **kwargs_train)
            ax[1,2].set_title('Map F1')
            ax[1,2].plot(test_map_f1[:ep+1], label='test map f1', **kwargs_test)
            ax[1,2].plot(valid_map_f1[:ep+1], label='valid map f1', **kwargs_val)
            ax[1,2].plot(map_f1[:ep+1], label='train map f1', **kwargs_train)
            for axes in ax.flatten():
                axes.legend()
                axes.grid(linestyle='--')
                axes.set_xlabel('Epochs')
            ax[0, 2].axis('off')
            plt.savefig(f'figures/{filename}_results.png', dpi=300)
            plt.close()

    print(f'Final performance:')
    print(f'Train Loss (Num/Map): {train_num_loss:.6}/{train_map_loss:.6}, \t Train Performance (Num/MapAUC/MapF1) {train_num_acc:.3}%/{train_auc:.3}/{train_f1:.3}')
    print(f'Valid Loss (Num/Map): {val_num_loss:.6}/{val_map_loss:.6}, \t Valid Performance (Num/MapAUC/MapF1) {val_num_acc:.3}%/{val_auc:.3}/{val_f1:.3}')
    print(f'Test Loss (Num/Map): {test_num_loss:.6}/{test_map_loss:.6}, \t Test Performance (Num/MapAUC/MapF1) {test_num_acc:.3}%/{test_auc:.3}/{test_f1:.3}')

    np.savez('results/'+filename,
             train_loss=optimised_losses,
             numb_loss=numb_losses,
             numb_acc=numb_accs,
             map_loss=map_losses,
             map_auc=map_auc,
             map_f1=map_f1,
             valid_loss=valid_optimised_losses,
             valid_numb_loss=valid_numb_losses,
             valid_numb_acc=valid_numb_accs,
             valid_map_loss=valid_map_losses,
             valid_map_auc=valid_map_auc,
             valid_map_f1=valid_map_f1,
             test_loss=test_optimised_losses,
             test_numb_loss=test_numb_losses,
             test_numb_acc=test_numb_accs,
             test_map_loss=test_map_losses,
             test_map_auc=test_map_auc,
             test_map_f1=test_map_f1)
    torch.save(model, f'models/{filename}.pt')


def train_content_gated_model(model, optimizer, config, scheduler=None):
    print('Content-gated model...')
    n_epochs = config.n_epochs
    device = config.device
    differential = config.differential
    use_loss = config.use_loss
    model_version = config.model_version
    control = config.control
    preglimpsed = config.preglimpsed
    eye_weight = config.eye_weight
    clip_grad_norm = 1

    # Load the data
    batch_size = 64
    seq_len = 11
    width = int(np.sqrt(model.dorsal.map_size))
    gaze, pix, num, shape, map_, min_num, min_shape = get_min_data(preglimpsed, device)
    test_set = 'min_trianglestar_2-7_varylum_130x130_2000_noresize'
    gaze_test, pix_test, num_test, shape_test, map_test, min_num_test, min_shape_test = get_min_data(test_set, device)
    seq_len = gaze.shape[1]
    nex = gaze.shape[0]
    # with torch.no_grad():
    #     verify_dataset(data, num, seq_len)

    # Calculate the weight per location to trade off precision and recall
    # As map_size increases, negative examples far outweight positive ones
    # Hoping this may also help with numerical stability
    # positive = (data.sum(axis=1) > 0).sum(axis=0)
    positive = map_.sum(axis=0).float()
    negative = -positive + nex
    pos_weight = torch.true_divide(negative, positive)
    # Remove infs
    to_replace = torch.tensor(nex).float().to(device)
    pos_weight = torch.where(torch.isinf(pos_weight), to_replace, pos_weight)
    map_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion = nn.CrossEntropyLoss()
    add_number_loss = False

    # To store learning curves
    num_losses = np.zeros((n_epochs,))
    num_accs = np.zeros((n_epochs,))
    min_num_losses = np.zeros((n_epochs,))
    min_num_accs = np.zeros((n_epochs,))
    min_shape_losses = np.zeros((n_epochs,))
    min_shape_accs = np.zeros((n_epochs,))
    map_losses = np.zeros((n_epochs,))
    map_auc = np.zeros((n_epochs,))
    map_f1 = np.zeros((n_epochs,))
    test_num_losses = np.zeros((n_epochs,))
    test_num_accs = np.zeros((n_epochs,))
    test_min_num_losses = np.zeros((n_epochs,))
    test_min_num_accs = np.zeros((n_epochs,))
    test_min_shape_losses = np.zeros((n_epochs,))
    test_min_shape_accs = np.zeros((n_epochs,))
    test_map_losses = np.zeros((n_epochs,))
    test_map_auc = np.zeros((n_epochs,))
    test_map_f1 = np.zeros((n_epochs,))

    # len(np.arange(144)[::5])**2
    # Where to save the results
    nonlinearity = 'tanh_' if model.dorsal.rnn.act_fun is not None else ''
    sched = '_sched' if config.use_schedule else ''
    pretrained = '-pretrained' if config.pretrained_map else ''

    filename = f'{model_version}{pretrained}_{nonlinearity}results_mapsz{model.map_size}_loss-{use_loss}{sched}'
    if preglimpsed is not None:
        filename += '_' + preglimpsed
    if eye_weight:
        filename += '_eye'
    print(f'Results will be saved in {filename}')

    dataset = TensorDataset(gaze, num, shape, map_, min_num, min_shape)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataset_test = TensorDataset(gaze_test, num_test, shape_test, map_test, min_num_test, min_shape_test)
    testloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
    print(f'**Training model with map size of {model.map_size}, hidden layer size {model.hidden_size}, use_loss={use_loss}')
    for ep in range(n_epochs):
        epoch_timer = Timer()
        print(f'Epoch {ep}, Learning rate: {optimizer.param_groups[0]["lr"]}')
        # m_correct = np.zeros(model.map_size,)
        num_correct = 0
        min_num_correct = 0
        min_shape_correct = 0
        auc = 0
        f1 = 0
        test_num_correct = 0
        test_min_num_correct = 0
        test_min_shape_correct = 0
        test_auc = 0
        test_f1 = 0

        train_loss = 0
        map_train_loss = 0
        number_train_loss = 0
        min_num_train_loss = 0
        min_shape_train_loss = 0
        test_loss = 0
        map_test_loss = 0
        number_test_loss = 0
        min_num_test_loss = 0
        min_shape_test_loss = 0
        # shuffle the sequence order on each epoch
        # for i, row in enumerate(data):
        #     data[i, :, :] = row[torch.randperm(seq_len), :]
        # Select which map target to use
        # for batch_idx, (inputs, map_label, numb_label) in enumerate(loader):

        # Training
        for batch_idx, (gaze, num_label, shape, map_label, min_num_label, min_shape_label) in enumerate(loader):
            model.zero_grad()
            current_batch_size = gaze.shape[0]
            hidden = torch.zeros((5, current_batch_size, model.hidden_size), device=device)
            hidden = hidden.to(device)

            for i in range(seq_len):
                input_i = gaze[:, i, :]
                shape_i = shape[:, i, :]
                min_shape, min_number, counts, maps, hidden = model(input_i, shape_i, hidden)
            n_, n_hrt, n_str, n_sqr, n_tri = counts
            map_, map_hrt, map_str, map_sqr, map_tri = maps

            map_loss = map_criterion(map_, map_label)
            number_loss = criterion(n_, num_label)
            min_num_loss = criterion(min_number, min_num_label)
            min_shape_loss = criterion(min_shape, min_shape_label)
            if use_loss == 'easy_then_hard':
                if ep < 500:
                    loss = map_loss + number_loss
                else:
                    loss = map_loss + number_loss + min_num_loss + min_shape_loss
            elif use_loss == 'no_map':
                loss = number_loss + min_num_loss + min_shape_loss
            elif use_loss == 'all':
                loss = map_loss + number_loss + min_num_loss + min_shape_loss
            # Calculate gradients
            loss.backward()
            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            # Take a gradient step
            optimizer.step()
            hidden.detach_()

            # Evaluate performance
            with torch.no_grad():
                train_loss += loss.item()
                number_train_loss += number_loss.item()
                map_train_loss += map_loss.item()
                min_num_train_loss += min_num_loss.item()
                min_shape_train_loss += min_shape_loss.item()

                sigout = torch.sigmoid(map_).round().cpu().flatten()
                map_label_flat = map_label.cpu().numpy().flatten()
                auc += roc_auc_score(map_label_flat, sigout)
                f1 += f1_score(map_label_flat, sigout)
                pred_n = n_.argmax(dim=1, keepdim=True)
                num_correct += pred_n.eq(num_label.view_as(pred_n)).sum().item()
                pred_min_num = min_number.argmax(dim=1, keepdim=True)
                min_num_correct += pred_min_num.eq(min_num_label.view_as(pred_min_num)).sum().item()
                pred_min_shape = min_shape.argmax(dim=1, keepdim=True)
                min_shape_correct += pred_min_shape.eq(min_shape_label.view_as(pred_min_shape)).sum().item()

        if config.use_schedule:
            scheduler.step()
        pct_done = round(100. * (ep / n_epochs))

        # normalize and store evaluation metrics
        train_loss /= batch_idx + 1
        map_train_loss /= batch_idx + 1
        number_train_loss /= batch_idx + 1
        min_num_train_loss /= batch_idx + 1
        min_shape_train_loss /= batch_idx + 1
        auc /= batch_idx+1
        f1 /= batch_idx+1
        num_acc = 100. * (num_correct / len(dataset))
        num_accs[ep] = num_acc
        min_num_acc = 100. * (min_num_correct / len(dataset))
        min_num_accs[ep] = min_num_acc
        min_shape_acc = 100. * (min_shape_correct / len(dataset))
        min_shape_accs[ep] = min_shape_acc
        map_losses[ep] = map_train_loss
        num_losses[ep] = number_train_loss
        min_num_losses[ep] = min_num_train_loss
        min_shape_losses[ep] = min_shape_train_loss
        map_auc[ep] = auc
        map_f1[ep] = f1

        if not ep % 2:
            # Print and save performance
            # print(f'Progress {pct_done}% trained: \t Loss (Map/Numb): {map_loss.item():.6}/{number_loss.item():.6}, \t Accuracy: {accs[0]:.3}% {accs[1]:.3}% {accs[2]:.3}% {accs[3]:.3}% {accs[4]:.3}% {accs[5]:.3}% {accs[6]:.3}% \t Number {numb_acc:.3}% \t Mean Map Acc {accs.mean():.3}%')
            print(f'Epoch {ep}, Progress {pct_done}% trained.')
            print(f'Train Loss (Num/Map/MinNum/MinShape): {number_train_loss:.6}/{map_train_loss:.6}/{min_num_train_loss:.6}/{min_shape_train_loss:.6}, \t Train Performance (Num/MapAUC/MapF1/MinNum/MinShape) {num_acc:.3}%/{auc:.3}/{f1:.3}/{min_num_acc:.3}%/{min_shape_acc:.3}%')

            if use_loss != 'number' and use_loss != 'no_map':
                with torch.no_grad():
                    # Make figure
                    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 10))
                    im = axs[0, 0].imshow(gaze[-1, :, :].sum(axis=0).detach().cpu().view(width, width), vmin=0, cmap='bwr')
                    axs[0, 0].set_title('Input Gaze (unsequenced)')
                    plt.colorbar(im, ax=axs[0,0], orientation='horizontal')

                    im1 = axs[1, 0].imshow(map_[-1, :].detach().cpu().view(width, width))
                    axs[1, 0].set_title(f'Predicted Map (Predicted number: {pred_n[-1].item()})')
                    plt.colorbar(im1, ax=axs[1,0], orientation='horizontal')
                    # fig.colorbar(im1, orientation='horizontal')

                    im2 = axs[0, 1].imshow(torch.sigmoid(map_[-1, :]).detach().cpu().view(width, width), vmin=0, vmax=1, cmap='bwr')
                    axs[0, 1].set_title(f'Sigmoid(Predicted Map) AUC={auc:.3} F1={f1:.3}')
                    plt.colorbar(im2, ax=axs[0,1], orientation='horizontal')

                    im3 = axs[1, 1].imshow(map_label[-1, :].detach().cpu().view(width, width), vmin=0, vmax=1, cmap='bwr')
                    axs[1, 1].set_title(f'Actual Map (number={num_label[-1].item()})')
                    plt.colorbar(im3, ax=axs[1,1], orientation='horizontal')
                    # plt.show()
                    plt.savefig(f'figures/predicted_maps/{filename}_ep{ep}.png', bbox_inches='tight', dpi=200)
                    plt.close()

        # Testing
        with torch.no_grad():
            model.eval()
            for batch_idx, (gaze, num_label, shape, map_label, min_num_label, min_shape_label) in enumerate(testloader):
                current_batch_size = gaze.shape[0]
                hidden = torch.zeros((5, current_batch_size, model.hidden_size))
                hidden = hidden.to(device)

                for i in range(seq_len):
                    input_i = gaze[:, i, :]
                    shape_i = shape[:, i, :]
                    min_shape, min_number, counts, maps, hidden = model(input_i, shape_i, hidden)
                n_, n_hrt, n_str, n_sqr, n_tri = counts
                map_, map_hrt, map_str, map_sqr, map_tri = maps

                map_loss = map_criterion(map_, map_label)
                number_loss = criterion(n_, num_label)
                min_num_loss = criterion(min_number, min_num_label)
                min_shape_loss = criterion(min_shape, min_shape_label)
                loss = map_loss + number_loss + min_num_loss + min_shape_loss

                test_loss += loss.item()
                number_test_loss += number_loss.item()
                map_test_loss += map_loss.item()
                min_num_test_loss += min_num_loss.item()
                min_shape_test_loss += min_shape_loss.item()

                sigout = torch.sigmoid(map_).round().cpu().flatten()
                map_label_flat = map_label.cpu().numpy().flatten()
                test_auc += roc_auc_score(map_label_flat, sigout)
                test_f1 += f1_score(map_label_flat, sigout)
                pred_n = n_.argmax(dim=1, keepdim=True)
                test_num_correct += pred_n.eq(num_label.view_as(pred_n)).sum().item()
                pred_min_num = min_number.argmax(dim=1, keepdim=True)
                test_min_num_correct += pred_min_num.eq(min_num_label.view_as(pred_min_num)).sum().item()
                pred_min_shape = min_shape.argmax(dim=1, keepdim=True)
                test_min_shape_correct += pred_min_shape.eq(min_shape_label.view_as(pred_min_shape)).sum().item()

        # normalize and store evaluation metrics

        test_loss /= batch_idx + 1
        map_test_loss /= batch_idx + 1
        number_test_loss /= batch_idx + 1
        min_num_test_loss /= batch_idx + 1
        min_shape_test_loss /= batch_idx + 1
        test_auc /= batch_idx+1
        test_f1 /= batch_idx+1

        test_num_acc = 100. * (test_num_correct / len(dataset_test))
        test_num_accs[ep] = test_num_acc
        test_min_num_acc = 100. * (test_min_num_correct / len(dataset_test))
        test_min_num_accs[ep] = test_min_num_acc
        test_min_shape_acc = 100. * (test_min_shape_correct / len(dataset_test))
        test_min_shape_accs[ep] = test_min_shape_acc

        test_map_losses[ep] = map_test_loss
        test_num_losses[ep] = number_test_loss
        test_min_num_losses[ep] = min_num_test_loss
        test_min_shape_losses[ep] = min_shape_test_loss
        test_map_auc[ep] = test_auc
        test_map_f1[ep] = test_f1

        if not ep % 2:
            # Print and save performance
            print(f'Test Loss (Num/Map/MinNum/MinShape): {number_test_loss:.6}/{map_test_loss:.6}/{min_num_test_loss:.6}/{min_shape_test_loss:.6}, \t Test Performance (Num/MapAUC/MapF1/MinNum/MinShape) {test_num_acc:.3}%/{test_auc:.3}/{test_f1:.3}/{test_min_num_acc:.3}%/{test_min_shape_acc:.3}%')
            # print(f'Progress {pct_done}% trained: \t Loss (Map/Numb): {map_loss.item():.6}/{number_loss.item():.6}, \t Accuracy: {accs[0]:.3}% {accs[1]:.3}% {accs[2]:.3}% {accs[3]:.3}% {accs[4]:.3}% {accs[5]:.3}% {accs[6]:.3}% \t Number {numb_acc:.3}% \t Mean Map Acc {accs.mean():.3}%')
            # print(f'Epoch {ep}, Progress {pct_done}% trained: \t Loss (Map/Numb): {map_train_loss:.6}/{number_train_loss:.6}, \t Performance (Num/MapAUC/MapF1/MinNum/MinShape) {num_acc:.3}%/{auc:.3}/{f1:.3}/{min_num_acc:.3}/{min_shape_acc:.3}')
            np.savez('results/'+filename, numb_accs=num_accs, map_auc=map_auc, map_f1=map_f1,
                     min_num_accs=min_num_accs, min_shape_accs=min_shape_accs,
                     numb_losses=num_losses, map_losses=map_losses, min_num_losses=min_num_losses,
                     min_shape_losses=min_shape_losses, test_numb_accs=test_num_accs, test_map_auc=test_map_auc, test_map_f1=test_map_f1,
                     test_min_num_accs=test_min_num_accs, test_min_shape_accs=test_min_shape_accs,
                     test_numb_losses=test_num_losses, test_map_losses=test_map_losses, test_min_num_losses=test_min_num_losses,
                     test_min_shape_losses=test_min_shape_losses)
        epoch_timer.stop_timer()

    print(f'Final Train performance:')
    print(f'Train Loss (Num/Map/MinNum/MinShape): {number_train_loss:.6}/{map_train_loss:.6}/{min_num_train_loss}/{min_shape_train_loss}, \t Train Performance (Num/MapAUC/MapF1/MinNum/MinShape) {num_acc:.3}%/{auc:.3}/{f1:.3}/{min_num_acc:.3}/{min_shape_acc:.3}')
    print(f'Final Test performance:')
    print(f'Test Loss (Num/Map/MinNum/MinShape): {number_test_loss:.6}/{map_test_loss:.6}/{min_num_test_loss}/{min_shape_test_loss}, \t Test Performance (Num/MapAUC/MapF1/MinNum/MinShape) {test_num_acc:.3}%/{test_auc:.3}/{test_f1:.3}/{test_min_num_acc:.3}/{test_min_shape_acc:.3}')

    # print(f'Loss (Map/Numb): {map_train_loss:.6}/{number_train_loss:.6}, \t Accuracy: \t Number {numb_acc:.3}% \t Map AUC: {auc:.3}% \t Map f1: {f1:.3}')
    np.savez('results/'+filename, numb_accs=num_accs, map_auc=map_auc, map_f1=map_f1,
             min_num_accs=min_num_accs, min_shape_accs=min_shape_accs,
             numb_losses=num_losses, map_losses=map_losses, min_num_losses=min_num_losses,
             min_shape_losses=min_shape_losses)
    torch.save(model, f'models/{filename}.pt')

def verify_dataset(data=None, target=None, seq_len=None, map_size=900):
    print('Verifying dataset...')
    if data is None:
        device = torch.device("cpu")
        seq_len = 7
        out_size = 7
        map_size = map_size
        data, target, target_map, _, _ = get_data(seq_len, out_size, map_size, device)

    # for i, row in enumerate(data):
    #     data[i, :, :] = row[torch.randperm(seq_len), :]
    width = int(np.sqrt(data.shape[-1]).round())
    data = data.numpy()
    target = target.numpy()
    # data.nonzero()

    # Verify that the number labels are correct
    correct = 0
    for row, label in zip(data, target):
        # row = data[0, :, :]
        pred = len(np.unique(row.nonzero()[1]))
        if label == pred:
            correct += 1
    print(f'{correct}/{data.shape[0]} sequences correct')

    # Verify equal number of examples per class
    classes = np.unique(target)
    n_examples = [len(target[target == class_]) for class_ in classes]
    assert all([n_examples[0] == tally for tally in n_examples])
    # print(len(target[target==2]))
    # print(len(target[target==3]))
    # print(len(target[target==4]))
    # print(len(target[target==5]))
    # print(len(target[target==6]))

    # Verify each sequence element consists of a single one hot vector
    assert all((data.sum(axis=2) == 1).flatten())

    # Inspect the distribution over map locations
    data.shape
    plt.imshow(data.sum(axis=(0, 1)).reshape(width, width))
    plt.colorbar()
    plt.savefig('toy_dist.png')
    plt.close()


def get_min_data(preglimpsed, train_on, device=None):
    """Synthesize data for unique task.

    Args:
        seq_len (int): Length of each sequence.
        n_classes (int): Number of output units. Length of target vector.
        device (torch.device): On which device to create the data tensors.

    Returns:
        FloatTensor: Input sequences of one hot vectors
        FloatTensor: Target vector for number of unique items in each sequence
        FloatTensor: Target vector for map of unique items (no count)
        FloatTensor: Target vector for control condition, alternative
                     representation of number of unique items

    """
    if device is None:
        device = torch.device("cpu")
    n_glimpses = 11

    with torch.no_grad():
        print(f'Loading presaved dataset {preglimpsed}...')
        dir = 'preglimpsed_location_sequences/'
        num = np.load(f'{dir}number_{preglimpsed}.npy')
        num = torch.from_numpy(num).long()
        num = num.to(device)
        shape = np.load(f'{dir}shape_{preglimpsed}.npy')
        shape = torch.from_numpy(shape).long()
        shape = shape.to(device)
        min_num = np.load(f'{dir}min_num_{preglimpsed}.npy')
        min_num = torch.from_numpy(min_num).long()
        min_num = min_num.to(device)
        min_shape = np.load(f'{dir}min_shape_{preglimpsed}.npy')
        min_shape = torch.from_numpy(min_shape).long()
        min_shape = min_shape.to(device)
        if train_on == 'loc':
            gaze = np.load(f'{dir}gazes_{preglimpsed}.npy')
            gaze = torch.from_numpy(gaze).float()
            gaze = gaze.to(device)
            data = gaze
        elif train_on == 'pix':
            pix = np.load(f'{dir}pix_{preglimpsed}.npy')
            pix = torch.from_numpy(pix).float()
            pix = pix.to(device)
            data = pix
        elif train_on == 'both':
            gaze = np.load(f'{dir}gazes_{preglimpsed}.npy')
            pix = np.load(f'{dir}pix_{preglimpsed}.npy')
            data = np.concatenate((gaze, pix), axis=2)
            data = torch.from_numpy(data).float()
            data = data.to(device)

        map_ = np.load(f'{dir}map_{preglimpsed}.npy')
        map_ = torch.from_numpy(map_).float()
        map_ = map_.to(device)

            # data = np.load(f'preglimpsed_location_sequences/all_data_{preglimpsed}.npy')
            # target = np.load(f'preglimpsed_location_sequences/all_target_{preglimpsed}.npy')
            # target = torch.from_numpy(target).long()
            # target = target.to(device)
            # target_map = np.load(f'preglimpsed_location_sequences/all_map_target_{preglimpsed}.npy')
            # target_map = torch.from_numpy(target_map)
            # target_map = target_map.to(device)

            # target_mapB = (data.sum(axis=1) > 0) * 1
            # target_mapB = torch.from_numpy(target_map).float()
            # target_mapB = target_map.to(device)

            # data = torch.from_numpy(data).float()
            # data = data.to(device)

    return data, num, shape, map_, min_num, min_shape

def get_data(seq_len=7, n_classes=7, map_size=1056, device=None, preglimpsed=None):
    """Synthesize data for unique task.

    Args:
        seq_len (int): Length of each sequence.
        n_classes (int): Number of output units. Length of target vector.
        device (torch.device): On which device to create the data tensors.

    Returns:
        FloatTensor: Input sequences of one hot vectors
        FloatTensor: Target vector for number of unique items in each sequence
        FloatTensor: Target vector for map of unique items (no count)
        FloatTensor: Target vector for control condition, alternative
                     representation of number of unique items

    """
    print(f'Synthesizing data (map_size={map_size})...')
    # n_rows = sum([math.comb(seq_len, k) + math.comb(math.comb(seq_len, k), seq_len-k) for k in range(2,6)])
    # [ncr(7, k) + ncr(ncr(seq_len, k), seq_len-k) for k in range(2, 6)]

    # n_rows = 1001
    # n_rows = 126*4  # 126 is the maximum for which we can have even distribution of classes
    classes = [2, 3, 4, 5, 6, 7]
    combinations_per_class = [ncr(map_size, n) for n in classes]
    max_examples_per_class = int(min(45000/len(classes), min(combinations_per_class)))
    class_tally = [0]*len(classes)
    n_rows = max_examples_per_class*len(classes)
    if device is None:
        device = torch.device("cpu")

    data = torch.zeros((n_rows, seq_len, map_size), dtype=torch.float, device=device)
    target = torch.zeros((n_rows), dtype=torch.long, device=device)
    target_map = torch.zeros((n_rows, map_size), dtype=torch.float, device=device)
    target_control = torch.zeros((n_rows, n_classes), dtype=torch.float, device=device)
    target_hist = torch.zeros((n_rows, map_size), dtype=torch.long, device=device)
    locations = np.arange(0, map_size)

    if preglimpsed is not None:
        with torch.no_grad():
            print(f'Loading presaved dataset {preglimpsed}...')
            data = np.load(f'preglimpsed_location_sequences/all_data_{preglimpsed}.npy')

            # nex, sq, fl = data.shape
            # datars = data.reshape(nex, sq, int(np.sqrt(fl)), int(np.sqrt(fl)))
            # data = datars[:, :, 1:-1, 1:-1]
            # new_fl = data.shape[-1]**2
            # data = data.reshape(nex, sq, -1)
            # print(data.shape)

            target = np.load(f'preglimpsed_location_sequences/all_target_{preglimpsed}.npy')
            target = torch.from_numpy(target).long()
            target = target.to(device)
            target_map = np.load(f'preglimpsed_location_sequences/all_map_target_{preglimpsed}.npy')
            target_map = torch.from_numpy(target_map)
            target_map = target_map.to(device)

            # target_mapB = (data.sum(axis=1) > 0) * 1
            # target_mapB = torch.from_numpy(target_map).float()
            # target_mapB = target_map.to(device)

            data = torch.from_numpy(data).float()
            data = data.to(device)




        return data, target, target_map, target_control, target_hist

    # min([ncr(map_size, n) for n in classes])

    i = 0
    for n_unique in classes:
        print(f'Working on class {n_unique} examples...')
        # for subset in combinations(locations, n_unique):
        for _ in range(max_examples_per_class):
            np.random.shuffle(locations)
            subset = np.array(random_combination(locations, n_unique))
            # data[i, subset] = 1
            # target[i] = len(subset)
            # n_unique = len(subset)
            # Fill the rest of the sequence
            # sequence = np.ones(seq_len,)
            # sequence[:n_unique] = subset
            sequence = subset
            # fill = seq_len - n_unique
            # for i in range(n_unique + 1, seq_len):
            #     sequence[i] = np.random.choice(subset)
            while len(sequence) < seq_len:
                sequence = np.append(sequence, np.random.choice(subset))
            #  Shuffle the order
            np.random.shuffle(sequence)
            # sequence = sequence[np.random.permutation(len(sequence))]


            # while len(sequence) < seq_len:
            #     subset_repeat = subset_repeat + subset
            # combs = [subset + end for end in combinations(subset_repeat, fill)]
            # combs = [combs[i] for i in np.random.permutation(len(combs))]
            # If we need more example sequences for this number, then
            # include sequences with different distributions of the same
            # locations. Otherwise, prioritize sequences that are as
            # diverse as possible in terms of locations
            # if ncr(len(locations), n_unique) >= max_examples_per_class:

            # Pick a fill set at random (combs is shuffled)
            # com = combs[0]
            data[i, range(seq_len), sequence] = 1
            target[i] = n_unique
            class_tally[n_unique - 2] += 1
            target_map[i, subset] = 1
            target_control[i, :n_unique] = 1
            sequence_ = list(sequence)
            target_hist[i, :] = torch.tensor([sequence_.count(j) for j in range(map_size)])
            i += 1
            # if class_tally[len(subset) - 2] >= max_examples_per_class:
            #     break
            # else:
            #     for com in combs:
            #         data[i, range(seq_len), com] = 1
            #         target[i] = n_unique
            #         class_tally[n_unique - 2] += 1
            #         target_nocount[i, subset] = 1
            #         target_control[i, :n_unique] = 1
            #         target_hist[i, :] = torch.tensor([com.count(j) for j in range(map_size)])
            #         i += 1
            #         if class_tally[len(subset) - 2] >= max_examples_per_class:
            #             break
            # if class_tally[len(subset) - 2] >= max_examples_per_class:
            #     break
            # data[i-inc:i, :, :]
    # assert all(data[:, torch.randperm(seq_len), :].sum(dim=2).numpy().flatten() == 1)
    # assert all([sum(target_nocount[k]) == target[k] for k in range(n_rows)])
    # data.zero_()
    # targets = torch.randint(0, n_classes, (batch_size,), device=device)
    # class_labels = torch.arange(0, n_classes)
    # for i in range(batch_size):
    #     shuffled = class_labels[torch.randperm(len(class_labels))]
    #     to_sample = shuffled[:targets[i] + 1]
    #     to_sample_cat = to_sample
    #     while len(to_sample_cat) < seq_len:
    #         sample_shuffled = to_sample[torch.randperm(len(to_sample))]
    #         to_sample_cat = torch.cat((to_sample_cat, sample_shuffled), dim=0)
    #     indices = to_sample_cat[:seq_len]
    #     data[i, range(seq_len), indices] = 1
    print(f'Synthesized {i} example sequences')
    # data = data[:i, :, :]
    # target = target[:i]
    # target_map = target_nocount[:i, :]
    # target_control = target_control[:i, :]
    # target_hist = target_hist[:i, :]

    return data, target, target_map, target_control, target_hist

def test_transfer(model):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()
    batch_size = 64
    seq_len = 7
    n_classes = 7

    data = torch.from_numpy(np.load('../relational_saccades/all_data_ds5_randseq.npy'))
    target = torch.from_numpy(np.load('../relational_saccades/all_target_ds5_randseq.npy'))
    target = target.long()
    dataset = TensorDataset(data, target)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    correct = 0
    test_loss = 0

    for batch_idx, (inputs, label) in enumerate(loader):
        bs = inputs.shape[0]
        hidden = model.initHidden(bs)
        model.zero_grad()

        for i in range(seq_len):
            output, hidden = model(inputs[:, i, :], hidden)
        loss = criterion(output, label)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(label.view_as(pred)).sum().item()
        test_loss += loss.item()
    test_loss /= batch_idx + 1
    # if not ep % 25:
    acc = 100. * (correct / len(dataset))
    print(f'Transfer performance: \t Loss: {test_loss:.6}, Accuracy: {acc:.6}% ({correct}/{len(dataset)})')

def main(config):
    map_size = config.map_size
    model_version = config.model_version
    use_loss = config.use_loss
    rnn_act = config.rnn_act
    preglimpsed = config.preglimpsed
    eye_weight = config.eye_weight
    detached = True if 'detached' in use_loss else False
    # lr = 0.01
    lr = config.lr
    mom = 0.9
    wd = config.wd
    wd_readout = config.wd
    wd_rnn = 0
    # config.n_epochs = 2000
    config.control = None


    # input_size = 1  # difference code
    # input_size = 26**2 # ds-factor=5 676
    # input_size = 10 # for testing
    # input_size = 33**2 # 1089   # one_hot
    # out_size = 8
    numb_size = 8

    # device = torch.device("cuda")
    if config.no_cuda:
        config.device = torch.device("cpu")
    else:
        config.device = torch.device("cuda")

    # map_sizes = np.append([7, 15], np.arange(25, 925, 25))
    # map_sizes = np.arange(100, 1000, 100)[::-1]
    # map_sizes = [7, 15, 25, 50, 100, 200, 300, 600]
    # map_sizes = [10, 25, 50, 100, 200, 400, 676]
    # map_sizes = [50, 100, 200, 400, 676]
    # map_sizes = [10, 15, 25, 50, 100, 200, 400, 676]
    # map_sizes = [1056]
    # for map_size in map_sizes:
    #     verify_dataset(map_size=map_size)

    # model_versions = ['two_step', 'two_step_ws', 'two_step_ws_init']
    # model_version = 'two_step'

    # model = RNN(input_size, hidden_size, out_size)
    # opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=mom)
    # train_rnn(model, opt, n_epochs)



    config.differential = False
    # map_loss = True
    # if differential:
    #     input_size = 1
    # map_sizes = np.arange(10, 900, 10)
    # map_sizes = [900]
    # n_exps = len(map_sizes)
    # num_acc = np.zeros(n_exps,)
    # map_acc = np.zeros(n_exps,)

    # map size x objective experiment (takes too long)

    # for i, map_size in enumerate(map_sizes):
    #     # for model_version in ['one_step_tanh','two_step', 'two_step_ws', 'two_step_ws_init']:
    #     for model_version in ['two_step_ws']:
    #         # for use_loss in ['number', 'both', 'map', 'map_then_both','both-detached']:
    #         for use_loss in ['map_then_both-detached']:
                # if use_loss == 'map_then_both' and map_size <= 50:
                #     continue
    kwargs = {'act':rnn_act, 'eye_weight':eye_weight, 'detached':detached, 'dropout':config.dropout}
    if config.train_on == 'pix':
        input_size = 1152
    elif config.train_on == 'loc':
        input_size = map_size
    elif config.train_on == 'both':
        input_size = map_size + 1152
    hidden_size = int(np.round(input_size*1.1))
    # hidden_size = map_size*2
    if model_version == 'two_step':
        model = TwoStepModel(input_size, hidden_size, map_size, numb_size, **kwargs)
    elif model_version == 'two_step_conv':
        model = ConvReadoutMapNet(input_size, hidden_size, map_size, numb_size, **kwargs)
    elif model_version == 'integrated':
        model = IntegratedModel(input_size, hidden_size, map_size, numb_size, **kwargs)
    elif model_version == 'three_step':
        model = ThreeStepModel(input_size, hidden_size, map_size, numb_size, **kwargs)
    elif model_version == 'content_gated':
        model = ContentGated_cheat(hidden_size, map_size, numb_size, **kwargs)
    elif model_version == 'two_step_ws':
        if 'detached' in use_loss:
            model = TwoStepModel_weightshare_detached(hidden_size, map_size, numb_size, init=False, **kwargs)
        else:
            model = TwoStepModel_weightshare(hidden_size, map_size, numb_size, init=False, **kwargs)
    elif model_version == 'two_step_ws_init':
        if 'detached' in use_loss:
            model = TwoStepModel_weightshare_detached(hidden_size, map_size, numb_size, init=True, **kwargs)
        else:
            model = TwoStepModel_weightshare(hidden_size, map_size, numb_size, init=True, **kwargs)
    elif 'one_step' in model_version:
        if use_loss == 'number':
            model = RNN(map_size, hidden_size, numb_size, **kwargs)
        elif use_loss == 'map':
            model = RNN(map_size, hidden_size, map_size, **kwargs)
    elif model_version == 'number_as_sum':
        model = NumberAsMapSum(map_size, hidden_size)
    else:
        print('Model version not implemented.')
        exit()

    print('Params to learn:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"\t {name} {param.shape}")

    if config.pretrained_map:
        # Load trained model and initialize rnn params with pretained
        if os.path.isdir('models_nooper'):
            saved_model = torch.load('models_nooper/two_step_results_mapsz900_loss-both-detached_sched_lr0.01_wd0.0_dr0.0_tanhrelu_rand_trainon-loc_no_hearts_2-7_varylum_130x130_90000_noresize_11glimpses.pt')
        else:
            saved_model = torch.load('models/two_step_results_mapsz900_loss-both-detached_sched_lr0.01_wd0.0_dr0.0_tanhrelu_rand_trainon-loc_no_hearts_2-7_varylum_130x130_90000_noresize_11glimpses.pt')

        # model.load_state_dict(saved_model)
        with torch.no_grad():
            model.rnn = saved_model.rnn
        # model.rnn.i2h = saved_model.rnn.i2h
        # model.rnn.i2o = saved_model.rnn.i2o
        for name, param in model.rnn.named_parameters():
            param.requires_grad = False


    # Apparently should move model to device before constructing optimizers for it
    model = model.to(config.device)

    if config.use_schedule:
        # Learning rate scheduler
        start_lr = 0.1
        scale = 0.9978  # 0.9955
        if 'detached' in use_loss:
            opt_rnn = torch.optim.SGD(model.rnn.parameters(), lr=start_lr, momentum=mom, weight_decay=wd_rnn)
            if 'two' in model_version:
                opt_readout = torch.optim.SGD(model.readout.parameters(), lr=start_lr, momentum=mom, weight_decay=wd_readout)
            elif 'three' in model_version:
                readout_params = [{'params': model.readout1.parameters()}, {'params':model.readout2.parameters()}]
                opt_readout = torch.optim.SGD(readout_params, lr=start_lr, momentum=mom, weight_decay=wd_readout)
            lambda1 = lambda epoch: scale ** epoch
            scheduler_rnn = torch.optim.lr_scheduler.LambdaLR(opt_rnn, lr_lambda=lambda1)
            scheduler_readout = torch.optim.lr_scheduler.LambdaLR(opt_readout, lr_lambda=lambda1)
            opt = [opt_rnn, opt_readout]
            scheduler = [scheduler_rnn, scheduler_readout]
        else:
            opt = torch.optim.SGD(model.parameters(), lr=start_lr, momentum=mom, weight_decay=wd)
            lambda1 = lambda epoch: scale ** epoch
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda1)
    else:
        if 'detached' in use_loss:
            opt_rnn = torch.optim.SGD(model.rnn.parameters(), lr=lr, momentum=mom, weight_decay=wd_rnn)
            # if 'two' in model_version:
            opt_readout = torch.optim.SGD(model.readout.parameters(), lr=lr, momentum=mom, weight_decay=wd_readout)
            # elif 'three' in model_version:
            #     readout_params = [{'params': model.readout1.parameters()}, {'params':model.readout2.parameters()}]
            #     opt_readout = torch.optim.SGD(readout_params, lr=start_lr, momentum=mom, weight_decay=wd)
            opt = [opt_rnn, opt_readout]
        else:
            opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=mom, weight_decay=wd)
            scheduler = None

        # n_epochs = 2000
        # start_lr = 0.1
        # scale = 0.9978
        # gamma=0.1
        # lrs = np.zeros((n_epochs+1,))
        # lrs_lr = np.zeros((n_epochs+1,))
        # lrs_exp = np.zeros((n_epochs+1,))
        # lrs[0] = start_lr
        # lrs_lr[0] = start_lr
        # lrs_exp[0] = start_lr
        # for ep in range(n_epochs):
        #     lrs[ep+1] = lrs[ep] * scale
        #     lrs_lr[ep+1] = start_lr * (scale ** ep)
        #     lrs_exp[ep+1] = lrs_exp[ep] * scale
        # print(lrs[1000])
        # lrs[:50]
        # print(lrs[-10:])
        # plt.plot(lrs)
        # plt.plot(lrs_lr, label='lambda')
        # plt.plot(lrs_exp, label='exponential')
        # plt.legend()

    if 'two' in model_version or 'three' in model_version or model_version=='integrated':
        # train_two_step_model(model, opt, n_epochs, device, differential,
        #                      use_loss, model_version, preglimpsed=preglimpsed, eye_weight=eye_weight)
        train_two_step_model(model, opt, config, scheduler)
    elif model_version == 'content_gated':
        train_content_gated_model(model, opt, config, scheduler)
    elif 'one' in model_version and (use_loss == 'number' or use_loss == 'map'):
        # train_rnn(model, opt, n_epochs, device, model_version, use_loss, preglimpsed, eye_weight)
        train_rnn(model, opt, config)
    elif model_version == 'number_as_sum':
        # train_sum_model(model, opt, n_epochs, device, model_version, use_loss)
        train_sum_model(model, opt, config)

    # plt.plot(num_acc, label='number')
    # plt.plot(map_acc, label='')

    # torch.save(model, 'models/two_step_model_with_maploss_bias.pt')

    # model = TwoStepModel_hist(hidden_size, map_size, numb_size)
    # opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=mom, weight_decay=wd)
    # train_two_step_model(model, opt, n_epochs, device, differential, control='hist')

    # map_loss = True
    # model = TwoStepModel_hardcode(input_size, hidden_size, map_size, numb_size)
    # opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=mom, weight_decay=wd)
    # train_two_step_model(model, opt, n_epochs, device, differential, map_loss)

def get_config():
    parser = argparse.ArgumentParser(description='PyTorch network settings')
    parser.add_argument('--model_version', type=str, default='two_step')
    parser.add_argument('--use_loss', type=str, default='both')
    parser.add_argument('--map_size', type=int, default=15)
    parser.add_argument('--rnn_act', type=str, default=None)
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--preglimpsed', type=str, default=None)
    parser.add_argument('--eye_weight', action='store_true', default=False)
    parser.add_argument('--use_schedule', action='store_true', default=False)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--wd', type=float, default=0) # 1e-6
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--train_on', type=str, default='loc')  ## loc, pix, or both
    parser.add_argument('--n_epochs', type=int, default=2000)
    parser.add_argument('--pretrained_map', action='store_true', default=False)
    config = parser.parse_args()
    print(config)
    return config

if __name__ == '__main__':
    config = get_config()
    main(config)
