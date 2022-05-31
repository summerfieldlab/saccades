import numpy as np
import math
import torch
from torch import nn
from scipy.stats import special_ortho_group

class PixelPlusShapeModel(nn.Module):
    """RNN receives joint embedding of pixels and predicted shape label.

    A convolutional module receives the raw pixel inputs at each glimpse and
    tries to predict the shape label per glimpse. The last fc layer of the
    conv module is the pixel embedding to be passed on to the RNN, which is
    trained to classify the numerosity of the image. The input to the RNN is a
    joint embedding of the pixel embedding and the predicted shape label. The
    joint embedding is constructed by concatenating them and applying a random
    rotation. This rotation may or may not be beneficial. Anecdotal evidence
    suggests that neural networks can have a hard time using auxiliary inputs
    when it represented in fewer input units than the primary input.
    """
    def __init__(self, hidden_size, **kwargs):
        super().__init__()
        drop_rnn = kwargs['drop_rnn'] if 'drop_rnn' in kwargs.keys() else 0.0
        drop_readout = kwargs['drop_readout'] if 'drop_readout' in kwargs.keys() else 0.0
        drop_emb = 0.1
        device = kwargs['device']
        # pix_size = 1152
        # self.width = 48
        # self.height = 24

        # pix_size = 512
        # self.width = 16*2
        # self.height = 16

        pix_size = 288
        self.width = 12*2
        self.height = 12

        shape_size = 14
        self.hidden_size = hidden_size
        rnn_out_size = 200
        fc2_size = 10
        number_size = 8

        self.conv = ConvNet(self.width, self.height, shape_size, drop_emb, big=False)
        emb_size = self.conv.fc1_size + shape_size # 120 + 14
        self.LReLU = nn.LeakyReLU(0.1)
        self.rnn = RNN(emb_size, self.hidden_size, rnn_out_size, dropout=drop_rnn)
        self.fc2 = nn.Linear(rnn_out_size, fc2_size)
        self.drop_layer = nn.Dropout(p=drop_readout)
        self.readout = nn.Linear(fc2_size, number_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.rot_mat = torch.from_numpy(special_ortho_group.rvs(emb_size)).float().to(device)

    def forward(self, pix, hidden):
        # reshape
        shape, pix_emb = self.conv(pix.view((-1, 1, self.width, self.height)))
        shape2 = shape.detach().clone()
        # Concatenate the shape output and the pixel embedding
        # Apply a random rotation to embed these two signals together
        # Otherwise, network may struggle to take advantage of shape signal
        rnn_input = torch.cat((pix_emb, self.softmax(shape2)), dim=1)
        rnn_input = torch.matmul(rnn_input, self.rot_mat)
        rnn_out, hidden = self.rnn(rnn_input, hidden)
        number = self.LReLU(self.fc2(rnn_out))
        number = self.drop_layer(number)
        number = self.readout(number)
        return number, shape, hidden


class ShapeModel(nn.Module):
    """RNN recieves only the (detached) prediced shape label as input."""
    def __init__(self, hidden_size, **kwargs):
        super().__init__()
        drop_rnn = kwargs['drop_rnn'] if 'drop_rnn' in kwargs.keys() else 0.0
        drop_readout = kwargs['drop_readout'] if 'drop_readout' in kwargs.keys() else 0.0
        drop_emb = 0.1
        device = kwargs['device']
        # pix_size = 1152
        # self.width = 48
        # self.height = 24

        pix_size = 512
        self.width = 16*2
        self.height = 16

        # pix_size = 288
        # self.width = 12*2
        # self.height = 12

        shape_size = 14
        self.hidden_size = hidden_size
        rnn_out_size = 200
        fc2_size = 10
        number_size = 8

        # self.conv = ConvNet(self.width, self.height, shape_size, drop_emb, big=True)
        self.conv = PixConvNet(self.width, self.height, shape_size, drop_emb)
        emb_size = self.conv.fc1_size + shape_size # 120 + 14
        self.LReLU = nn.LeakyReLU(0.1)
        self.rnn = RNN(7, self.hidden_size, rnn_out_size, dropout=drop_rnn)
        self.fc2 = nn.Linear(rnn_out_size, fc2_size)
        self.drop_layer = nn.Dropout(p=drop_readout)
        self.readout = nn.Linear(fc2_size, number_size)
        # self.rot_mat = torch.from_numpy(special_ortho_group.rvs(emb_size)).float().to(device)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, pix, hidden, bce=False):
        # reshape
        shape, pix_emb = self.conv(pix.view((-1, 1, self.width, self.height)))
        if bce:
            shape2 = torch.sigmoid(shape[:, :7].detach().clone())
        else:
            shape2 = self.softmax(shape[:, :7].detach().clone())
        # Concatenate the shape output and the pixel embedding
        # Apply a random rotation to embed these two signals together
        # Otherwise, network may struggle to take advantage of shape signal
        # rnn_input = torch.cat((pix_emb, self.softmax(shape2)), dim=1)
        # rnn_input = torch.matmul(rnn_input, self.rot_mat)
        rnn_out, hidden = self.rnn(shape2, hidden)
        number = self.LReLU(self.fc2(rnn_out))
        number = self.drop_layer(number)
        number = self.readout(number)
        return number, shape, hidden

class Distinctive(nn.Module):
    def __init__(self, hidden_size, **kwargs):
        super().__init__()
        drop_rnn = kwargs['drop_rnn'] if 'drop_rnn' in kwargs.keys() else 0.0
        drop_readout = kwargs['drop_readout'] if 'drop_readout' in kwargs.keys() else 0.0
        drop_emb = 0.1
        # pix_size = 1152
        # self.width = 48
        # self.height = 24

        # pix_size = 512
        # self.width = 16*2
        # self.height = 16

        pix_size = 288
        self.width = 12*2
        self.height = 12

        shape_size = 14
        self.hidden_size = hidden_size
        rnn_out_size = 200
        fc2_size = 10
        number_size = 8

        self.conv = ConvNet(self.width, self.height, shape_size, drop_emb, big=False)
        emb_size = self.conv.fc1_size # 120
        self.LReLU = nn.LeakyReLU(0.1)
        self.rnn = RNN(emb_size, self.hidden_size, rnn_out_size, dropout=drop_rnn)
        self.fc2 = nn.Linear(rnn_out_size, fc2_size)
        self.drop_layer = nn.Dropout(p=drop_readout)
        self.readout = nn.Linear(fc2_size, number_size)

    def forward(self, pix, hidden):
        # reshape
        shape, pix_emb = self.conv(pix.view((-1, 1, self.width, self.height)))
        rnn_out, hidden = self.rnn(pix_emb, hidden)
        number = self.LReLU(self.fc2(rnn_out))
        number = self.drop_layer(number)
        number = self.readout(number)
        return number, shape, hidden

class DistinctiveCheat(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        drop_rnn = kwargs['drop_rnn'] if 'drop_rnn' in kwargs.keys() else 0.0
        drop_readout = kwargs['drop_readout'] if 'drop_readout' in kwargs.keys() else 0.0
        input_size = 14
        self.hidden_size = 25
        output_size = 25
        fc2_size = 10
        number_size = 8
        self.fc1 = nn.Linear(14, 14)
        self.LReLU = nn.LeakyReLU(0.1)
        self.rnn = RNN(input_size, self.hidden_size, output_size, dropout=drop_rnn)
        self.fc2 = nn.Linear(output_size, fc2_size)
        self.drop_layer = nn.Dropout(p=drop_readout)
        self.readout = nn.Linear(fc2_size, number_size)

    def forward(self, glimpse_label, hidden):
        glimpse_emb = self.LReLU(self.fc1(glimpse_label))
        rnn_out, hidden = self.rnn(glimpse_emb, hidden)
        number = self.LReLU(self.fc2(rnn_out))
        number = self.drop_layer(number)
        number = self.readout(number)
        return number, hidden


class DistinctiveCheat_small(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        drop_rnn = kwargs['drop_rnn'] if 'drop_rnn' in kwargs.keys() else 0.0
        drop_readout = kwargs['drop_readout'] if 'drop_readout' in kwargs.keys() else 0.0
        input_size = 14
        self.hidden_size = 25
        output_size = 25
        fc2_size = 10
        number_size = 8
        # self.fc1 = nn.Linear(14, 14)
        # self.LReLU = nn.LeakyReLU(0.1)
        self.rnn = RNN(input_size, self.hidden_size, number_size, dropout=drop_rnn)
        # self.fc2 = nn.Linear(output_size, fc2_size)
        # self.drop_layer = nn.Dropout(p=drop_readout)
        # self.readout = nn.Linear(fc2_size, number_size)

    def forward(self, glimpse_label, hidden):
        # glimpse_emb = self.LReLU(self.fc1(glimpse_label))
        number, hidden = self.rnn(glimpse_label, hidden)
        # number = self.LReLU(self.fc2(rnn_out))
        # number = self.drop_layer(number)
        # number = self.readout(number)
        return number, hidden


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
        """Integrate pixel and gaze location streams.

        One embedding layer for pixels, merged with place code gaze location
        before input into RNN trained to produce map of object locations.
        Three layer conv readout from map.
        """
        super().__init__()
        act = kwargs['act'] if 'act' in kwargs.keys() else None
        eye_weight = kwargs['eye_weight'] if 'eye_weight' in kwargs.keys() else False
        detached = kwargs['detached'] if 'detached' in kwargs.keys() else False
        # dropout = kwargs['dropout'] if 'dropout' in kwargs.keys() else 0.0
        drop_rnn = kwargs['drop_rnn'] if 'drop_rnn' in kwargs.keys() else 0.0
        drop_readout = kwargs['drop_readout'] if 'drop_readout' in kwargs.keys() else 0.0
        self.hidden_size = hidden_size
        self.map_size = map_size
        self.width = int(np.sqrt(map_size))
        self.out_size = output_size
        self.detached = detached
        # detached=False should always be false for the twostep model here,
        # self.pix_fc1 = nn.Linear(1152, 2000)

        # self.pix_fc1 = nn.Linear(1152, map_size)  # when glimpse width=24
        self.pix_fc1 = nn.Linear(288, map_size)  # when glimpse width=12
        self.LReLU = nn.LeakyReLU(0.1)
        self.pix_fc2 = nn.Linear(map_size, map_size)

        # even if we want to detach in the forward of this class
        self.rnn = ConvReadoutMapNet(map_size*2, hidden_size, map_size, output_size, **kwargs)
        self.readout = self.rnn.readout
        # self.rnn = TwoStepModel(map_size*2, hidden_size, map_size, output_size, detached=detached, dropout=0)
        # self.rnn = tanhRNN(input_size, hidden_size, map_size)
        # self.readout1 = nn.Linear(map_size, map_size, bias=True)
        # self.readout = nn.Linear(map_size, output_size, bias=True)
        # self.readout = ConvNet(self.width, self.width, self.out_size, drop_readout, big=False)

    def forward(self, x, hidden):
        gaze = x[:, :self.map_size]
        pix = x[:, self.map_size:]
        pix = self.LReLU(self.pix_fc1(pix))
        pix = self.LReLU(self.pix_fc2(pix))
        # pix = torch.relu(self.pix_fc2(pix))
        combined = torch.cat((gaze, pix), dim=1)
        number, map_, hidden = self.rnn(combined, hidden)

        return number, map_, hidden

class TwoRNNs(nn.Module):
    def __init__(self, input_size, hidden_size, map_size, output_size, **kwargs):
        super().__init__()
        act = kwargs['act'] if 'act' in kwargs.keys() else None
        eye_weight = kwargs['eye_weight'] if 'eye_weight' in kwargs.keys() else False
        detached = kwargs['detached'] if 'detached' in kwargs.keys() else False
        # dropout = kwargs['dropout'] if 'dropout' in kwargs.keys() else 0.0
        drop_rnn = kwargs['drop_rnn'] if 'drop_rnn' in kwargs.keys() else 0.0
        drop_readout = kwargs['drop_readout'] if 'drop_readout' in kwargs.keys() else 0.0
        pix_to_map = kwargs['pix_to_map'] if 'pix_to_map' in kwargs.keys() else False
        self.hidden_size = hidden_size
        self.map_size = map_size
        self.width = int(np.sqrt(map_size))
        self.out_size = output_size
        self.detached = detached
        # detached=False should always be false for the twostep model here,
        # self.pix_fc1 = nn.Linear(1152, 2000)

        # self.pix_fc1 = nn.Linear(1152, map_size)  # when glimpse width=24
        pix_sz = 288
        # Two pixel embedding layers
        self.pix_fc1 = nn.Linear(pix_sz, map_size)  # when glimpse width=12
        self.LReLU = nn.LeakyReLU(0.1)
        self.pix_fc2 = nn.Linear(map_size, map_size)

        # Two RNNs, one for pixels, one to produce a map of objects
        self.pix_rnn = RNN(pix_sz, hidden_size, map_size, act, eye_weight, drop_rnn)
        map_in_sz = input_size if pix_to_map else map_size
        self.map_rnn = RNN(map_in_sz, hidden_size, map_size, act, eye_weight, drop_rnn)

        # Readout from both streams
        self.pix_readout = nn.Linear(map_size, self.out_size)
        self.map_readout = ConvNet(self.width, self.width, self.out_size, drop_readout)

        # Combined readout
        self.fc = nn.Linear(self.out_size*2, self.out_size)


    def forward(self, x, hidden, pix_to_map=False):
        """The first ."""
        pix = x[:, self.map_size:]
        pix = self.LReLU(self.pix_fc1(pix))
        pix = self.LReLU(self.pix_fc2(pix))
        pix, hidden[self.map_size:] = self.pix_rnn(pix, hidden[self.map_size:])
        pix = self.LReLU(self.pix_readout(pix))

        gaze = x[:, :self.map_size]
        if pix_to_map:
            input_to_map_rnn = torch.cat((gaze, pix), 1)
        else:
            input_to_map_rnn = gaze
        map_, hidden[:self.map_size] = self.map_rnn(input_to_map_rnn, hidden[:self.map_size])
        gaze = self.map_readout(map.view((-1, 1, self.width, self.width)))
        combined = torch.cat((pix, gaze), dim=1)
        number = self.fc(combined)
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
        drop_rnn = kwargs['drop_rnn'] if 'drop_rnn' in kwargs.keys() else 0.0
        drop_readout = kwargs['drop_readout'] if 'drop_readout' in kwargs.keys() else 0.0
        self.rotate = kwargs['rotate'] if 'rotate' in kwargs.keys() else False
        big = kwargs['big'] if 'big' in kwargs.keys() else False
        self.hidden_size = hidden_size
        self.map_size = map_size
        self.width = int(np.sqrt(map_size))
        self.out_size = output_size
        self.detached = detached
        self.rnn = RNN(input_size, hidden_size, map_size, act, eye_weight, drop_rnn)
        self.initHidden = self.rnn.initHidden
        self.readout = ConvNet(self.width, self.width, self.out_size, drop_readout, big=big)
        if self.rotate:
            self.rot_mat = torch.from_numpy(special_ortho_group.rvs(input_size)).float()


    def forward(self, x, hidden):
        batch_size = x.shape[0]
        if self.rotate:
            x = torch.matmul(x, self.rot_mat)
        map_, hidden = self.rnn(x, hidden)
        # map_ = self.drop_layer(map_)
        if self.detached:
            map_to_pass_on = map_.detach()
        else:
            map_to_pass_on = map_

        # If this model is mapping rather than counting, nonlineartities get applied later
        if self.map_size != self.out_size: # counting
            map_to_pass_on = torch.tanh(torch.relu(map_to_pass_on))
        if self.rotate:
            map_to_pass_on = torch.matmul(map_to_pass_on, self.rot_mat)
        map_to_pass_on = map_to_pass_on.view((-1, 1, self.width, self.width))
        number, _ = self.readout(map_to_pass_on)
        # number = torch.relu(self.fc(map))
        return number, map_, hidden

class ConvNet(nn.Module):
    def __init__(self, width, height, output_size, dropout, big=False):
        super().__init__()
        # Larger version
        if big:
            self.kernel1_size = 3
            self.cnn1_nchannels_out = 128
            self.poolsize = 2
            self.kernel2_size = 2
            self.cnn2_nchannels_out = 256
            self.kernel3_size = 2
            self.cnn3_nchannels_out = 32
        else:  # Smaller version
            self.kernel1_size = 3
            self.cnn1_nchannels_out = 16
            self.poolsize = 2
            self.kernel2_size = 2
            self.cnn2_nchannels_out = 12
            self.kernel3_size = 2
            self.cnn3_nchannels_out = 9

        self.LReLU = nn.LeakyReLU(0.1)

        # Default initialization is init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # (Also assumes leaky Relu for gain)
        # which is appropriate for all layers here
        self.conv1 = nn.Conv2d(1, self.cnn1_nchannels_out, self.kernel1_size)    # (NChannels_in, NChannels_out, kernelsize)
        # torch.nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        self.pool = nn.MaxPool2d(self.poolsize, self.poolsize)     # kernel height, kernel width
        self.conv2 = nn.Conv2d(self.cnn1_nchannels_out, self.cnn2_nchannels_out, self.kernel2_size)   # (NChannels_in, NChannels_out, kernelsize)
        self.conv3 = nn.Conv2d(self.cnn2_nchannels_out, self.cnn3_nchannels_out, self.kernel3_size)
        # torch.nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        # track the size of the cnn transformations
        # self.cnn2_width_out = ((width - self.kernel1_size+1) - self.kernel2_size + 1) // self.poolsize
        # self.cnn2_height_out = ((height - self.kernel1_size+1) - self.kernel2_size + 1) // self.poolsize
        self.cnn3_width_out = (((width - self.kernel1_size+1) - self.kernel2_size + 1)  // self.poolsize) - self.kernel3_size+1
        self.cnn3_height_out = (((height - self.kernel1_size+1) - self.kernel2_size + 1)  // self.poolsize) - self.kernel3_size+1

        # pass through FC layers
        # self.fc1 = nn.Linear(int(self.cnn2_nchannels_out * self.cnn2_width_out * self.cnn2_height_out), 120)  # size input, size output
        self.fc1_size = 120
        # import pdb; pdb.set_trace()
        self.fc1 = nn.Linear(int(self.cnn3_nchannels_out * self.cnn3_width_out * self.cnn3_height_out), self.fc1_size)  # size input, size output

        self.fc2 = nn.Linear(self.fc1_size, output_size)

        # Dropout
        self.drop_layer = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.LReLU(self.conv1(x))
        x = self.pool(self.LReLU(self.conv2(x)))
        x = self.LReLU(self.conv3(x))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3]) # this reshapes the tensor to be a 1D vector, from whatever the final convolutional layer output
        # add dropout before fully connected layers, widest part of network
        x = self.drop_layer(x)
        fc1 = self.LReLU(self.fc1(x))
        out = self.fc2(fc1)
        return out, fc1


class PixConvNet(nn.Module):
    def __init__(self, width, height, output_size, dropout):
        super().__init__()
        self.width = width
        self.height = height
        self.kernel1_size = 4
        self.cnn1_nchannels_out = 15
        self.poolsize = 2
        self.kernel2_size = 3
        self.cnn2_nchannels_out = 15
        self.fc1_size = 120

        # pixels layers
        self.conv1a = nn.Conv2d(1, self.cnn1_nchannels_out, self.kernel1_size)    # (NChannels_in, NChannels_out, kernelsize)
        self.conv1b = nn.Conv2d(1, self.cnn1_nchannels_out, self.kernel1_size)
        self.pix_drop_layer = nn.Dropout(p=dropout)
        self.pool = nn.MaxPool2d(self.poolsize, self.poolsize)     # kernel height, kernel width
        self.conv2 = nn.Conv2d(self.cnn1_nchannels_out*2, self.cnn2_nchannels_out, self.kernel2_size)   # (NChannels_in, NChannels_out, kernelsize)

        # track the size of the cnn transformations
        self.cnn2_width_out = ((height - self.kernel1_size+1) - self.kernel2_size + 1) // (self.poolsize)
        self.cnn2_height_out = ((height - self.kernel1_size+1) - self.kernel2_size + 1) // (self.poolsize)

        # pass through FC layers
        self.fc1_pix = nn.Linear(int(self.cnn2_nchannels_out * self.cnn2_width_out * self.cnn2_height_out), self.fc1_size)  # size input, size output
        self.shape_out = nn.Linear(self.fc1_size, output_size)

    def forward(self, x):
        patch_a = x[:, :, :self.height, :]
        patch_b = x[:, :, self.height:, :]

        whata = torch.relu(self.conv1a(patch_a))
        whatb = torch.relu(self.conv1b(patch_b))
        what = torch.cat((whata, whatb), 1)

        what = self.pool(torch.relu(self.conv2(what)))
        what = what.view(-1, what.shape[1]*what.shape[2]*what.shape[3]) # this reshapes the tensor to be a 1D vector, from whatever the final convolutional layer output
        what = torch.relu(self.fc1_pix(what))
        what = self.pix_drop_layer(what)
        shape = self.shape_out(what)
        return shape, what
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
