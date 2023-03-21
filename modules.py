import numpy as np
import math
import torch
from torch import nn
from scipy.stats import special_ortho_group

class ConvNet(nn.Module):
    def __init__(self, width, height, map_size, output_size, **kwargs):
        super().__init__()
        grid = kwargs['grid'] if 'grid' in kwargs.keys() else 9
        big = kwargs['big'] if 'big' in kwargs.keys() else False
        dropout = kwargs['dropout'] if 'dropout' in kwargs.keys() else 0.0
        # Larger version
        if big:
            self.kernel1_size = 3
            self.cnn1_nchannels_out = 56
            self.poolsize = 2
            self.kernel2_size = 2
            self.cnn2_nchannels_out = 56
            self.kernel3_size = 2
            self.cnn3_nchannels_out = 56

            # self.kernel1_size = 3
            # self.cnn1_nchannels_out = 128
            # self.poolsize = 2
            # self.kernel2_size = 2
            # self.cnn2_nchannels_out = 256
            # self.kernel3_size = 2
            # self.cnn3_nchannels_out = 32
        elif grid==3: # Smaller version
            self.kernel1_size = 3
            self.cnn1_nchannels_out = 33
            self.poolsize = 2
            self.kernel2_size = 2
            self.cnn2_nchannels_out = 33
            self.kernel3_size = 2
            self.cnn3_nchannels_out = 33
        elif grid==6: # Smaller version
            self.kernel1_size = 3
            self.cnn1_nchannels_out = 33
            self.poolsize = 2
            self.kernel2_size = 2
            self.cnn2_nchannels_out = 30
            self.kernel3_size = 2
            self.cnn3_nchannels_out = 20
        elif grid==9:
            self.kernel1_size = 3
            self.cnn1_nchannels_out = 33
            self.poolsize = 2
            self.kernel2_size = 2
            self.cnn2_nchannels_out = 20
            self.kernel3_size = 2
            self.cnn3_nchannels_out = 20
        self.output_size = output_size
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
        self.fc1_size = 256
        # self.fc1_size = 
        # import pdb; pdb.set_trace()
        self.fc1 = nn.Linear(int(self.cnn3_nchannels_out * self.cnn3_width_out * self.cnn3_height_out), self.fc1_size)  # size input, size output
        self.fc2 = nn.Linear(self.fc1_size, map_size)
        self.fc3 = nn.Linear(map_size, output_size)

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
        fc2 = self.LReLU(self.fc2(fc1))
        out = self.fc3(fc2)
        return out, fc2, fc1
    

class RNN(nn.Module):

    # you can also accept arguments in your model constructor
    def __init__(self, data_size, hidden_size, output_size, act=None):
        super().__init__()

        self.hidden_size = hidden_size
        input_size = data_size + hidden_size
        if act == 'tanh':
            self.act_fun = torch.tanh
            gain = 5/3
        elif act == 'relu':
            self.act_fun = torch.relu
            gain = np.sqrt(2)
        elif act == 'sig':
            self.act_fun = nn.Sigmoid()
            gain = 1
        elif act == 'lrelu':
            self.act_fun = nn.LeakyReLU(0.1)
            gain = np.sqrt(2)
        else:
            self.act_fun = None
            gain = 1

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.init_params(gain)


    def forward(self, data, last_hidden):
        input = torch.cat((data, last_hidden), 1)
        hidden = self.i2h(input)
        if self.act_fun is not None:
            hidden = self.act_fun(hidden)
        output = self.h2o(hidden)
        return output, hidden

    def init_params(self, gain):
        if self.act_fun == 'relu':
            nn.init.kaiming_uniform_(self.i2h.weight, a=math.sqrt(5), nonlinearity='relu')
            nn.init.kaiming_uniform_(self.h2o.weight, a=math.sqrt(5), nonlinearity='relu')
        elif self.act_fun == 'lrelu':
            nn.init.kaiming_uniform_(self.i2h.weight, a=math.sqrt(5), nonlinearity='leaky_relu')
            nn.init.kaiming_uniform_(self.h2o.weight, a=math.sqrt(5), nonlinearity='leaky_relu')
        elif self.act_fun is None:
            nn.init.kaiming_uniform_(self.i2h.weight, a=math.sqrt(5), nonlinearity='linear')
            nn.init.kaiming_uniform_(self.h2o.weight, a=math.sqrt(5), nonlinearity='linear')
        else:
            nn.init.xavier_uniform_(self.i2h.weight, gain=gain)
            nn.init.xavier_uniform_(self.h2o.weight, gain=gain)

    def initHidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)


class MultRNN(nn.Module):
    """ Include multiplicative or gated interactions between input and hidden.

    Normally this would require a weight tensor. Here we use a factorized
    version as described in:
    Sutskever, I., Martens, J., & Hinton, G. (2011). Generating text with
    recurrent neural networks. Proceedings of the 28th International Conference
    on Machine Learning, ICML 2011, 1017–1024.
    """
    def __init__(self, input_size, hidden_size, factor_size, output_size, small_weights):
        super().__init__()
        self.hidden_size = hidden_size
        self.i2f = nn.Linear(input_size, factor_size, bias=False)
        self.h2f = nn.Linear(hidden_size, factor_size, bias=False)
        self.f2h = nn.Linear(factor_size, hidden_size, bias=False)
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2o = nn.Linear(hidden_size, output_size, bias=True)
        self.gain = 5/3
        self.params = [self.i2f, self.h2f, self.f2h, self.i2h, self.h2o]
        self.small_weights = small_weights
        self.init_params()

    def forward(self, x_t,  h_tm1):
        # zTU = a.t @ self.i2ha.weight
        # Vx = self.i2hb.weight @ b
        # zTWx = a.t @ self.i2hab.weight @ b
        # hidden = zTU + Vx + zTWx + self.i2h.bias
        # output = self.h2o.weight
        W_fx = self.i2f.weight
        W_fh = self.h2f.weight
        W_hf = self.f2h.weight
        W_hx = self.i2h.weight
        W_oh = self.h2o.weight
        b_o = self.h2o.bias

        # left = torch.diag(W_fx @ x_t.t())
        left = torch.diag(x_t @ W_fx.t())
        # right = W_fh @ h_tm1.t()
        right = h_tm1 @ W_fh.t()
        f_t = left * right
        # f_t = left @ right
        h_t = torch.tanh(f_t @ W_hf.t() + x_t @ W_hx.t())
        o_t = h_t @ W_oh + b_o
        return o_t, h_t

    def init_params(self):
        for par in self.params:
            if self.small_weights:
                nn.init.normal_(par.weight, mean=0, std=0.1)
            else:
                nn.init.xavier_uniform_(par.weight, gain=self.gain)

    def initHidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)


class MultiplicativeLayer(nn.Module):
    def __init__(self, z_size, x_size, out_size, small_weights):
        super().__init__()
        self.U = nn.Linear(z_size, out_size, bias=True)
        self.V = nn.Linear(x_size, out_size, bias=False)
        self.W = nn.Parameter(torch.zeros((z_size, out_size, x_size)))
        self.params = [self.U.weight, self.V.weight, self.W]
        self.small_weights = small_weights
        self.init_params()

    def init_params(self):
        for par in self.params:
            if self.small_weights:
                nn.init.normal_(par, mean=0, std=0.1)
            else:
                nn.init.kaiming_uniform_(par, a=math.sqrt(5), nonlinearity='leaky_relu')


    def forward(self, x, z):
        # zTU = z.t @ self.U
        # zTU = self.U(z)
        # # Vx = self.V.weight @ x
        # Vx = self.V(x)
        # zTWx = z @ self.W @ x.t()
        # # z.unsqueeze(1)
        # # test = torch.tensordot(torch.tensordot(z, self.W), x.t())
        # out = zTU + Vx + zTWx
        zTW = torch.einsum('ij,jkl->ikl', z, self.W)
        # zz = z.unsqueeze(2)
        # torch.einsum('ijk,lk->ilj', self.W, z)
        # np.einsum('ijk,lk->ilj', self.W, z)
        # tensordot(a2D,a3D,((-1,),(-1,))).transpose(1,0,2)
        W_prime =  zTW + self.V.weight
        b_prime = self.U(z)

        W_primex = torch.einsum('ij,ikj->ik', x, W_prime)
        y = W_primex + b_prime
        return y