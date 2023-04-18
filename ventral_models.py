import numpy as np
import math
import torch
from torch import nn
from scipy.stats import special_ortho_group
from skimage.transform import warp_polar
from utils import Timer
# from modules import RNN, MultRNN, MultiplicativeLayer

class LogPolarBasicMLP(nn.Module):
    def __init__(self, input_size, layer_width, penult_size, output_size, device, drop=0.5):
        super().__init__()
        self.device = device
        self.imsize = (48, 42)
        self.layer0 = nn.Linear(input_size, layer_width)
        self.layer1 = nn.Linear(layer_width, layer_width)
        self.layer2 = nn.Linear(layer_width, layer_width)
        self.drop_layer = nn.Dropout(p=drop)
        self.layer3 = nn.Linear(layer_width, 100)
        self.layer4 = nn.Linear(100, penult_size)
        self.out = nn.Linear(penult_size, output_size)
        self.LReLU = nn.LeakyReLU(0.1)

    def warp(self, batch_of_img, xx, yy):
        nex, h, w = batch_of_img.shape
        warped_batch = np.zeros((nex, h*w))
        for i, (img, x, y) in enumerate(zip(batch_of_img, xx, yy)):
            warped = warp_polar(torch.squeeze(img), scaling='log', output_shape=self.imsize, center=(y.numpy(), x.numpy()), mode='edge')
            warped_batch[i] = warped.flatten()
        warped_batch = torch.from_numpy(warped_batch).float().to(self.device)
        return warped_batch

    def forward(self, im, xx, yy):
        # warped = warp_polar(torch.squeeze(im), scaling='log', output_shape=self.imsize, center=(yy.numpy(), xx.numpy()), mode='edge')
        # warped = torch.from_numpy(warped.flatten()).unsqueeze(0).to(self.device)
        # timer = Timer()
        warped = self.warp(im, xx, yy)
        # timer.stop_timer()
        x = self.LReLU(self.layer0(warped))
        x = self.LReLU(self.layer1(x))
        x = self.LReLU(self.layer2(x))
        x = self.drop_layer(x)
        x = self.LReLU(self.layer3(x))
        x = self.LReLU(self.layer4(x))
        pred = self.out(x)
        return pred, x

class BasicMLP(nn.Module):
    def __init__(self, input_size, layer_width, penult_size, output_size, drop=0.5):
        super().__init__()
        self.penult_size = penult_size
        self.layer0 = nn.Linear(input_size, layer_width)
        self.layer1 = nn.Linear(layer_width, layer_width)
        self.layer2 = nn.Linear(layer_width, layer_width)
        self.drop_layer = nn.Dropout(p=drop)
        self.layer3 = nn.Linear(layer_width, 100)
        self.layer4 = nn.Linear(100, penult_size)
        self.out = nn.Linear(penult_size, output_size)
        self.LReLU = nn.LeakyReLU(0.1)


    def forward(self, x):
        x = self.LReLU(self.layer0(x))
        x = self.LReLU(self.layer1(x))
        x = self.LReLU(self.layer2(x))
        x = self.drop_layer(x)
        x = self.LReLU(self.layer3(x))
        x = self.LReLU(self.layer4(x))
        pred = self.out(x)
        return pred, x
    

class MLP(nn.Module):
    def __init__(self, input_size, layer_width, penult_size, output_size, drop=0.5):
        super().__init__()
        self.layer0 = nn.Linear(input_size, layer_width)
        self.BN0 = torch.nn.BatchNorm1d(layer_width)
        self.layer1 = nn.Linear(layer_width, layer_width)
        self.BN1 = torch.nn.BatchNorm1d(layer_width)
        self.layer2 = nn.Linear(layer_width, layer_width)
        self.BN2 = torch.nn.BatchNorm1d(layer_width)
        self.drop_layer = nn.Dropout(p=drop)
        self.layer3 = nn.Linear(layer_width, 100)
        self.BN3 = torch.nn.BatchNorm1d(100)
        self.layer4 = nn.Linear(100, penult_size)
        self.BN4 = torch.nn.BatchNorm1d(penult_size)
        # self.layers = [self.layer1, self.layer2, self.layer3]
        self.out = nn.Linear(penult_size, output_size)
        self.LReLU = nn.LeakyReLU(0.1)


    def forward(self, x):
        x = self.LReLU(self.BN0(self.layer0(x)))
        x = self.LReLU(self.BN1(self.layer1(x)))
        x = self.LReLU(self.BN2(self.layer2(x)))
        x = self.drop_layer(x)
        x = self.LReLU(self.BN3(self.layer3(x)))
        x = self.LReLU(self.BN4(self.layer4(x)))
        # for layerid in range(self.n_layers):
        #     x = self.LReLU(self.layers[layerid](x))
        pred = self.out(x)
        # pred = torch.clamp(pred, -1e6, 1e6)
        return pred, x


class old_MLP(nn.Module):
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
        # x = self.LReLU(self.BN4(self.layer4(x)))
        pred = self.out(x)
        # pred = torch.clamp(pred, -1e6, 1e6)
        return pred, x

class ConvNet(nn.Module):
    def __init__(self, width, height, penult_size, output_size, **kwargs):
        super().__init__()
        grid = kwargs['grid'] if 'grid' in kwargs.keys() else 9
        big = kwargs['big'] if 'big' in kwargs.keys() else False
        dropout = kwargs['dropout'] if 'dropout' in kwargs.keys() else 0.0
        # Larger version
        # if big:
        #     self.kernel1_size = 2
        #     self.cnn1_nchannels_out = 56
        #     self.poolsize = 2
        #     self.kernel2_size = 2
        #     self.cnn2_nchannels_out = 56
        #     self.kernel3_size = 2
        #     self.cnn3_nchannels_out = 56

            # self.kernel1_size = 3
            # self.cnn1_nchannels_out = 128
            # self.poolsize = 2
            # self.kernel2_size = 2
            # self.cnn2_nchannels_out = 256
            # self.kernel3_size = 2
            # self.cnn3_nchannels_out = 32
        # elif grid==3: # Smaller version
        self.kernel1_size = (2, 1)
        self.cnn1_nchannels_out = 33
        # self.poolsize = 2
        self.kernel2_size = (1, 2)
        self.cnn2_nchannels_out = 33
        # self.kernel3_size = 2
        # self.cnn3_nchannels_out = 33
        # elif grid==6: # Smaller version
        #     self.kernel1_size = 3
        #     self.cnn1_nchannels_out = 33
        #     self.poolsize = 2
        #     self.kernel2_size = 2
        #     self.cnn2_nchannels_out = 30
        #     self.kernel3_size = 2
        #     self.cnn3_nchannels_out = 20
        # elif grid==9:
        #     self.kernel1_size = 3
        #     self.cnn1_nchannels_out = 33
        #     self.poolsize = 2
        #     self.kernel2_size = 2
        #     self.cnn2_nchannels_out = 20
        #     self.kernel3_size = 2
        #     self.cnn3_nchannels_out = 20
        self.output_size = output_size
        self.LReLU = nn.LeakyReLU(0.1)

        # Default initialization is init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # (Also assumes leaky Relu for gain)
        # which is appropriate for all layers here
        self.conv1 = nn.Conv2d(1, self.cnn1_nchannels_out, self.kernel1_size)    # (NChannels_in, NChannels_out, kernelsize)
        self.BN0 = torch.nn.BatchNorm2d(self.cnn1_nchannels_out)
        # torch.nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        # self.pool = nn.MaxPool2d(self.poolsize, self.poolsize)     # kernel height, kernel width
        self.conv2 = nn.Conv2d(self.cnn1_nchannels_out, self.cnn2_nchannels_out, self.kernel2_size)   # (NChannels_in, NChannels_out, kernelsize)
        self.BN1 = torch.nn.BatchNorm2d(self.cnn2_nchannels_out)
        # self.conv3 = nn.Conv2d(self.cnn2_nchannels_out, self.cnn3_nchannels_out, self.kernel3_size)
        # torch.nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        # track the size of the cnn transformations
        # self.cnn2_width_out = ((width - self.kernel1_size+1) - self.kernel2_size + 1) // self.poolsize
        # self.cnn2_height_out = ((height - self.kernel1_size+1) - self.kernel2_size + 1) // self.poolsize
        # self.cnn3_width_out = (((width - self.kernel1_size+1) - self.kernel2_size + 1)  // self.poolsize) - self.kernel3_size+1
        # self.cnn3_height_out = (((height - self.kernel1_size+1) - self.kernel2_size + 1)  // self.poolsize) - self.kernel3_size+1
        self.cnn2_width_out = ((width - self.kernel1_size[0]+1) - self.kernel2_size[0] + 1)
        self.cnn2_height_out = ((height - self.kernel1_size[1]+1) - self.kernel2_size[1] + 1)

        # FC layers
        self.fc1_size = 100
        self.fc2_size = penult_size
        self.fc1 = nn.Linear(int(self.cnn2_nchannels_out * self.cnn2_width_out * self.cnn2_height_out), self.fc1_size)  # size input, size output
        # self.fc1_size = 120
        
        # import pdb; pdb.set_trace()
        # self.fc1 = nn.Linear(int(self.cnn3_nchannels_out * self.cnn3_width_out * self.cnn3_height_out), self.fc1_size)  # size input, size output
        self.fc2 = nn.Linear(self.fc1_size, self.fc2_size)
        self.fc3 = nn.Linear(self.fc2_size, output_size)

        # Dropout
        self.drop_layer = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.BN0(self.LReLU(self.conv1(x)))
        # x = self.pool(self.LReLU(self.conv2(x)))
        x = self.BN1(self.LReLU(self.conv2(x)))
        # x = self.LReLU(self.conv3(x))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3]) # this reshapes the tensor to be a 1D vector, from whatever the final convolutional layer output
        # add dropout before fully connected layers, widest part of network
        x = self.drop_layer(x)
        x = self.LReLU(self.fc1(x))
        fc2 = self.LReLU(self.fc2(x))
        out = self.fc3(fc2)
        return out, fc2
