# %%
import numpy as np
import math
import torch
from torch import nn
from scipy.stats import special_ortho_group, multivariate_normal
from models import RNN, RNN2, MultRNN, MultiplicativeLayer
import ventral_models as vmod
TRAIN_SHAPES = [0,  2,  4,  5,  8,  9, 14, 15, 16]
imsize = (48, 42)
image_template = np.zeros((48, 42))
GRID = np.linspace(0.1, 0.9, 6)

def soft_argmax(voxels, device):
	"""
    Copied from https://github.com/Fdevmsy/PyTorch-Soft-Argmax
	Arguments: voxel patch in shape (batch_size, channel, H, W, depth)
	Return: 3D coordinates in shape (batch_size, channel, 3)
	"""
	assert voxels.dim()==5
	# alpha is here to make the largest element really big, so it
	# would become very close to 1 after softmax
	alpha = 1000.0 
	N,C,H,W,D = voxels.shape
	soft_max = nn.functional.softmax(voxels.view(N,C,-1)*alpha,dim=2)
	soft_max = soft_max.view(voxels.shape)
	indices_kernel = torch.arange(start=0,end=H*W*D, device=device).unsqueeze(0)
	indices_kernel = indices_kernel.view((H,W,D))
	conv = soft_max*indices_kernel
	indices = conv.sum(2).sum(2).sum(2)
	z = indices%D
	y = (indices/D).floor()%W
	x = (((indices/D).floor())/W).floor()%H
	coords = torch.stack([x,y,z],dim=2)
	return coords

# %%
class Glimpsing(nn.Module):
    def __init__(self, salience_size, hidden_size, map_size, output_size, **kwargs):
        super().__init__()
        self.device = kwargs['device']
        self.height, self.width = salience_size
        self.glimpse_width = 6
        self.half_glim = self.glimpse_width // 2
        pix_size = self.glimpse_width**2
        self.where_look_x = nn.Linear(hidden_size + self.height*self.width, self.width)
        self.where_look_y = nn.Linear(hidden_size + self.height*self.width, self.height)
        
        self.Softmax = nn.LogSoftmax(dim=1)
        self.pretrained_ventral = PretrainedVentral(pix_size, hidden_size, map_size, output_size, **kwargs)
        self.initHidden = self.pretrained_ventral.initHidden
    
    def forward(self, image, saliency, hidden):
        pre_glimpse = torch.cat((hidden, saliency.view(hidden.shape[0], -1)), dim=1)
        x = self.where_look_x(pre_glimpse)
        N, H = x.shape
        import pdb; pdb.set_trace()
        x = soft_argmax(x.view(N, 1, H, 1, 1), self.device)[:,:,0]  # Take first dim because 1D argmax
        y = self.where_look_y(pre_glimpse)
        N, H = y.shape
        y = soft_argmax(y.view(N, 1, H, 1, 1), self.device)[:,:,0] # Take first dim because 1D argmax
        glimpse_contents = self.extract_glimpse(x, y, image)
        input_ = torch.cat((x, y, glimpse_contents), dim=1)
        num, shape, map_, hidden = self.pretrained_ventral(input_, hidden)
        return num, shape, map_, hidden
    
    def extract_glimpse(self, x, y, image):
        """
        In the spatial (4-D) case, for input with shape (N,C,Hin,Win) and grid with shape (N,Hout,Wout,2)
        """

        hg = self.half_glim
        import pdb;pdb.set_trace()
        N, nch, Hin, Win = image.shape
        x = torch.stack((x-2.5, x-1.5, x-0.5, x+0.5, x+1.5, x+2.5), dim=-1)
        x -= self.width/2
        x /= self.width/2
        y = torch.stack((y-2.5, y-1.5, y-0.5, y+0.5, y+1.5, y+2.5), dim=-1)
        y -= self.height/2
        y /= self.height/2
        grid = torch.stack((x, y)).view(N, 6, 6, 2)

        # theta = 
        # size = (N, 1, self.glimpse_width, self.glimpse_width)
        # grid = nn.functional.affine_grid(theta, size)
        glimpse_pixels = nn.functional.grid_sample(image.view(N, 1, Hin, Win), grid)
        # glimpse_pixels = image[y - hg: y + hg, x - hg: x + hg].flatten()
        return glimpse_pixels.flatten()



class FeedForward(nn.Module):
    def __init__(self, in_size, hidden_size, map_size, output_size, **kwargs):
        super().__init__()
        self.output_size = output_size
        drop = kwargs['dropout'] if 'dropout' in kwargs.keys() else 0
        self.layer1 = nn.Linear(in_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.layer4 = nn.Linear(hidden_size, hidden_size)
        self.layer5 = nn.Linear(hidden_size, map_size)
        self.layer6 = nn.Linear(map_size, output_size)
        self.drop_layer = nn.Dropout(p=drop)
        self.LReLU = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.LReLU(self.layer1(x))
        x = self.LReLU(self.layer2(x))
        x = self.LReLU(self.layer3(x))
        x = self.LReLU(self.layer4(x))
        x = self.drop_layer(x)
        map = self.LReLU(self.layer5(x))
        out = self.layer6(map)
        return out, map


class PretrainedVentral(nn.Module):
    def __init__(self, pix_size, hidden_size, map_size, output_size, **kwargs):
        super().__init__()
        self.output_size = output_size
        self.train_on = kwargs['train_on']
        ventral_file = kwargs['ventral']
        if 'loss-ce' in ventral_file:
            self.ce = True
        self.finetune = kwargs['finetune']
        self.sort = kwargs['sort']
        no_pretrain = kwargs['no_pretrain']
        if self.train_on == 'xy':
            shape_rep_len = 0
        elif self.sort:
            shape_rep_len = 2
        else:
            # shape_rep_len = 8
            shape_rep_len = 4
            # shape_rep_len = len(TRAIN_SHAPES)
        if self.finetune:
            drop = 0.25
        else:
            drop = 0
        ventral_output_size = 25
        output_size = 6
        if 'mlp' in ventral_file:
            self.ventral = vmod.MLP(pix_size, 1024, 3, output_size, drop=drop)
        elif 'cnn' in ventral_file:
            self.cnn = True
            width = 6; height = 6
            penult_size = 4
            self.ventral = vmod.ConvNet(width, height, penult_size, ventral_output_size, dropout=drop)
        if not no_pretrain:
            print('Loading saved ventral model parameters...')
            self.ventral.load_state_dict(torch.load(ventral_file))
        # self.ventral.eval()
        # self.rnn = RNNClassifier2stream(shape_rep_len+100, hidden_size, map_size, output_size, **kwargs)
        self.rnn = RNNClassifier2stream(shape_rep_len, hidden_size, map_size, output_size, **kwargs)
        self.initHidden = self.rnn.initHidden
        self.Softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):
        if self.train_on == 'shape':
            pix = x
        elif self.train_on == 'xy':
            xy = x
        else:
            xy = x[:, :2]  # xy coords are first two input features
            pix = x[:, 2:]
        if self.cnn:
            n, p = pix.shape
            pix = pix.view(n, 1, 6, 6)
        if self.train_on != 'xy':
            if not self.finetune:
                with torch.no_grad():
                    # shape_rep = self.ventral(pix)[:, 1:3] # ignore 0th, take 1st and 2nd column
                    shape_rep, penult = self.ventral(pix) # for BCE experiment
                    # shape_rep = torch.sigmoid(shape_rep[:, TRAIN_SHAPES])
                    # the 0th output was trained with  BCEWithLogitsLoss so need to apply sigmoid
                    # shape_rep[:, 0] = torch.sigmoid(shape_rep[:, 0])
                    # shape_rep = torch.concat((shape_rep[:, :2], penult), dim=1)
                    if self.sort:
                        shape_rep = shape_rep[:, :2].detach().clone()
                        if self.ce:
                            shape_rep = self.Softmax(shape_rep)
                    else:
                        shape_rep = penult.detach().clone()

            else:
                shape_rep, penult = self.ventral(pix)
                # shape_rep, _ = self.ventral(pix)
                if self.sort:
                    shape_rep = shape_rep[:, :2]
                    if self.ce:
                            shape_rep = self.Softmax(shape_rep)
                else:
                    shape_rep = penult

            if self.train_on == 'both':
                # x = torch.concat((xy, shape_rep.detach().clone()), dim=1)
                x = torch.concat((xy, shape_rep), dim=1)
            elif self.train_on == 'shape':
                x = shape_rep
        elif self.train_on == 'xy':
            x = xy

        num, shape, map_, hidden, premap, penult = self.rnn(x, hidden)
        return num, shape, map_, hidden, premap, penult


class RNNClassifier2stream2map(nn.Module):
    def __init__(self, pix_size, hidden_size, map_size, output_size, **kwargs):
        super().__init__()
        # map_size = 9
        self.train_on = kwargs['train_on']
        self.n_out = 6
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
        self.map_dist_readout = nn.Linear(hidden_size, map_size)
        self.after_map_count = nn.Linear(map_size, map_size)
        self.after_map_dist = nn.Linear(map_size, map_size)
        self.after_map_all = nn.Linear(map_size, map_size)
        if self.par:
            self.notmap = nn.Linear(hidden_size, map_size)
            self.num_readout_count = nn.Linear(map_size * 2, output_size)
            self.num_readout_dist = nn.Linear(map_size * 2, 3)
            self.num_readout_all = nn.Linear(map_size * 2, output_size +2 )
        else:
            self.num_readout_count = nn.Linear(map_size, output_size)
            self.num_readout_dist = nn.Linear(map_size, 3)
            self.num_readout_all = nn.Linear(map_size, output_size + 2)

        self.initHidden = self.rnn.initHidden
        self.sigmoid = nn.Sigmoid()
        self.LReLU = nn.LeakyReLU(0.1)

    def forward(self, x, hidden):
        if self.train_on == 'both':
            xy = x[:,:2]  # xy coords are first two input features
            pix = x[:,2:]
            xy = self.LReLU(self.xy_embedding(xy))
            pix = self.LReLU(self.pix_embedding(pix))
            shape = self.shape_readout(pix)
            combined = torch.cat((shape, xy, pix), dim=-1)
        elif self.train_on == 'xy':
            xy = x
            xy = self.LReLU(self.xy_embedding(xy))
            combined = xy
        elif self.train_on == 'shape':
            pix = x
            pix = self.LReLU(self.pix_embedding(pix))
            shape = self.shape_readout(pix)
            combined = pix

        x = self.LReLU(self.joint_embedding(combined))
        x, hidden = self.rnn(x, hidden)
        x = self.drop_layer(x)

        map_count = self.map_readout(x)
        map_dist = self.map_dist_readout(x)
        # full map is sum of the two submaps
        map_all = torch.add(map_count, map_dist)

        sig_count = self.sigmoid(map_count)
        sig_dist = self.sigmoid(map_dist)
        sig_all = self.sigmoid(map_all)
        if self.detach:
            map_to_pass_on_count = sig_count.detach().clone()
            map_to_pass_on_dist = sig_dist.detach().clone()
            map_to_pass_on_all = sig_all.detach().clone()
        else:
            map_to_pass_on_count = sig_count
            map_to_pass_on_dist = sig_dist
            map_to_pass_on_all = sig_all

        penult_count = self.LReLU(self.after_map_count(map_to_pass_on_count))
        penult_dist = self.LReLU(self.after_map_dist(map_to_pass_on_dist))
        penult_all = self.LReLU(self.after_map_all(map_to_pass_on_all))
        # if self.par:
        #     # Two parallel layers, one to be a map, the other not
        #     notmap = self.notmap(x)
        #     penult = torch.cat((penult, notmap), dim=1)
        # num = self.num_readout(map_to_pass_on)
        num_count = self.num_readout_count(penult_count)
        num_dist = self.num_readout_dist(penult_dist)
        num_all = self.num_readout_all(penult_all)
        all_num = (num_count, num_dist, num_all)
        all_maps = (map_count, map_dist, map_all)
        return all_num, shape, all_maps, hidden

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
        self.train_on = kwargs['train_on']
        self.output_size = output_size
        n_shapes = kwargs['n_shapes'] if 'n_shapes' in kwargs.keys() else 20
        self.act = kwargs['act'] if 'act' in kwargs.keys() else None
        self.detach = kwargs['detach'] if 'detach' in kwargs.keys() else False
        drop = kwargs['dropout'] if 'dropout' in kwargs.keys() else 0
        self.par = kwargs['parallel'] if 'parallel' in kwargs.keys() else False
        self.pix_embedding = nn.Linear(pix_size, hidden_size//2)
        self.pix_embedding2 = nn.Linear(hidden_size//2, hidden_size//2)
        # self.shape_readout = nn.Linear(hidden_size, n_shapes)
        self.xy_embedding = nn.Linear(2, hidden_size//2)
        # self.joint_embedding = nn.Linear(hidden_size//2 + n_shapes, hidden_size)
        if self.train_on == 'both':
            self.joint_embedding = nn.Linear(hidden_size, hidden_size)
        else:
            self.joint_embedding = nn.Linear(hidden_size//2, hidden_size)
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
        if self.train_on == 'both':
            xy = x[:, :2]  # xy coords are first two input features
            pix = x[:, 2:]
            xy = self.LReLU(self.xy_embedding(xy))
            pix = self.LReLU(self.pix_embedding(pix))
            pix = self.LReLU(self.pix_embedding2(pix))
            # shape = self.shape_readout(pix)
            # shape_detached = shape.detach().clone()
            # combined = torch.cat((shape, xy, pix), dim=-1)
            # combined = torch.cat((shape_detached, xy), dim=-1)
            combined = torch.cat((xy, pix), dim=-1)
        elif self.train_on == 'xy':
            xy = x
            xy = self.LReLU(self.xy_embedding(xy))
            combined = xy
            pix = torch.zeros_like(xy)
        elif self.train_on == 'shape':
            pix = x
            pix = self.LReLU(self.pix_embedding(pix))
            combined = pix
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
        return num, pix, map_, hidden, x, penult


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
        device = torch.device("cuda")
        self.mock_shape_pred = torch.zeros((10,)).to(device)

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
        shape = self.mock_shape_pred
        return num, shape, map, hidden


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
