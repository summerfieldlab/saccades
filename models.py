# %%
import numpy as np
import math
import torch
from torch import nn
from scipy.stats import special_ortho_group, multivariate_normal
from prettytable import PrettyTable
from modules import RNN, ConvNet, MultRNN, MultiplicativeLayer, SparseLinear
import ventral_models as vmod
from skimage.transform import warp_polar, rotate


def choose_model(config, model_dir):
    # Prepare model arguments
    model_type = config.model_type
    device = config.device
    max_num = config.max_num
    no_symbol = True if config.shape_input == 'parametric' else False
    drop = config.dropout
    n_classes = max_num - config.min_num + 1
    if 'unique' in config.challenge:
        n_classes = 3
    elif 'distract' in config.challenge and config.target_type == 'all':
        n_classes = n_classes + 2

    n_shapes = 2 if config.same and config.sort else 25 # 20 or 25
    if config.ventral is not None:
        ventral = model_dir + '/ventral/' + config.ventral
    else:
        ventral = None
    finetune = True if 'finetune' in model_type else False
    whole_im = True if 'whole' in model_type else False
    height, width = config.height, config.width
    xy_sz = height*width if config.place_code else 2
    sigmoid = False if config.use_loss == 'num' else True
    mod_args = {'h_size': config.h_size, 'act': config.act,
                'detach': config.detach, 'format':config.shape_input,
                'n_classes':n_classes, 'dropout': drop, 'grid': config.grid,
                'n_shapes':n_shapes, 'ventral':ventral, 'train_on':config.train_on,
                'finetune': finetune, 'device':device, 'sort':config.sort,
                'no_pretrain': config.no_pretrain, 'whole':whole_im,
                'n_glimpses': config.n_glimpses, 'xy_sz':xy_sz, 'mult':config.mult, 
                'pass_penult':config.pass_penult, 'sigmoid':sigmoid}
    if 'par' in model_type:# == 'rnn_classifier_par':
        # Model with two parallel streams at the level of the map. Only one
        # stream is optimized to match the map. The other of the same size
        # is free, only influenced by the num loss.
        mod_args['parallel'] = True
    hidden_size = mod_args['h_size']
    output_size = mod_args['n_classes'] #+ 1
    shape_format = mod_args['format']
    train_on = mod_args['train_on']
    grid = mod_args['grid']
    n_shapes = mod_args['n_shapes']
    # grid_to_im_shape = {3:[27, 24], 6:[48, 42], 9:[69, 60]}
    # height, width = grid_to_im_shape[grid]
    map_size = grid**2
    
    if 'symbolic' in shape_format:
        sh_sz = n_shapes#20#25
        if config.sort and config.same:
            sh_sz = 2
        else:
            sh_sz = n_shapes#20#25
    elif 'pixel' in shape_format:
        sh_sz = 2 if '+' in shape_format else 1
    else:
        sh_sz = width * height
    if '2channel' in shape_format:
        sh_sz *= 2
    in_sz = xy_sz if train_on=='xy' else sh_sz if train_on =='shape' else sh_sz + xy_sz

    # Initialize the selected model class
    if 'ventral' in model_type:
        model = PretrainedVentral(sh_sz, hidden_size, map_size, output_size, **mod_args).to(device)
    elif 'mlp' in model_type:
        in_size = height * width
        model = FeedForward(in_size, hidden_size, map_size, output_size, **mod_args).to(device)
    elif 'rnn_classifier' in model_type:
        if '2stream' in model_type:
            model = RNNClassifier2stream(sh_sz, hidden_size, map_size, output_size, **mod_args).to(device)
    elif model_type == 'recurrent_control':
        in_sz = height * width
        model = RNNClassifier2stream(in_sz, hidden_size, map_size, output_size, **mod_args).to(device)
    elif 'cnn' in model_type:
        dropout = mod_args['dropout']
        if model_type == 'bigcnn':
            mod_args['big'] = True
            model = ConvNet(width, height, map_size, output_size, **mod_args).to(device)
        else:
            model = ConvNet(width, height, map_size, output_size, **mod_args).to(device)
    else:
        print(f'Model type {model_type} not implemented. Exiting')
        exit()
    # if small_weights:
    #     model.init_small()  # not implemented yet
    print('Params to learn:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"\t {name} {param.shape}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total number of trainable model params: {total_params}')

    def count_parameters(model):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params+=params
        print(table)
        print(f"Total Trainable Params: {total_params}")
        return total_params

    count_parameters(model)
    return model


class FeedForward(nn.Module):
    def __init__(self, in_size, hidden_size, map_size, output_size, **kwargs):
        super().__init__()
        self.output_size = output_size
        drop = kwargs['dropout'] if 'dropout' in kwargs.keys() else 0
        self.sig = kwargs['sigmoid']
        self.layer1 = nn.Linear(in_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.layer4 = nn.Linear(hidden_size, hidden_size)
        self.layer5 = nn.Linear(hidden_size, map_size)
        self.layer6 = nn.Linear(map_size, output_size)
        self.drop_layer = nn.Dropout(p=drop)
        self.LReLU = nn.LeakyReLU(0.1)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.LReLU(self.layer1(x))
        x = self.LReLU(self.layer2(x))
        x = self.LReLU(self.layer3(x))
        x = self.LReLU(self.layer4(x))
        x = self.drop_layer(x)
        map = self.layer5(x) # want to return the map without any nonlinearity applied because that is what BCEWithLogitsLoss is expecting
        if self.sig:
            map_to_pass_on = self.Sigmoid(map) # Only apply sigmoid if the map loss is being optimised, otherwise it makes learning less stable for no reason, at least in RNNs. Maybe less so here.
        else:
            map_to_pass_on = self.LReLU(map)
        out = self.layer6(map)
        return out, map, x


class PretrainedVentral(nn.Module):
    """Used for the distractor task. Has the main dual-stream module as rnn."""
    def __init__(self, pix_size, hidden_size, map_size, output_size, **kwargs):
        super().__init__()
        self.output_size = output_size
        self.train_on = kwargs['train_on']
        ventral_file = kwargs['ventral']
        self.whole_im = kwargs['whole']
        self.gate = kwargs['gate'] if 'gate' in kwargs.keys() else False
        self.ce = True if 'loss-ce' in ventral_file else False
        self.finetune = kwargs['finetune']
        self.sort = kwargs['sort']
        self.pass_penult =  kwargs['pass_penult']
        no_pretrain = kwargs['no_pretrain']
        penult_size = 10#8
        self.xy_size = kwargs['xy_sz'] if 'xy_sz' in kwargs.keys() else 2
        if self.train_on == 'xy':
            shape_rep_len = 0
        elif self.pass_penult:
            shape_rep_len = penult_size
        else:
            shape_rep_len = 2
            
        if self.finetune:
            drop = 0.4
        else:
            drop = 0
        ventral_output_size = 2
        # output_size = 6
        if 'mlp' in ventral_file :#or 'logpolar' in ventral_file:
            layer_width = 1024
            # self.ventral = vmod.MLP(pix_size, 1024, 3, ventral_output_size, drop=drop)
            self.ventral = vmod.BasicMLP(pix_size, layer_width, penult_size, ventral_output_size, drop=drop)
            self.cnn = False
        elif 'cnn' in ventral_file:
            self.cnn = True
            if self.whole_im or 'logpolar' in ventral_file:
                self.width = 42; self.height = 48
            else:
                self.width = 6; self.height = 6
            self.ventral = vmod.ConvNet(self.width, self.height, penult_size, ventral_output_size, dropout=drop)
        if not no_pretrain:
            print('Loading saved ventral model parameters...')
            try:  # Try loading the whole model first, this is the way to go in case we make changes to the ventral model
                self.ventral = torch.load(ventral_file)
                if hasattr(self.ventral, 'penult_size'):
                    assert self.ventral.penult_size == penult_size
            except:  # Otherwise just load the state variables. Assumes the same architecture as initialized above.
                self.ventral.load_state_dict(torch.load(ventral_file))

        self.rnn = RNNClassifier2stream(shape_rep_len, hidden_size, map_size, output_size, **kwargs)
        self.initHidden = self.rnn.initHidden
        self.Softmax = nn.Softmax(dim=1)

    def forward(self, x, hidden):
        if self.train_on == 'shape':
            pix = x
        elif self.train_on == 'xy':
            xy = x
        else:
            xy = x[:, :self.xy_size]  # xy coords are first two input features normally, unless place code
            pix = x[:, self.xy_size:]
        if self.cnn and not self.whole_im and self.train_on != 'xy':
            n, p = pix.shape
            pix = pix.view(n, 1, self.height, self.width) # REALLY IMPORTANT THAT THIS IS CORRECT, won't throw error when wrong
        if self.train_on != 'xy':
            if not self.finetune:
                with torch.no_grad():
                    shape_pred, penult_ven = self.ventral(pix) # for BCE experiment
                    if self.pass_penult:
                        shape_rep = penult_ven.detach().clone()
                    else:
                        shape_rep = shape_pred[:, :2].detach().clone()
                        if self.ce:
                            shape_rep = self.Softmax(shape_rep)            

            else:
                shape_pred, penult_ven = self.ventral(pix)
                if self.pass_penult:
                    shape_rep = penult_ven
                else:
                    shape_rep = shape_pred[:, :2]
                    if self.ce:
                            shape_rep = self.Softmax(shape_rep)

            if self.train_on == 'both':
                x = torch.cat((xy, shape_rep), dim=1)
            elif self.train_on == 'shape':
                x = shape_rep
        elif self.train_on == 'xy':
            x = xy
            shape_pred = None
        
        num, pix, map_, hidden, premap, penult = self.rnn(x, hidden)
        return num, shape_pred, map_, hidden, premap, penult

class RNNClassifier2stream(nn.Module):
    """Main dual-stream network. 
    
    Used for simple counting as is and as part of PretrainVentral for the
    distractor task."""
    def __init__(self, pix_size, hidden_size, map_size, output_size, **kwargs):
        super().__init__()
        self.train_on = kwargs['train_on']
        self.output_size = output_size
        n_shapes = kwargs['n_shapes'] if 'n_shapes' in kwargs.keys() else 20
        self.act = kwargs['act'] if 'act' in kwargs.keys() else None
        self.detach = kwargs['detach'] if 'detach' in kwargs.keys() else False
        drop = kwargs['dropout'] if 'dropout' in kwargs.keys() else 0
        self.par = kwargs['parallel'] if 'parallel' in kwargs.keys() else False
        self.xy_size = kwargs['xy_sz'] if 'xy_sz' in kwargs.keys() else 2
        self.mult = kwargs['mult'] if 'mult' in kwargs.keys() else False
        self.sig = kwargs['sigmoid'] if 'sigmoid' in kwargs.keys() else False
        if self.mult:
            embedding_size = 64
        else:
            embedding_size = hidden_size
            self.pix_embedding = nn.Linear(pix_size, embedding_size//2)
            self.xy_embedding = nn.Linear(self.xy_size, embedding_size//2)
        if self.train_on == 'both':
            if self.mult:
                self.joint_embedding = nn.Linear(self.xy_size*pix_size, embedding_size)
            else:
                self.joint_embedding = nn.Linear(embedding_size, embedding_size)
        else:
            self.joint_embedding = nn.Linear(embedding_size//2, embedding_size)

        self.rnn = RNN(embedding_size, hidden_size, hidden_size, self.act)
        self.drop_layer = nn.Dropout(p=drop)
        self.map_readout = nn.Linear(hidden_size, map_size)
        if self.par:
            self.notmap = nn.Linear(hidden_size, map_size)
            self.num_readout = nn.Linear(map_size * 2, output_size, bias=False)
        else:
            self.num_readout = nn.Linear(map_size, output_size, bias=False)

        self.initHidden = self.rnn.initHidden
        self.sigmoid = nn.Sigmoid()
        self.LReLU = nn.LeakyReLU(0.1)

    def forward(self, x, hidden):
        if self.train_on == 'both':
            xy = x[:, :self.xy_size]  # xy coords are first two input features typically unless place code
            pix = x[:, self.xy_size:]
            
            if not self.mult:
                xy = self.LReLU(self.xy_embedding(xy))
                pix = self.LReLU(self.pix_embedding(pix))
                combined = torch.cat((xy, pix), dim=-1)
            else:
                combined = torch.einsum('ij,ik->ijk', xy, pix).reshape(-1, xy.shape[1]*pix.shape[1])          
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
        # If not including the map loss term in the optimized objective function, this sigmoid is unncessary and 
        # contributes to vanishing gradients. Therefore, when use_loss == num, replace with LReLu.
        if self.sig:
            sig = self.sigmoid(map_)
        else:
            sig = self.LReLU(map_)
        if self.detach:
            map_to_pass_on = torch.round(sig.detach()).clone()
        else:
            map_to_pass_on = sig

        # penult = self.LReLU(self.after_map(map_to_pass_on))
        # function fom map to number should be linear so best to omit notlinearity, although because you allready applied sigmoid, lrelu wouldn't do anything anyway
        # penult = self.after_map(map_to_pass_on) # this extra layer probably isn't helping with anything and just increases the number of params
        if self.par:
            # Two parallel layers, one to be a map, the other not
            notmap = self.notmap(x)
            penult = torch.cat((map_to_pass_on, notmap), dim=1)
        else:
            penult = map_to_pass_on
        num = self.num_readout(penult)
        return num, pix, map_, hidden, x, penult


