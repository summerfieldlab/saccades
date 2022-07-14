"""Train simple models on data synthesized from toy model."""
import os
import argparse
from itertools import product
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
import models_toy as mod
import toy_model_data as toy
from count_unique import Timer
# from labml_nn.hypernetworks.hyper_lstm import HyperLSTM
from hypernet import HyperLSTM
# torch.set_num_threads(15)

mom = 0.9
wd = 0
start_lr = 0.1
BATCH_SIZE = 1024
n_glimpses = 4
device = torch.device("cuda")
# device = torch.device("cpu")
criterion = nn.CrossEntropyLoss()
criterion_noreduce = nn.CrossEntropyLoss(reduction='none')
criterion_mse = nn.MSELoss()
criterion_mse_noreduce = nn.MSELoss(reduction='none')
pos_weight = torch.ones([9], device=device) * 2
criterion_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
criterion_bce_noreduce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')


def train_model(rnn, optimizer, scheduler, loaders, base_name, config):
    n_epochs = config.n_epochs
    recurrent_iterations = config.n_iters
    cross_entropy = config.cross_entropy
    nonsymbolic = config.no_symbol
    train_loader, test_loaders = loaders
    train_loss = np.zeros((n_epochs,))
    train_map_loss = np.zeros((n_epochs,))
    train_num_loss = np.zeros((n_epochs,))
    train_acc = np.zeros((n_epochs,))

    test_loss = [np.zeros((n_epochs,)) for _ in config.test_shapes]
    test_map_loss = [np.zeros((n_epochs,)) for _ in config.test_shapes]
    test_num_loss = [np.zeros((n_epochs,)) for _ in config.test_shapes]
    test_acc = [np.zeros((n_epochs,)) for _ in config.test_shapes]
    columns = ['pass count', 'correct', 'predicted', 'true', 'loss', 'num loss', 'map loss', 'epoch', 'train_shapes', 'test_shapes']

    def train_nosymbol(loader):
        rnn.train()
        correct = 0
        epoch_loss = 0
        num_epoch_loss = 0
        map_epoch_loss = 0
        for i, (xy, shape, target, locations, _) in enumerate(loader):
            rnn.zero_grad()
            input_dim, seq_len = xy.shape[0], xy.shape[1]
            new_order = torch.randperm(seq_len)
            for i in range(input_dim):
                xy[i, :, :] = xy[i, new_order, :]
                shape[i, :, :, :] = shape[i, new_order, :, :]

            hidden = rnn.initHidden(input_dim)
            hidden = hidden.to(device)
            for _ in range(recurrent_iterations):
                for t in range(n_glimpses):
                    pred_num, map, hidden = rnn(xy[:, t, :], shape[:, t, :, :], hidden)
            if cross_entropy:
                num_loss = criterion(pred_num, target)
                pred = pred_num.argmax(dim=1, keepdim=True)
            else:
                num_loss = criterion_mse(torch.squeeze(pred_num), target)
                pred = torch.round(pred_num)

            if config.use_loss == 'num':
                loss = num_loss
                map_loss_to_add = -1
            elif config.use_loss == 'map':
                map_loss = criterion_bce(map, locations)
                map_loss_to_add = map_loss.item()
                loss = map_loss
            elif config.use_loss == 'both':
                map_loss = criterion_bce(map, locations)
                map_loss_to_add = map_loss.item()
                loss = num_loss + map_loss
            loss.backward()
            nn.utils.clip_grad_norm_(rnn.parameters(), 2)
            optimizer.step()

            correct += pred.eq(target.view_as(pred)).sum().item()
            epoch_loss += loss.item()
            num_epoch_loss += num_loss.item()
            map_epoch_loss += map_loss_to_add
        scheduler.step()
        accuracy = 100. * correct/len(loader.dataset)
        epoch_loss /= len(loader)
        num_epoch_loss /= len(loader)
        map_epoch_loss /= len(loader)
        return epoch_loss, num_epoch_loss, accuracy, map_epoch_loss

    def train(loader):
        rnn.train()
        correct = 0
        epoch_loss = 0
        num_epoch_loss = 0
        map_epoch_loss = 0
        for i, (input, target, locations, _) in enumerate(loader):

            seq_len = input.shape[1]
            for i, row in enumerate(input):
                input[i, :, :] = row[torch.randperm(seq_len), :]
            input_dim = input.shape[0]

            rnn.zero_grad()

            hidden = rnn.initHidden(input_dim)
            if config.model_type is not 'hyper':
                hidden = hidden.to(device)
            # data = toy.generate_dataset(BATCH_SIZE)
            # # data.loc[0]
            # xy = torch.tensor(data['xy']).float()
            # shape = torch.tensor(data['shape']).float()
            # target = torch.tensor(data['numerosity']).long()
            for _ in range(recurrent_iterations):
                for t in range(n_glimpses):
                    pred_num, map, hidden = rnn(input[:, t, :], hidden)

            if cross_entropy:
                num_loss = criterion(pred_num, target)
                pred = pred_num.argmax(dim=1, keepdim=True)
            else:
                num_loss = criterion_mse(torch.squeeze(pred_num), target)
                pred = torch.round(pred_num)

            if config.use_loss == 'num':
                loss = num_loss
                map_loss_to_add = -1
            elif config.use_loss == 'map':
                map_loss = criterion_bce(map, locations)
                map_loss_to_add = map_loss.item()
                loss = map_loss
            elif config.use_loss == 'both':
                map_loss = criterion_bce(map, locations)
                map_loss_to_add = map_loss.item()
                loss = num_loss + map_loss
            loss.backward()
            nn.utils.clip_grad_norm_(rnn.parameters(), 2)
            optimizer.step()

            correct += pred.eq(target.view_as(pred)).sum().item()
            epoch_loss += loss.item()
            num_epoch_loss += num_loss.item()
            map_epoch_loss += map_loss_to_add
        scheduler.step()
        accuracy = 100. * correct/len(loader.dataset)
        epoch_loss /= len(loader)
        num_epoch_loss /= len(loader)
        map_epoch_loss /= len(loader)
        return epoch_loss, num_epoch_loss, accuracy, map_epoch_loss

    def test_nosymbol(loader, epoch):
        rnn.eval()
        n_correct = 0
        epoch_loss = 0
        num_epoch_loss = 0
        map_epoch_loss = 0
        test_results = pd.DataFrame(columns=columns)
        for i, (xy, shape, target, locations, pass_count) in enumerate(loader):

            input_dim = xy.shape[0]

            batch_results = pd.DataFrame(columns=columns)
            hidden = rnn.initHidden(input_dim)
            hidden = hidden.to(device)

            for _ in range(recurrent_iterations):
                if config.model_type == 'hyper':
                    pred_num, map, hidden = rnn(input, hidden)
                else:
                    for t in range(n_glimpses):
                        pred_num, map, hidden = rnn(xy[:, t, :], shape[:, t, :, :], hidden)

            if cross_entropy:
                num_loss = criterion_noreduce(pred_num, target)
                pred = pred_num.argmax(dim=1, keepdim=True)
            else:
                num_loss = criterion_mse_noreduce(torch.squeeze(pred_num), target)
                pred = torch.round(pred_num)

            if config.use_loss == 'num':
                loss = num_loss
                # map_loss = criterion_bce_noreduce(map, locations)
                map_loss_to_add = -1
            elif config.use_loss == 'map':
                map_loss = criterion_bce_noreduce(map, locations)
                map_loss_to_add = map_loss.mean().item()
                loss = map_loss
            elif config.use_loss == 'both':
                map_loss = criterion_bce_noreduce(map, locations)
                map_loss_to_add = map_loss.mean().item()
                loss = num_loss + map_loss.mean()

            correct = pred.eq(target.view_as(pred))
            batch_results['pass count'] = pass_count.detach().cpu().numpy()
            batch_results['correct'] = correct.cpu().numpy()
            batch_results['predicted'] = pred.detach().cpu().numpy()
            batch_results['true'] = target.detach().cpu().numpy()
            batch_results['loss'] = loss.detach().cpu().numpy()
            try:
                batch_results['map loss'] = map_loss.detach().cpu().numpy()
            except:
                batch_results['map loss'] = np.ones(loss.shape) * -1
            batch_results['num loss'] = num_loss.detach().cpu().numpy()
            batch_results['epoch'] = epoch
            test_results = pd.concat((test_results, batch_results))

            n_correct += pred.eq(target.view_as(pred)).sum().item()
            epoch_loss += loss.mean().item()
            num_epoch_loss += num_loss.mean().item()
            map_epoch_loss += map_loss_to_add

        accuracy = 100. * n_correct/len(loader.dataset)
        epoch_loss /= len(loader)
        num_epoch_loss /= len(loader)
        map_epoch_loss /= len(loader)
        return epoch_loss, num_epoch_loss, accuracy, map_epoch_loss, test_results

    def test(loader, epoch):
        rnn.eval()
        n_correct = 0
        epoch_loss = 0
        num_epoch_loss = 0
        map_epoch_loss = 0
        test_results = pd.DataFrame(columns=columns)
        for i, (input, target, locations, pass_count) in enumerate(loader):
            if nonsymbolic:
                xy, shape = input
                input_dim = xy.shape[0]
            else:
                input_dim = input.shape[0]
            batch_results = pd.DataFrame(columns=columns)
            hidden = rnn.initHidden(input_dim)
            if config.model_type != 'hyper':
                hidden = hidden.to(device)
            # data = toy.generate_dataset(BATCH_SIZE)
            # # data.loc[0]
            # xy = torch.tensor(data['xy']).float()
            # shape = torch.tensor(data['shape']).float()
            # target = torch.tensor(data['numerosity']).long()
            for _ in range(recurrent_iterations):
                if config.model_type == 'hyper':
                    pred_num, map, hidden = rnn(input, hidden)
                else:
                    for t in range(n_glimpses):
                        if nonsymbolic:
                            pred_num, map, hidden = rnn(xy[:, t, :], shape[:, t, :, :], hidden)
                        else:
                            pred_num, map, hidden = rnn(input[:, t, :], hidden)
            if cross_entropy:
                num_loss = criterion_noreduce(pred_num, target)
                pred = pred_num.argmax(dim=1, keepdim=True)
            else:
                num_loss = criterion_mse_noreduce(torch.squeeze(pred_num), target)
                pred = torch.round(pred_num)

            if config.use_loss == 'num':
                loss = num_loss
                # map_loss = criterion_bce_noreduce(map, locations)
                map_loss_to_add = -1
            elif config.use_loss == 'map':
                map_loss = criterion_bce_noreduce(map, locations)
                map_loss_to_add = map_loss.mean().item()
                loss = map_loss
            elif config.use_loss == 'both':
                map_loss = criterion_bce_noreduce(map, locations)
                map_loss_to_add = map_loss.mean().item()
                loss = num_loss + map_loss.mean()

            correct = pred.eq(target.view_as(pred))
            batch_results['pass count'] = pass_count.detach().cpu().numpy()
            batch_results['correct'] = correct.cpu().numpy()
            batch_results['predicted'] = pred.detach().cpu().numpy()
            batch_results['true'] = target.detach().cpu().numpy()
            batch_results['loss'] = loss.detach().cpu().numpy()
            try:
                batch_results['map loss'] = map_loss.detach().cpu().numpy()
            except:
                batch_results['map loss'] = np.ones(loss.shape) * -1
            batch_results['num loss'] = num_loss.detach().cpu().numpy()
            batch_results['epoch'] = epoch
            test_results = pd.concat((test_results, batch_results))

            n_correct += pred.eq(target.view_as(pred)).sum().item()
            epoch_loss += loss.mean().item()
            num_epoch_loss += num_loss.mean().item()
            map_epoch_loss += map_loss_to_add

        accuracy = 100. * n_correct/len(loader.dataset)
        epoch_loss /= len(loader)
        num_epoch_loss /= len(loader)
        map_epoch_loss /= len(loader)
        return epoch_loss, num_epoch_loss, accuracy, map_epoch_loss, test_results

    test_results = pd.DataFrame(columns=columns)
    for ep in range(n_epochs):
        epoch_timer = Timer()
        if nonsymbolic:
            train_f = train_nosymbol
            test_f = test_nosymbol
        else:
            train_f = train
            test_f = test
        # Train
        epoch_tr_loss, epoch_tr_num_loss, tr_accuracy, epoch_tr_map_loss = train_f(train_loader)
        train_loss[ep] = epoch_tr_loss
        train_num_loss[ep] = epoch_tr_num_loss
        train_acc[ep] = tr_accuracy
        train_map_loss[ep] = epoch_tr_map_loss
        # Test
        for ts, (test_loader, test_shapes) in enumerate(zip(test_loaders, config.test_shapes)):
            epoch_te_loss, epoch_te_num_loss, te_accuracy, epoch_te_map_loss, epoch_df = test_f(test_loader, ep)
            epoch_df['train shapes'] = str(config.train_shapes)
            epoch_df['test shapes'] = str(test_shapes)

            test_results = pd.concat((test_results, epoch_df), ignore_index=True)
            test_loss[ts][ep] = epoch_te_loss
            test_num_loss[ts][ep] = epoch_te_num_loss
            test_acc[ts][ep] = te_accuracy
            test_map_loss[ts][ep] = epoch_te_map_loss
            base_name_test = base_name + f'_test-shapes-{test_shapes}'

            if not ep % 5:
                test_results['accuracy'] = test_results['correct'].astype(int)*100
                data = test_results[test_results['test shapes'] == str(test_shapes)]
                data = data[data['pass count'] < 6]
                sns.lineplot(data=data, x='epoch', y='num loss', hue='pass count')
                plt.plot(train_num_loss[:ep + 1], '--', color='black', label='training loss')
                mt = config.model_type + '-nosymbol' if nonsymbolic else config.model_type
                title = f'{mt} trainon-{config.train_on} train_shapes-{config.train_shapes} \n test_shapes-{test_shapes} useloss-{config.use_loss} noise-{config.noise_level}'
                plt.title(title)
                ylim = plt.ylim()
                plt.ylim([-0.05, ylim[1]])
                plt.grid()
                plt.savefig(f'figures/toy/test_num-loss_{base_name_test}.png', dpi=300)
                plt.close()

                sns.lineplot(data=data, x='epoch', y='map loss', hue='pass count')
                plt.plot(train_map_loss[:ep + 1], '--', color='black', label='training loss')
                plt.title(title)
                ylim = plt.ylim()
                plt.ylim([-0.05, ylim[1]])
                plt.grid()
                plt.savefig(f'figures/toy/test_map-loss_{base_name_test}.png', dpi=300)
                plt.close()
                # sns.countplot(data=test_results[test_results['correct']==True], x='epoch', hue='pass count')
                # plt.savefig(f'figures/toy/test_correct_{base_name}.png', dpi=300)
                # plt.close()

                accuracy = data.groupby(['epoch', 'pass count']).mean()
                sns.lineplot(data=accuracy, x='epoch', hue='pass count', y='accuracy')
                plt.grid()
                plt.title(title)
                plt.ylim([0, 102])
                plt.ylabel('Accuracy on number task')
                plt.savefig(f'figures/toy/accuracy_{base_name_test}.png', dpi=300)
                plt.close()

        epoch_timer.stop_timer()
        # import pdb;pdb.set_trace()
        if isinstance(test_loss, list):
            print(f'Epoch {ep}. LR={optimizer.param_groups[0]["lr"]} \t (Train/Val/Test) Loss={train_loss[ep]:.4}/{test_loss[0][ep]:.4}/{test_loss[1][ep]:.4} \t Accuracy={train_acc[ep]}%/{test_acc[0][ep]}%/{test_acc[1][ep]}%')
        else:
            print(f'Epoch {ep}. LR={optimizer.param_groups[0]["lr"]} \t (Train/Test) Loss={train_loss[ep]:.4}/{test_loss[ep]:.4}/ \t Accuracy={train_acc[ep]}%/{test_acc[ep]}%')
    results_list = [train_loss, train_acc, train_map_loss, test_loss, test_acc, test_map_loss, test_results]
    return rnn, results_list

def save_dataset(fname, noise_level, size, min_pass_count, max_pass_count, shapes_set):
    data = toy.generate_dataset(noise_level, size, min_pass_count, max_pass_count, shapes_set)
    data.to_pickle(fname)
    return data

def get_dataset(noise_level, size, min_pass_count=0, max_pass_count=6, shapes_set=np.arange(9)):
    fname = f'toysets/toy_dataset_nl-{noise_level}_diff-{min_pass_count}-{max_pass_count}_{shapes_set}_{size}.pkl'
    if os.path.exists(fname):
        print(f'Loading saved dataset {fname}')
        data = pd.read_pickle(fname)
    else:
        print('Generating new dataset')
        data = save_dataset(fname, noise_level, size, min_pass_count, max_pass_count, shapes_set)
    return data

def get_loader(dataset, train_on, cross_entropy_loss, outer, nonsymbolic):
    """Prepare a torch DataLoader for the provided dataset.

    Other input arguments control what the input features should be and what
    datatype the target should be, depending on what loss function will be used.
    The outer argument appends the flattened outer product of the two input
    vectors (xy and shape) to the input tensor. This is hypothesized to help
    enable the network to rely on an integration of the two streams
    """
    # Create shape and or xy tensors
    if train_on == 'both' or train_on =='shape':
        dataset['shape1'] = dataset['shape']
        def convert(symbolic):
            """return array of 4 lists of nonsymbolic"""
            # dataset size x n glimpses x n shapes (100 x 4 x 9)
            # want to convert to 100 x 4 x n_shapes in this glimpse x 3)
            coords = [(x,y) for (x,y) in product([0.2, 0.5, 0.8], [0.2, 0.5, 0.8])]
            indexes = np.arange(9)
            # [word for sentence in text for  word in sentence]
            # nonsymbolic = [(glimpse[idx], coords[idx][0], coords[idx][1]) for glimpse in symbolic for idx in glimpse.nonzero()[0]]
            nonsymbolic = [[],[],[],[]]
            for i, glimpse in enumerate(symbolic):
                np.random.shuffle(indexes)
                nonsymbolic[i] = [(glimpse[idx], coords[idx][0], coords[idx][1]) for idx in indexes]
            return nonsymbolic
        if nonsymbolic:
            converted = dataset['shape1'].apply(convert)
            shape = torch.tensor(converted).float().to(device)
            # shape = [torch.tensor(glimpse).float().to(device) for glimpse in converted]
        else:
            shape = torch.tensor(dataset['shape']).float().to(device)
    if train_on == 'both' or train_on == 'xy':
        xy = torch.tensor(dataset['xy']).float().to(device)

    # Create merge input (or not)
    if train_on == 'xy':
        input = xy
    elif train_on == 'shape':
        input = shape
    elif train_on == 'both':
        if outer:
            # dataset['shape.t'] = dataset['shape'].apply(lambda x: np.transpose(x))
            # kernel = np.outer(sh, xy) for sh, xy in zip
            def get_outer(xy, shape):
                return [np.outer(x,s).flatten() for x,s in zip(xy, shape)]
            dataset['kernel'] = dataset.apply(lambda x: get_outer(x.xy, x.shape1), axis=1)
            kernel = torch.tensor(dataset['kernel']).float().to(device)
            input = torch.cat((xy, shape, kernel), dim=-1)
        elif not nonsymbolic:
            input = torch.cat((xy, shape), dim=-1)

    if cross_entropy_loss:
        target = torch.tensor(dataset['numerosity']).long().to(device)
    else:
        target = torch.tensor(dataset['numerosity']).float().to(device)
    pass_count = torch.tensor(dataset['pass count']).float().to(device)
    true_loc = torch.tensor(dataset['locations']).float().to(device)

    if nonsymbolic:
        dset = TensorDataset(xy, shape, target, true_loc, pass_count)
    else:
        dset = TensorDataset(input, target, true_loc, pass_count)
    loader = DataLoader(dset, batch_size=BATCH_SIZE, shuffle=True)
    return loader

def get_model(model_type, small_weights, train_on, outer, hidden_size, act, alt_rnn, no_symbol):
    xy_sz = 2
    sh_sz = 9
    in_sz = xy_sz if train_on=='xy' else sh_sz if train_on=='shape' else sh_sz + xy_sz
    if train_on == 'both' and outer:
        in_sz += xy_sz * sh_sz
    output_size = 5
    if model_type == 'num_as_mapsum':
        if no_symbol:
            model = mod.NumAsMapsum_nosymbol(in_sz, hidden_size, output_size, act, alt_rnn).to(device)
        else:
            model = mod.NumAsMapsum(in_sz, hidden_size, output_size, act, alt_rnn).to(device)
    elif model_type == 'detached':
        if no_symbol:
            model = mod.DetachedReadout_nosymbol(in_sz, hidden_size, output_size, act, alt_rnn).to(device)
        else:
            model = mod.DetachedReadout(in_sz, hidden_size, output_size, act, alt_rnn).to(device)
    elif model_type == 'rnn_classifier':
        if no_symbol:
            model = mod.RNNClassifier_nosymbol(in_sz, hidden_size, output_size, act, alt_rnn).to(device)
        else:
            model = mod.RNNClassifier(in_sz, hidden_size, output_size, act, alt_rnn).to(device)
    elif model_type == 'rnn_regression':
        model = mod.RNNRegression(in_sz, hidden_size, output_size, act, alt_rnn).to(device)
    elif model_type == 'mult':
        model = mod.MultiplicativeModel(in_sz, hidden_size, output_size, small_weights).to(device)
    elif model_type == 'hyper':
        model = mod.HyperModel(in_sz, hidden_size, output_size).to(device)
    else:
        print(f'Model type {model_type} not implemented. Exiting')
        exit()
    # if small_weights:
    #     model.init_small()  # not implemented yet
    return model

def get_config():
    parser = argparse.ArgumentParser(description='PyTorch network settings')
    parser.add_argument('--model_type', type=str, default='num_as_mapsum')
    parser.add_argument('--train_on', type=str, default='xy', help='xy, shape, or both')
    parser.add_argument('--noise_level', type=float, default=1.6)
    parser.add_argument('--train_size', type=int, default=100000)
    parser.add_argument('--test_size', type=int, default=5000)
    parser.add_argument('--n_iters', type=int, default=1, help='how many times the rnn should loop through sequence')
    parser.add_argument('--rotate', action='store_true', default=False)  # not implemented
    parser.add_argument('--small_weights', action='store_true', default=False)  # not implemented
    parser.add_argument('--n_epochs', type=int, default=500)
    parser.add_argument('--use_loss', type=str, default='both', help='num, map or both')
    parser.add_argument('--pretrained', type=str, default='num_as_mapsum-xy_nl-0.0_niters-1_5eps_10000_map-loss_ep-5.pt')  # not implemented
    parser.add_argument('--outer', action='store_true', default=False)
    parser.add_argument('--h_size', type=int, default=25)
    parser.add_argument('--min_pass', type=int, default=0)
    parser.add_argument('--max_pass', type=int, default=6)
    parser.add_argument('--act', type=str, default=None)
    parser.add_argument('--alt_rnn', action='store_true', default=False)
    parser.add_argument('--no_symbol', action='store_true', default=False)
    parser.add_argument('--train_shapes', type=list, default=[0, 1, 2, 3, 5, 6, 7, 8])
    parser.add_argument('--test_shapes', nargs='*', type=list, default=[[0, 1, 2, 3, 5, 6, 7, 8], [4]])
    # parser.add_argument('--no_cuda', action='store_true', default=False)
    # parser.add_argument('--preglimpsed', type=str, default=None)
    # parser.add_argument('--use_schedule', action='store_true', default=False)
    # parser.add_argument('--drop_readout', type=float, default=0.5)
    # parser.add_argument('--drop_rnn', type=float, default=0.1)
    # parser.add_argument('--wd', type=float, default=0) # 1e-6
    # parser.add_argument('--lr', type=float, default=0.01)

    # parser.add_argument('--debug', action='store_true', default=False)
    # parser.add_argument('--bce', action='store_true', default=False)
    config = parser.parse_args()
    config.train_shapes = [int(i) for i in config.train_shapes]
    for j, test_set in enumerate(config.test_shapes):
        config.test_shapes[j] = [int(i) for i in test_set]
    print(config)
    return config

def main():
    # Process input arguments
    config = get_config()
    model_type = config.model_type
    if 'classifier' in model_type or model_type == 'detached':
        config.cross_entropy = True
    else:
        config.cross_entropy = False
    train_on = config.train_on
    noise_level = config.noise_level
    train_size = config.train_size
    test_size = config.test_size
    n_iters = config.n_iters
    n_epochs = config.n_epochs
    min_pass = config.min_pass
    max_pass = config.max_pass
    use_loss = config.use_loss

    kernel = '-kernel' if config.outer else ''
    act = config.act if config.act is not None else ''
    alt_rnn = '2' if config.alt_rnn else ''
    model_desc = f'{model_type}{alt_rnn}-{act}_hsize-{config.h_size}'
    data_desc = f'input-{train_on}{kernel}_nl-{noise_level}_diff-{min_pass}-{max_pass}_trainshapes-{config.train_shapes}_{train_size}'
    # train_desc = f'loss-{use_loss}_niters-{n_iters}_{n_epochs}eps'
    train_desc = f'loss-{use_loss}_{n_epochs}eps'
    base_name = f'{model_desc}_{data_desc}_{train_desc}'
    if config.small_weights:
        base_name += '_small'
    if config.no_symbol:
        base_name += '_nonsymbol'
    # Prepare datasets and torch dataloaders
    trainset = get_dataset(noise_level, train_size, min_pass, max_pass, config.train_shapes)
    testsets = [get_dataset(noise_level, test_size, shapes_set=test_shapes) for test_shapes in config.test_shapes]
    train_loader = get_loader(trainset, config.train_on, config.cross_entropy, config.outer, config.no_symbol)
    test_loaders = [get_loader(testset, config.train_on, config.cross_entropy, config.outer, config.no_symbol) for testset in testsets]
    loaders = [train_loader, test_loaders]

    # Prepare model and optimizer
    model = get_model(model_type, config.small_weights, train_on, config.outer, config.h_size, config.act, config.alt_rnn, config.no_symbol)
    opt = SGD(model.parameters(), lr=start_lr, momentum=mom, weight_decay=wd)
    scheduler = StepLR(opt, step_size=n_epochs/10, gamma=0.7)

    # Train model and save trained model
    model, results = train_model(model, opt, scheduler, loaders, base_name, config)
    torch.save(model.state_dict(), f'models/toy/toy_model_{base_name}_ep-{n_epochs}.pt')

    # Organize and save results
    train_loss, train_acc, train_map_loss, test_loss, test_acc, test_map_loss, test_results = results
    test_results.to_pickle(f'results/toy/detailed_test_results_{base_name}.pkl')
    df_train = pd.DataFrame()
    df_test_list = [pd.DataFrame() for _ in config.test_shapes]
    df_train['loss'] = train_loss
    df_train['map loss'] = train_map_loss
    df_train['accuracy'] = train_acc
    df_train['epoch'] = np.arange(n_epochs)
    df_train['rnn iterations'] = n_iters
    df_train['dataset'] = 'train'
    for ts, test_shapes in enumerate(config.test_shapes):
        df_test_list[ts]['loss'] = test_loss[ts]
        df_test_list[ts]['map loss'] = test_map_loss[ts]
        df_test_list[ts]['accuracy'] = test_acc[ts]
        df_test_list[ts]['dataset'] = f'test {test_shapes}'
        df_test_list[ts]['epoch'] = np.arange(n_epochs)

    df_test = pf.concat(df_test_list)
    df_test['rnn iterations'] = n_iters
    df = pd.concat((df_train, df_test))
    df.to_pickle(f'results/toy/toy_results_{base_name}.pkl')


if __name__ == '__main__':
    main()


# Eventually the plot we want to make is
# sns.countplot(data=correct, x='pass count', hue='rnn iterations')
