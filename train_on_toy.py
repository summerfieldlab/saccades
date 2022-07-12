"""Train simple models on data synthesized from toy model."""
import os
import argparse
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from models import RNN, RNN2, MultRNN, MultiplicativeLayer
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


class NumAsMapsum(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, act=None, alternate_rnn=False):
        super().__init__()
        map_size = 9
        self.embedding = nn.Linear(input_size, hidden_size)
        if alternate_rnn:
            self.rnn = RNN2(hidden_size, hidden_size, hidden_size, act)
        else:
            self.rnn = RNN(hidden_size, hidden_size, hidden_size, act)
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


class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, act=None, alternate_rnn=False):
        super().__init__()
        map_size = 9
        self.embedding = nn.Linear(input_size, hidden_size)
        if alternate_rnn:
            self.rnn = RNN2(hidden_size, hidden_size, hidden_size, act)
        else:
            self.rnn = RNN(hidden_size, hidden_size, hidden_size, act)
        self.readout = nn.Linear(hidden_size, output_size)
        self.initHidden = self.rnn.initHidden
        self.LReLU = nn.LeakyReLU(0.1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        x = self.LReLU(self.embedding(x))
        x, hidden = self.rnn(x, hidden)
        num = self.readout(x)
        return num, None, hidden

    def init_small(self):
        pass


class RNNRegression(RNNClassifier):
    def __init__(self, input_size, hidden_size, output_size, act=None, alternate_rnn=False):
        super().__init__(input_size, hidden_size, output_size, act=None, alternate_rnn=False)
        map_size = 9
        self.embedding = nn.Linear(input_size, hidden_size)
        if alternate_rnn:
            self.rnn = RNN2(hidden_size, hidden_size, hidden_size, act)
        else:
            self.rnn = RNN(hidden_size, hidden_size, hidden_size, act)
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


def train_model(rnn, optimizer, scheduler, loaders, base_name, config):
    n_epochs = config.n_epochs
    recurrent_iterations = config.n_iters
    cross_entropy = config.cross_entropy
    train_loader, test_loader = loaders
    train_loss = np.zeros((n_epochs,))
    train_map_loss = np.zeros((n_epochs,))
    train_num_loss = np.zeros((n_epochs,))
    train_acc = np.zeros((n_epochs,))
    test_loss = np.zeros((n_epochs,))
    test_map_loss = np.zeros((n_epochs,))
    test_num_loss = np.zeros((n_epochs,))
    test_acc = np.zeros((n_epochs,))
    columns = ['pass count', 'correct', 'predicted', 'true', 'loss', 'num loss', 'map loss', 'epoch']

    def train(loader):
        rnn.train()
        correct = 0
        epoch_loss = 0
        num_epoch_loss = 0
        map_epoch_loss = 0
        for i, (input, target, locations, _) in enumerate(loader):
            # shuffle the sequence order to increase dataset size
            seq_len = input.shape[1]
            for i, row in enumerate(input):
                input[i, :, :] = row[torch.randperm(seq_len), :]

            rnn.zero_grad()
            input_dim = input.shape[0]
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
                    # t=0
                    # input = torch.cat((xy[:, t, :], shape[:, t, :]), dim=1)
                    # input.shape
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

    def test(loader, epoch):
        rnn.eval()
        n_correct = 0
        epoch_loss = 0
        num_epoch_loss = 0
        map_epoch_loss = 0
        test_results = pd.DataFrame(columns=columns)
        for i, (input, target, locations, pass_count) in enumerate(loader):
            batch_results = pd.DataFrame(columns=columns)
            input_dim = input.shape[0]
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
                        # t=0
                        # input = torch.cat((xy[:, t, :], shape[:, t, :]), dim=1)
                        # input.shape
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
        epoch_tr_loss, epoch_tr_num_loss, tr_accuracy, epoch_tr_map_loss = train(train_loader)
        epoch_te_loss, epoch_te_num_loss, te_accuracy, epoch_te_map_loss, epoch_df = test(test_loader, ep)
        test_results = pd.concat((test_results, epoch_df), ignore_index=True)
        train_loss[ep] = epoch_tr_loss
        train_num_loss[ep] = epoch_tr_num_loss
        train_acc[ep] = tr_accuracy
        train_map_loss[ep] = epoch_tr_map_loss
        test_loss[ep] = epoch_te_loss
        test_num_loss[ep] = epoch_te_num_loss
        test_acc[ep] = te_accuracy
        test_map_loss[ep] = epoch_te_map_loss
        if not ep % 5:
            sns.lineplot(data=test_results, x='epoch', y='num loss', hue='pass count')
            plt.plot(train_num_loss[:ep + 1], '--', color='black', label='training loss')
            plt.title(f'{config.model_type} trainon-{config.train_on} useloss-{config.use_loss} noise-{config.noise_level}')
            ylim = plt.ylim()
            plt.ylim([-0.05, ylim[1]])
            plt.grid()
            plt.savefig(f'figures/toy/test_num-loss_{base_name}.png', dpi=300)
            plt.close()

            sns.lineplot(data=test_results, x='epoch', y='map loss', hue='pass count')
            plt.plot(train_map_loss[:ep + 1], '--', color='black', label='training loss')
            plt.title(f'{config.model_type} trainon-{config.train_on} useloss-{config.use_loss} noise-{config.noise_level}')
            ylim = plt.ylim()
            plt.ylim([-0.05, ylim[1]])
            plt.grid()
            plt.savefig(f'figures/toy/test_map-loss_{base_name}.png', dpi=300)
            plt.close()
            # sns.countplot(data=test_results[test_results['correct']==True], x='epoch', hue='pass count')
            # plt.savefig(f'figures/toy/test_correct_{base_name}.png', dpi=300)
            # plt.close()
            test_results['accuracy'] = test_results['correct'].astype(int)*100
            accuracy = test_results.groupby(['epoch', 'pass count']).mean()
            sns.lineplot(data=accuracy, x='epoch', hue='pass count', y='accuracy')
            plt.grid()
            plt.title(f'{config.model_type} trainon-{config.train_on} useloss-{config.use_loss} noise-{config.noise_level}')
            plt.ylim([0, 102])
            plt.ylabel('Accuracy on number task')
            plt.savefig(f'figures/toy/accuracy_{base_name}.png', dpi=300)
            plt.close()

        epoch_timer.stop_timer()
        print(f'Epoch {ep}. LR={optimizer.param_groups[0]["lr"]} \t (Train/Test) Loss={train_loss[ep]:.4}/{test_loss[ep]:.4}/ \t Accuracy={train_acc[ep]}%/{test_acc[ep]}%')
    results_list = [train_loss, train_acc, train_map_loss, test_loss, test_acc, test_map_loss, test_results]
    return rnn, results_list

def save_dataset(fname, noise_level, size, min_pass_count, max_pass_count):
    data = toy.generate_dataset(noise_level, size, min_pass_count, max_pass_count)
    data.to_pickle(fname)
    return data

def get_dataset(noise_level, size, min_pass_count=0, max_pass_count=6):
    fname = f'toysets/toy_dataset_nl-{noise_level}_diff-{min_pass_count}-{max_pass_count}_{size}.pkl'
    if os.path.exists(fname):
        print('Loading saved dataset')
        data = pd.read_pickle(fname)
    else:
        print('Generating new dataset')
        data = save_dataset(fname, noise_level, size, min_pass_count, max_pass_count)
    return data

def get_loader(dataset, train_on, cross_entropy_loss, outer):
    """Prepare a torch DataLoader for the provided dataset.

    Other input arguments control what the input features should be and what
    datatype the target should be, depending on what loss function will be used.
    The outer argument appends the flattened outer product of the two input
    vectors (xy and shape) to the input tensor. This is hypothesized to help
    enable the network to rely on an integration of the two streams
    """
    if train_on == 'both':
        dataset['shape1'] = dataset['shape']
        shape = torch.tensor(dataset['shape']).float().to(device)
        xy = torch.tensor(dataset['xy']).float().to(device)
        if outer:
            # dataset['shape.t'] = dataset['shape'].apply(lambda x: np.transpose(x))
            # kernel = np.outer(sh, xy) for sh, xy in zip
            def get_outer(xy, shape):
                return [np.outer(x,s).flatten() for x,s in zip(xy, shape)]
            dataset['kernel'] = dataset.apply(lambda x: get_outer(x.xy, x.shape1), axis=1)
            kernel = torch.tensor(dataset['kernel']).float().to(device)
            input = torch.cat((xy, shape, kernel), dim=-1)
        else:
            input = torch.cat((xy, shape), dim=-1)
    elif train_on == 'xy':
        input = torch.tensor(dataset['xy']).float().to(device)
    elif train_on == 'shape':
        input = torch.tensor(dataset['shape']).float().to(device)

    if cross_entropy_loss:
        target = torch.tensor(dataset['numerosity']).long().to(device)
    else:
        target = torch.tensor(dataset['numerosity']).float().to(device)
    pass_count = torch.tensor(dataset['pass count']).float().to(device)
    true_loc = torch.tensor(dataset['locations']).float().to(device)
    dset = TensorDataset(input, target, true_loc, pass_count)
    loader = DataLoader(dset, batch_size=BATCH_SIZE, shuffle=True)
    return loader

def get_model(model_type, small_weights, train_on, outer, hidden_size, act, alt_rnn):
    xy_sz = 2
    sh_sz = 9
    in_sz = xy_sz if train_on=='xy' else sh_sz if train_on=='shape' else sh_sz + xy_sz
    if train_on == 'both' and outer:
        in_sz += xy_sz * sh_sz
    output_size = 5
    if model_type == 'num_as_mapsum':
        model = NumAsMapsum(in_sz, hidden_size, output_size, act, alt_rnn).to(device)
    elif model_type == 'rnn_classifier':
        model = RNNClassifier(in_sz, hidden_size, output_size, act, alt_rnn).to(device)
    elif model_type == 'rnn_regression':
        model = RNNRegression(in_sz, hidden_size, output_size, act, alt_rnn).to(device)
    elif model_type == 'mult':
        model = MultiplicativeModel(in_sz, hidden_size, output_size, small_weights).to(device)
    elif model_type == 'hyper':
        model = HyperModel(in_sz, hidden_size, output_size).to(device)
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
    print(config)
    return config

def main():
    # Process input arguments
    config = get_config()
    model_type = config.model_type
    config.cross_entropy = True if model_type=='rnn_classifier' else False
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
    base_name = f'{model_type}{alt_rnn}-{act}-{train_on}{kernel}_hsize-{config.h_size}_loss-{use_loss}_nl-{noise_level}_diff-{min_pass}-{max_pass}_niters-{n_iters}_{n_epochs}eps_{train_size}'
    if config.small_weights:
        base_name += '_small'
    # Prepare datasets and torch dataloaders
    trainset = get_dataset(noise_level, train_size, min_pass, max_pass)
    testset = get_dataset(noise_level, test_size)
    train_loader = get_loader(trainset, config.train_on, config.cross_entropy, config.outer)
    test_loader = get_loader(testset, config.train_on, config.cross_entropy, config.outer)
    loaders = [train_loader, test_loader]

    # Prepare model and optimizer
    model = get_model(model_type, config.small_weights, train_on, config.outer, config.h_size, config.act, config.alt_rnn)
    opt = SGD(model.parameters(), lr=start_lr, momentum=mom, weight_decay=wd)
    scheduler = StepLR(opt, step_size=n_epochs/10, gamma=0.7)

    # Train model and save trained model
    model, results = train_model(model, opt, scheduler, loaders, base_name, config)
    torch.save(model.state_dict(), f'models/toy/toy_model_{base_name}_ep-{n_epochs}.pt')

    # Organize and save results
    train_loss, train_acc, train_map_loss, test_loss, test_acc, test_map_loss, test_results = results
    test_results.to_pickle(f'results/toy/detailed_test_results_{base_name}.pkl')
    df_train = pd.DataFrame(columns=['train loss', 'train accuracy', 'update', 'rnn iterations'])
    df_test = pd.DataFrame(columns=['train loss', 'train accuracy', 'update', 'rnn iterations'])
    df_train['loss'] = train_loss
    df_train['map loss'] = train_map_loss
    df_train['accuracy'] = train_acc
    df_train['epoch'] = np.arange(n_epochs)
    df_train['rnn iterations'] = n_iters
    df_test['loss'] = train_loss
    df_test['map loss'] = train_map_loss
    df_test['accuracy'] = train_acc
    df_test['epoch'] = np.arange(n_epochs)
    df_test['rnn iterations'] = n_iters
    df = pd.concat((df_train, df_test))
    df.to_pickle(f'results/toy/toy_results_{base_name}.pkl')


if __name__ == '__main__':
    main()


# Eventually the plot we want to make is
# sns.countplot(data=correct, x='pass count', hue='rnn iterations')
