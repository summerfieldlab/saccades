import sys
import argparse
from itertools import combinations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.functional import one_hot
torch.set_num_threads(5)
import math
import random
import operator as op
from functools import reduce
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score

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

class TwoStepModel(nn.Module):
    def __init__(self, hidden_size, map_size, output_size, act=None):
        super(TwoStepModel, self).__init__()
        self.hidden_size = hidden_size
        self.map_size = map_size
        self.out_size = output_size
        self.rnn = RNN(map_size, hidden_size, map_size, act)
        # self.rnn = tanhRNN(input_size, hidden_size, map_size)
        self.fc = nn.Linear(map_size, output_size, bias=True)

    def forward(self, x, hidden):
        map_, hidden = self.rnn(x, hidden)
        number = self.fc(torch.sigmoid(map_))
        # number = torch.relu(self.fc(map))
        return map_, hidden, number

class TwoStepModel_hist(nn.Module):
    def __init__(self, input_size, hidden_size, map_size, output_size):
        super(TwoStepModel_hist, self).__init__()
        fc_size = 64
        self.rnn = linearRNN_hist(input_size, hidden_size, map_size)
        # self.rnn = tanhRNN(input_size, hidden_size, map_size)
        self.fc1 = nn.Linear(map_size, fc_size)
        self.fc2 = nn.Linear(fc_size, output_size)

    def forward(self, x, hidden):
        outs, hidden, hist = self.rnn(x, hidden)
        fc_layer = torch.relu(self.fc1(hist))
        number = self.fc2(fc_layer)
        # number = torch.relu(self.fc(map))
        return outs, hidden, number

class TwoStepModel_weightshare(nn.Module):
    def __init__(self, hidden_size, map_size, output_size, init, act=None):
        super(TwoStepModel_weightshare, self).__init__()
        # self.saved_model = torch.load('models/two_step_model_with_maploss_bias.pt')
        self.hidden_size = hidden_size
        self.map_size = map_size
        self.out_size = output_size
        self.rnn = RNN(map_size, hidden_size, map_size, act)
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
        return map_, hidden, number


class TwoStepModel_weightshare_detached(TwoStepModel_weightshare):

    def forward(self, x, hidden):
        map_, hidden = self.rnn(x, hidden)
        # number = self.fc(torch.sigmoid(map_))
        map_detached = map_.detach()
        number = self.readout(torch.sigmoid(map_detached))
        # number = torch.relu(self.fc(map))
        return map_, hidden, number

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
        self.weight_mat = self.weight.expand(map_size, out_size)
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
        out = torch.addmm(self.bias, map, self.weight_mat)
        return out


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, act=None):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.out_size = output_size
        self.hidden_size = hidden_size
        if act == 'tanh':
            self.act_fun = torch.tanh
        else:
            self.act_fun = None

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        if self.act_fun:
            hidden = self.act_fun(hidden)
        output = self.i2o(combined)
        # output = self.softmax(output)
        return output, hidden

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

class tanhRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(tanhRNN, self).__init__()
        self.out_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = torch.tanh(self.i2h(combined))
        output = self.i2o(combined)
        # output = self.softmax(output)
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)







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


def train_rnn(model, optimizer, n_epochs, device, model_version, use_loss='number'):
    print('Linear RNN on unique task...')
    rnn = model.to(device)
    rnn.train()
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()
    batch_size = 64
    seq_len = 7
    n_classes = 7
    # data, target, _, _ = get_data(seq_len, n_classes, device)
    data, target, map_, _, _  = get_data(seq_len, rnn.out_size, rnn.input_size, device)
    if use_loss == 'map';
        target = map_
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
            loss.backward()
            optimizer.step()
            hidden.detach_()

            with torch.no_grad():
                out_local = output
                # Evaluate performance
                pred = out_local.argmax(dim=1, keepdim=True)
                correct += pred.eq(label.view_as(pred)).sum().item()
                train_loss += loss.item()
        train_loss /= batch_idx + 1
        numb_losses[ep] = train_loss
        acc = 100. * (correct / len(dataset))
        numb_accs[ep] = acc
        if not ep % 5:
            pct_done = round(100. * (ep / n_epochs))
            print(f'Epoch {ep}, Progress {pct_done}% trained: \t Loss: {train_loss:.6}, Accuracy: {acc:.6}% ({correct}/{len(dataset)})')
    filename = f'results/{model_version}results_mapsz{rnn.input_size}_loss-number.npz'
    np.savez(filename, numb_accs=numb_accs, numb_losses=numb_losses)

def train_two_step_model(model, optimizer, n_epochs, device, differential,
                         use_loss, model_version, control=False):
    print('Two step model...')

    rnn = model.rnn
    model.to(device)

    # Synthesize the data
    batch_size = 64
    seq_len = 7
    data, num, map_, map_control, hist = get_data(seq_len, model.out_size, model.map_size, device)
    # with torch.no_grad():
    #     verify_dataset(data, num, seq_len)

    # Calculate the weight per location to trade off precision and recall
    # As map_size increases, negative examples far outweight positive ones
    # Hoping this may also help with numerical stability
    positive = (data.sum(axis=1) > 0).sum(axis=0)
    negative = -positive + data.shape[0]
    pos_weight = torch.true_divide(negative, positive)
    map_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion = nn.CrossEntropyLoss()
    add_number_loss = False

    # To store learning curves
    numb_losses = np.zeros((n_epochs,))
    numb_accs = np.zeros((n_epochs,))
    map_losses = np.zeros((n_epochs,))
    map_accs = np.zeros((n_epochs, model.map_size))
    map_auc = np.zeros((n_epochs,))

    # Where to save the results
    if control:
        filename = f'reults/two_step_results_{control}.npz'
    else:
        if use_loss == 'map_then_both':
            filename = f'results/{model_version}_results_mapsz{model.map_size}_loss-{use_loss}-hardthresh.npz'
        else:
            filename = f'results/{model_version}_results_mapsz{model.map_size}_loss-{use_loss}.npz'

    if control == 'non_spatial':
        dataset = TensorDataset(data, map_control, num)
    elif control == 'hist':
        dataset = TensorDataset(data, hist, num)
    else:
        dataset = TensorDataset(data, map_, num)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f'**Training model with map size of {model.map_size}, hidden layer size {model.hidden_size}, use_loss={use_loss}')
    for ep in range(n_epochs):
        m_correct = np.zeros(model.map_size,)
        n_correct = 0
        auc =  0
        train_loss = 0
        map_train_loss = 0
        # shuffle the sequence order on each epoch
        # for i, row in enumerate(data):
        #     data[i, :, :] = row[torch.randperm(seq_len), :]
        # Select which map target to use
        for batch_idx, (inputs, map_label, numb_label) in enumerate(loader):
            model.zero_grad()
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
                map, hidden, number = model(this_input, hidden)
                prev_input = this_input

            if control == 'hist':
                map_loss = [criterion(map[j], map_label[:, j]) for j in range(len(map))]
                map_loss = sum(map_loss)/len(map_loss)
            else:
                map_loss = map_criterion(map, map_label)
            number_loss = criterion(number, numb_label)
            if use_loss == 'map':
                loss = map_loss
                loss.backward()
            elif use_loss == 'number':
                loss = number_loss
                loss.backward()
            elif use_loss == 'both':
                loss = map_loss + number_loss
                loss.backward()
            elif use_loss == 'map_then_both' or use_loss == 'map_then_both-detached':
                if ep > 0 and (map_auc[ep - 1] > 0.98 or ep > n_epochs / 2):
                    add_number_loss = True
                if add_number_loss:
                    loss = map_loss + number_loss
                else:
                    loss = map_loss
                loss.backward()
            elif use_loss == 'both-detached':

                # model.train_readout()
                # number_loss.backward(retain_graph=True)
                # model.train_rnn()
                # map_loss.backward()
                loss = map_loss + number_loss
                loss.backward()
                # model.train()

            else:
                print('must specify which loss to optimize')

            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            # Take a gradient step
            optimizer.step()
            hidden.detach_()

            # Evaluate performance
            with torch.no_grad():
                train_loss += loss.item()
                map_train_loss += map_loss.item()
                numb_local = number
                map_local = map
                map_label_local = map_label
                if control == 'hist':
                    for j in range(len(map_local)):
                        pred = map_local[j].argmax(dim=1, keepdim=True)
                        m_correct[j] += pred.eq(map_label[:, j].view_as(pred)).sum().item()
                    # m_correct += sum(torch.isclose(map_local, map_label)).cpu().numpy()
                else:
                    sigout = torch.sigmoid(map_local).round()
                    m_correct += sum(sigout == map_label_local).cpu().numpy()
                    auc += roc_auc_score(map_label_local.cpu().numpy().flatten(), sigout.cpu().flatten())
                pred = numb_local.argmax(dim=1, keepdim=True)
                n_correct += pred.eq(numb_label.view_as(pred)).sum().item()

        # Make figure
        # fig, axs = plt.subplots(2, 2, sharex=True, figsize=(10, 10))
        # im = axs[0, 0].imshow(this_input[0, :].detach().cpu().view(30, 30), vmin=0, vmax=1, cmap='bwr')
        # axs[0, 0].set_title('Example Input')
        # plt.colorbar(im, ax=axs[0,0], orientation='horizontal')
        #
        # im1 = axs[1, 0].imshow(map_local[0, :].detach().cpu().view(30, 30))
        # axs[1, 0].set_title('Predicted Map')
        # plt.colorbar(im1, ax=axs[1,0], orientation='horizontal')
        # # fig.colorbar(im1, orientation='horizontal')
        #
        # im2 = axs[0, 1].imshow(torch.sigmoid(map_local[0, :]).detach().cpu().view(30, 30), vmin=0, vmax=1, cmap='bwr')
        # axs[0, 1].set_title(f'Sigmoid(Predicted Map) (Predicted number: {pred[0]})')
        # plt.colorbar(im2, ax=axs[0,1], orientation='horizontal')
        #
        # im3 = axs[1, 1].imshow(map_label_local[0, :].detach().cpu().view(30, 30), vmin=0, vmax=1, cmap='bwr')
        # axs[1, 1].set_title(f'Actual Map (number={numb_label[0]})')
        # plt.colorbar(im3, ax=axs[1,1], orientation='horizontal')
        # # plt.show()
        # plt.savefig(f'figures/predicted_maps/predicted_map_ep{ep}.png', bbox_inches='tight', dpi=200)
        # plt.close()

        train_loss /= batch_idx + 1
        map_train_loss /= batch_idx + 1

        accs = 100. * (m_correct / len(dataset))
        map_accs[ep, :] = accs
        auc /= batch_idx+1

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


    return numb_acc, map_acc

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

def get_data(seq_len=7, n_classes=7, map_size=1056, device=None):
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

    min([ncr(map_size, n) for n in classes])

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
    lr = 0.01
    mom = 0.9
    wd = 0.0
    n_epochs = 2000

    # input_size = 1  # difference code
    input_size = 26**2 # ds-factor=5 676
    # input_size = 10 # for testing
    # input_size = 33**2 # 1089   # one_hot
    # out_size = 8
    numb_size = 8
    dropout_prob = 0.0

    # device = torch.device("cuda")
    if config.no_cuda:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    # map_sizes = np.append([7, 15], np.arange(25, 925, 25))
    # map_sizes = np.arange(100, 1000, 100)[::-1]
    # map_sizes = [7, 15, 25, 50, 100, 200, 300, 600]
    # map_sizes = [10, 25, 50, 100, 200, 400, 676]
    # map_sizes = [50, 100, 200, 400, 676]
    map_sizes = [10, 15, 25, 50, 100, 200, 400, 676]
    # map_sizes = [1056]
    # for map_size in map_sizes:
    #     verify_dataset(map_size=map_size)

    # model_versions = ['two_step', 'two_step_ws', 'two_step_ws_init']
    # model_version = 'two_step'

    # model = RNN(input_size, hidden_size, out_size)
    # opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=mom)
    # train_rnn(model, opt, n_epochs)



    differential = False
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
    hidden_size = int(np.round(map_size + map_size*0.1))
    # hidden_size = map_size*2
    if model_version == 'two_step':
        model = TwoStepModel(hidden_size, map_size, numb_size, act=rnn_act)
    elif model_version == 'two_step_ws':
        if 'detached' in use_loss:
            model = TwoStepModel_weightshare_detached(hidden_size, map_size, numb_size, init=False, act=rnn_act)
        else:
            model = TwoStepModel_weightshare(hidden_size, map_size, numb_size, init=False, act=rnn_act)
    elif model_version == 'two_step_ws_init':
        if 'detached' in use_loss:
            model = TwoStepModel_weightshare_detached(hidden_size, map_size, numb_size, init=True, act=rnn_act)
        else:
            model = TwoStepModel_weightshare(hidden_size, map_size, numb_size, init=True, act=rnn_act)
    elif 'one_step' in model_version:
        if use_loss == 'number':
            model = RNN(map_size, hidden_size, numb_size, act=rnn_act)
        elif use_loss == 'map':
            model = RNN(map_size, hidden_size, map_size, act=rnn_act)
    else:
        print('Model version not implemented.')
        exit()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=mom, weight_decay=wd)
    if 'two' in model_version:
        train_two_step_model(model, opt, n_epochs, device, differential, use_loss, model_version)
    elif 'one' in model_version and use_loss == 'number':
        train_rnn(model, opt, n_epochs, device, model_version)

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
    config = parser.parse_args()
    print(config)
    return config

if __name__ == '__main__':
    config = get_config()
    main(config)
