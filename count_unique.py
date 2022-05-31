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
torch.backends.cudnn.benchmark = False
import math
import random
import operator as op
from functools import reduce
from matplotlib import pyplot as plt
import matplotlib as mpl
from sklearn.metrics import roc_auc_score, f1_score, r2_score
from torchmetrics import R2Score
import pandas as pd
import seaborn as sns
import models

print(f'matplotlib version {mpl.__version__}')
r2score = R2Score(num_outputs=14)
# r2score = R2Score()

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

def train_two_step_model(model, optimizer, config, scheduler=None):
    """Main train/test loop."""
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
    # drop = config.dropout
    drop_rnn = config.drop_rnn
    drop_readout = config.drop_readout
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
            filename = f'{model_version}{pretrained}_{nonlinearity}results_mapsz{model.map_size}_loss-{use_loss}-hardthresh{sched}_lr{config.lr}_wd{config.wd}_dr-rnn{drop_rnn}_dr-ro{drop_readout}_tanhrelu_rand_trainon-{config.train_on}'
        else:
            filename = f'{model_version}{pretrained}_{nonlinearity}results_mapsz{model.map_size}_loss-{use_loss}{sched}_lr{config.lr}_wd{config.wd}_dr-rnn{drop_rnn}_dr-ro{drop_readout}_tanhrelu_rand_trainon-{config.train_on}'
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

            # for i, row in enumerate(inputs):
            # shuffle the sequence order
                # inputs[i, :, :] = row[torch.randperm(seq_len), :]

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
                ep_thresh = 500
                retain_graph = False
                #if ep > 0 and (map_f1[ep - 1] > 0.5 or ep > n_epochs / 2):
                if ep > ep_thresh:
                    add_number_loss = True
                    retrain_graph = True
                # model.train_rnn()
                map_loss.backward(retain_graph=retain_graph)
                nn.utils.clip_grad_norm_(model.rnn.parameters(), clip_grad_norm)
                opt_rnn.step()
                if add_number_loss:
                    # model.train_readout()
                    number_loss.backward()
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
                readout_scheduler_rate = 1 if config.pretrained_map else 2
                if (add_number_loss or (use_loss == 'both') or (use_loss == 'both-detached') or ('number' in use_loss)):  #and ep % readout_scheduler_rate:
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
            plt.suptitle(f'{config.model_version}, nonlin={config.rnn_act}, loss={config.use_loss}, dr-rnn={config.drop_rnn}, dr-ro={config.drop_readout} \n tanh(relu(map)), seq not shuffled, pos_weight=130-4.5/4.5 \n {preglimpsed}',fontsize=20)
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

def train_nomap_model(model, optimizer, config, scheduler=None):
    """For training shape cheat model."""
    print('no map model...')
    binarize = True
    n_epochs = config.n_epochs
    device = config.device
    model_version = config.model_version
    control = config.control
    preglimpsed = config.preglimpsed
    eye_weight = config.eye_weight
    rnn = model.rnn
    # drop = config.dropout
    drop_rnn = config.drop_rnn
    drop_readout = config.drop_readout
    clip_grad_norm = 1

    # Synthesize or load the data
    batch_size = 64
    preglimpsed_train = preglimpsed + '_train' if preglimpsed is not None else None
    _, num, shape, _, _, _ = get_min_data(preglimpsed_train, config.train_on, device)
    seq_len = shape.shape[1]
    criterion = nn.CrossEntropyLoss()

    trainset = TensorDataset(shape, num)
    preglimpsed_val = preglimpsed + '_valid' if preglimpsed is not None else None
    _, num, shape, _, _, _ = get_min_data(preglimpsed_val, config.train_on, device)
    validset = TensorDataset(shape, num)
    preglimpsed_test = preglimpsed + '_test0' if preglimpsed is not None else None
    _, num, shape, _, _, _ = get_min_data(preglimpsed_test, config.train_on, device)
    testset = TensorDataset(shape, num)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(validset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True)

    # To store learning curves
    numb_losses = np.zeros((n_epochs+1,))
    numb_accs = np.zeros((n_epochs+1,))
    valid_numb_losses = np.zeros((n_epochs+1,))
    valid_numb_accs = np.zeros((n_epochs+1,))
    test_numb_losses = np.zeros((n_epochs+1,))
    test_numb_accs = np.zeros((n_epochs+1,))

    # Where to save the results
    nonlinearity = config.rnn_act + '_' if config.rnn_act is not None else ''
    sched = '_sched' if config.use_schedule else ''
    pretrained = '-pretrained' if config.pretrained_map else ''
    filename = f'{model_version}{pretrained}_{nonlinearity}results{sched}_lr{config.lr}_wd{config.wd}_dr-rnn{drop_rnn}_dr-ro{drop_readout}'
    if preglimpsed is not None:
        filename += '_' + preglimpsed
    if eye_weight:
        filename += '_eye'
    if binarize:
        filename += '_bin'
    print(filename)

    print(f'**Training cheat model, hidden layer size {model.hidden_size}')
    def train(loader, ep):
        model.train()
        n_correct = 0
        train_loss = 0
        number_train_loss = 0
        for batch_idx, (inputs, numb_label) in enumerate(loader):
            model.zero_grad()
            if binarize:
                inputs = 1.0 * (inputs > 0)
            batch_size = inputs.shape[0]

            # for i, row in enumerate(inputs):
            # shuffle the sequence order
                # inputs[i, :, :] = row[torch.randperm(seq_len), :]

            hidden = rnn.initHidden(batch_size)
            hidden = hidden.to(device)

            # FORWARD PASS
            for i in range(seq_len):
                this_input = inputs[:, i, :]
                number, hidden = model(this_input, hidden)

            loss = criterion(number, numb_label)
            loss.backward()
            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            # Take a gradient step
            optimizer.step()
            hidden.detach_()

            # Evaluate performance
            with torch.no_grad():
                train_loss += loss.item()
                numb_local = number
                pred = numb_local.argmax(dim=1, keepdim=True)
                n_correct += pred.eq(numb_label.view_as(pred)).sum().item()
        if config.use_schedule:
            scheduler.step()
        # Evaluate performance
        train_loss /= len(loader)
        numb_acc = 100. * (n_correct / len(loader.dataset))
        return train_loss, numb_acc

    def test(loader, **kwargs):
        model.eval()
        n_correct = 0
        train_loss = 0
        with torch.no_grad():
            for batch_idx, (inputs, numb_label) in enumerate(loader):
                if binarize:
                    inputs = 1.0 * (inputs > 0)
                batch_size = inputs.shape[0]
                hidden = rnn.initHidden(batch_size)
                hidden = hidden.to(device)
                for i in range(seq_len):
                    this_input = inputs[:, i, :]
                    number, hidden = model(this_input, hidden)
                loss = criterion(number, numb_label)

                # Evaluate performance
                train_loss += loss.item()
                numb_local = number
                pred = numb_local.argmax(dim=1, keepdim=True)
                n_correct += pred.eq(numb_label.view_as(pred)).sum().item()
            # Evaluate performance
            train_loss /= len(loader)
            numb_acc = 100. * (n_correct / len(loader.dataset))
        return train_loss, numb_acc
    # Evaluate performance before training
    numb_losses[0], numb_accs[0] = test(train_loader)
    valid_numb_losses[0], valid_numb_accs[0] = test(valid_loader)
    test_numb_losses[0], test_numb_accs[0] = test(test_loader)
    print('Performance before Training')
    print(f'Train Loss (Num): {numb_losses[0]:.6}, \t Train Performance (Num) {numb_accs[0]:.3}%')
    print(f'Valid Loss (Num): {valid_numb_losses[0]:.6}, \t Valid Performance (Num) {valid_numb_accs[0]:.3}%')
    print(f'Test Loss (Num): {test_numb_losses[0]:.6}, \t Test Performance (Num) {test_numb_accs[0]:.3}%')

    for ep in range(1, n_epochs + 1):
        print(f'Epoch {ep}, Learning rate: {optimizer.param_groups[0]["lr"]}')

        train_loss, train_num_acc = train(train_loader, ep)
        val_loss, val_num_acc = test(valid_loader, which_set='valid')
        test_loss, test_num_acc = test(test_loader, which_set='test')

        numb_losses[ep] = train_loss
        numb_accs[ep] = train_num_acc
        valid_numb_losses[ep] = val_loss
        valid_numb_accs[ep] = val_num_acc
        test_numb_losses[ep] = test_loss
        test_numb_accs[ep] = test_num_acc
        pct_done = round(100. * (ep / n_epochs))
        if not ep % 5 or ep == 1:
            # Print and save performance
            # print(f'Progress {pct_done}% trained: \t Loss (Map/Numb): {map_loss.item():.6}/{number_loss.item():.6}, \t Accuracy: {accs[0]:.3}% {accs[1]:.3}% {accs[2]:.3}% {accs[3]:.3}% {accs[4]:.3}% {accs[5]:.3}% {accs[6]:.3}% \t Number {numb_acc:.3}% \t Mean Map Acc {accs.mean():.3}%')
            print(f'Epoch {ep}, Progress {pct_done}% trained')
            print(f'Train Loss (Num): {train_loss:.6}, \t Train Performance (Num) {train_num_acc:.3}%')
            print(f'Valid Loss (Num): {val_loss:.6}, \t Valid Performance (Num) {val_num_acc:.3}%')
            print(f'Test Loss (Num): {test_loss:.6}, \t Test Performance (Num) {test_num_acc:.3}%')

            np.savez('results/'+filename,
                     numb_loss=numb_losses,
                     numb_acc=numb_accs,
                     valid_numb_loss=valid_numb_losses,
                     valid_numb_acc=valid_numb_accs,
                     test_numb_loss=test_numb_losses,
                     test_numb_acc=test_numb_accs)
            torch.save(model, f'models/{filename}.pt')
            # Plot performance
            kwargs_train = {'alpha': 0.8, 'color': 'blue'}
            kwargs_val = {'alpha': 0.8, 'color': 'cyan'}
            kwargs_test = {'alpha': 0.8, 'color': 'red'}
            row, col = 2, 2
            fig, ax = plt.subplots(row, col, figsize=(6*col, 6*row))
            plt.suptitle(f'{config.model_version}, nonlin={config.rnn_act}, dr-rnn={config.drop_rnn}, dr-ro={config.drop_readout}, seq not shuffled  \n {preglimpsed}', fontsize=18)
            ax[1, 0].set_title('Number Loss')
            ax[1, 0].plot(test_numb_losses[:ep+1], label='test number loss', **kwargs_test)
            ax[1, 0].plot(valid_numb_losses[:ep+1], label='valid number loss', **kwargs_val)
            ax[1, 0].plot(numb_losses[:ep+1], label='train number loss', **kwargs_train)
            ax[1, 1].set_title('Number Accuracy')
            ax[1, 1].set_ylim([0, 101])
            ax[1, 1].plot(test_numb_accs[:ep+1], label='test number acc', **kwargs_test)
            ax[1, 1].plot(valid_numb_accs[:ep+1], label='valid number acc', **kwargs_val)
            ax[1, 1].plot(numb_accs[:ep+1], label='train number acc', **kwargs_train)
            ax[0, 0].axis('off')
            ax[0, 1].axis('off')
            for axes in ax.flatten():
                axes.legend()
                axes.grid(linestyle='--')
                axes.set_xlabel('Epochs')
            plt.savefig(f'figures/{model_version}/{filename}_results.png', dpi=300)
            plt.close()

    print(f'Final performance:')
    print(f'Train Loss (Num): {train_loss:.6}, \t Train Performance (Num) {train_num_acc:.3}%')
    print(f'Valid Loss (Num): {val_loss:.6}, \t Valid Performance (Num) {val_num_acc:.3}%')
    print(f'Test Loss (Num): {test_loss:.6}, \t Test Performance (Num) {test_num_acc:.3}%')

    np.savez('results/'+filename,
             numb_loss=numb_losses,
             numb_acc=numb_accs,
             valid_numb_loss=valid_numb_losses,
             valid_numb_acc=valid_numb_accs,
             test_numb_loss=test_numb_losses,
             test_numb_acc=test_numb_accs)
    torch.save(model, f'models/{filename}.pt')

def train_shape_number(model, optimizer, config, scheduler=None):
    """For training model on pixels with aux shape loss.

    Depending on the model, the aux shape loss is just used to shape the pixel
    embedding or the predicted shape vector is jointly embedded with the pixel
    embedding.
    """
    print('Shape and number model...')
    bce = config.bce
    n_epochs = config.n_epochs
    device = config.device
    model_version = config.model_version
    control = config.control
    preglimpsed = config.preglimpsed
    eye_weight = config.eye_weight
    rnn = model.rnn
    # drop = config.dropout
    drop_rnn = config.drop_rnn
    drop_readout = config.drop_readout
    use_loss = config.use_loss
    clip_grad_norm = 2
    criterion = nn.CrossEntropyLoss()
    crit_mse = nn.MSELoss()
    crit_bce = nn.BCEWithLogitsLoss(pos_weight=torch.ones([7], device=device)*6)  #BCELoss()
    # pos_weight = 6 because there are 6 negative examples for each 1 positive example ()
    # Synthesize or load the train, test and validation data
    batch_size = 64
    nclasses = 6
    preglimpsed_train = preglimpsed + '_train' if preglimpsed is not None else None
    # data, num, shape, map_, min_num, min_shape
    pix, num, shape, _, _, _ = get_min_data(preglimpsed_train, config.train_on, device)
    seq_len = shape.shape[1]
    trainset = TensorDataset(pix, shape, num)
    preglimpsed_val = preglimpsed + '_valid' if preglimpsed is not None else None
    pix, num, shape, _, _, _ = get_min_data(preglimpsed_val, config.train_on, device)
    validset = TensorDataset(pix, shape, num)
    preglimpsed_test = preglimpsed + '_test0' if preglimpsed is not None else None
    pix, num, shape, _, _, _ = get_min_data(preglimpsed_test, config.train_on, device)
    testset = TensorDataset(pix, shape, num)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(validset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True)

    # To store learning curves
    train_losses = np.zeros((n_epochs+1,))
    numb_losses = np.zeros((n_epochs+1,))
    shape_losses = np.zeros((n_epochs+1,))
    # shape_r2 = np.zeros((n_epochs+1,14))
    shape_accs_bce = np.zeros((n_epochs+1,))
    shape_accs_ce = np.zeros((n_epochs+1,))
    numb_accs = np.zeros((n_epochs+1,))
    valid_losses = np.zeros((n_epochs+1,))
    valid_numb_losses = np.zeros((n_epochs+1,))
    valid_shape_losses = np.zeros((n_epochs+1,))
    # valid_shape_r2 = np.zeros((n_epochs+1,14))
    valid_shape_accs_bce = np.zeros((n_epochs+1,))
    valid_shape_accs_ce = np.zeros((n_epochs+1,))
    valid_numb_accs = np.zeros((n_epochs+1,))
    valid_conf_mats = np.zeros((n_epochs+1, nclasses, nclasses))
    test_losses = np.zeros((n_epochs+1,))
    test_numb_losses = np.zeros((n_epochs+1,))
    test_shape_losses = np.zeros((n_epochs+1,))
    # test_shape_r2 = np.zeros((n_epochs+1,14))
    test_shape_accs_bce = np.zeros((n_epochs+1,))
    test_shape_accs_ce = np.zeros((n_epochs+1,))
    test_numb_accs = np.zeros((n_epochs+1,))
    test_conf_mats = np.zeros((n_epochs+1, nclasses, nclasses))

    # Where to save the results
    nonlinearity = config.rnn_act + '_' if config.rnn_act is not None else ''
    sched = '_sched' if config.use_schedule else ''
    pretrained = '-pretrained' if config.pretrained_map else ''
    filename = f'{model_version}{pretrained}_{nonlinearity}results_loss-{use_loss}{sched}_lr{config.lr}_wd{config.wd}_dr-rnn{drop_rnn}_dr-ro{drop_readout}'
    if preglimpsed is not None:
        filename += '_' + preglimpsed
    if eye_weight:
        filename += '_eye'
    if bce:
        filename += '_bce'

    print(f'**Training distinctive shape and number model, hidden layer size {model.hidden_size}')
    def train(loader, ep):
        model.train()
        n_correct = 0
        train_loss = 0
        numb_train_loss = 0
        shape_train_loss = 0
        # shape_train_r2 = torch.zeros((14,))
        correct_shape_bce = torch.zeros((14,), device=device)
        correct_shape_ce = 0
        cor_shape_bce2 = 0
        for batch_idx, (pix, shape_label, numb_label) in enumerate(loader):
            model.zero_grad()
            batch_size = pix.shape[0]
            # for i, row in enumerate(inputs):
            # shuffle the sequence order
                # inputs[i, :, :] = row[torch.randperm(seq_len), :]
            hidden = rnn.initHidden(batch_size)
            hidden = hidden.to(device)

            # FORWARD PASS
            this_shape_loss = 0
            # this_shape_r2 = torch.zeros((14,))
            for i in range(seq_len):
                this_input = pix[:, i, :]
                number, shape, hidden = model(this_input, hidden, bce)
                ce_label = torch.argmax(shape_label[:, i, :7], dim=1)
                bce_label = 1.0 * (shape_label[:, i, :7] > 0)
                if 'shape' in use_loss:
                    # shape_loss = shape_crit(shape, shape_label[:, i, :])
                    # mse_loss = crit_mse(shape, shape_label[:, i, :])
                    if bce:
                        # shape_loss = crit_bce(shape[:, :7], bce_label)
                        shape_loss = crit_bce(shape[:, :7], shape_label[:, i, :7])
                    else:
                        shape_loss = criterion(shape, ce_label)
                    shape_loss.backward(retain_graph=True)
                    this_shape_loss += shape_loss.item()
                    with torch.no_grad():
                        sigshape = torch.sigmoid(shape[:, :7])
                        # this_shape_r2 += r2score(sigshape, shape_label[:, i, :]) #torchmetrics
                        shape_pred = torch.round(sigshape)
                        # bin_shape_label = 1 * (shape_label[:, i, :] > 0)
                        # correct_shape_bce += shape_pred.eq(bce_label).sum(dim=0)
                        cor_shape2 = shape_pred.eq(bce_label.view_as(sigshape))
                        # in order to count a glimpse as correctly labeled, the predicted glimpse vector must be approximately equal glimpse label
                        cor_shape_bce2 += sum([all(row) for row in cor_shape2])

                        pred_shape = torch.argmax(shape[:, :7], dim=1, keepdim=True)
                        correct_shape_ce += pred_shape.eq(ce_label.view_as(pred_shape)).sum(dim=0).item()
                    # this_shape_r2 += r2_score(shape_label[:, i, :].detach().cpu(), shape.detach().cpu())
                else:
                    shape_loss = -1
                    this_shape_loss += shape_loss
                    # this_shape_r2 += -1
            avg_shape_loss = this_shape_loss/seq_len
            # avg_shape_r2 = this_shape_r2/seq_len
            numb_loss = criterion(number, numb_label)
            numb_loss.backward()
            if 'shape' in use_loss:
                loss = numb_loss + avg_shape_loss
            else:
                loss = numb_loss
            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            # Take a gradient step
            optimizer.step()
            hidden.detach_()

            # Evaluate performance
            with torch.no_grad():
                train_loss += loss.item()
                numb_train_loss += numb_loss.item()
                shape_train_loss += avg_shape_loss
                # shape_train_r2 += avg_shape_r2
                numb_local = number
                pred = numb_local.argmax(dim=1, keepdim=True)
                n_correct += pred.eq(numb_label.view_as(pred)).sum().item()
            if config.debug:
                break
        if config.use_schedule:
            scheduler.step()

        # Evaluate performance
        train_loss /= len(loader)
        numb_train_loss /= len(loader)
        shape_train_loss /= len(loader)
        # shape_train_r2 /= len(loader)
        numb_acc = 100. * (n_correct / len(loader.dataset))
        shape_acc_bce = 100. * (correct_shape_bce / (len(loader.dataset)*seq_len))
        shape_acc_bce2 = 100. * (cor_shape_bce2 / (len(loader.dataset)*seq_len))
        shape_acc_ce = 100. * (correct_shape_ce / (len(loader.dataset)*seq_len))
        shape_acc_bce = shape_acc_bce.cpu().numpy()
        return train_loss, numb_train_loss, shape_train_loss, numb_acc, shape_acc_bce2, shape_acc_ce

    def test(loader, **kwargs):
        model.eval()
        n_correct = 0
        test_loss = 0
        numb_test_loss = 0
        shape_test_loss = 0
        # shape_test_r2 = torch.zeros((14,))
        correct_shape_bce = torch.zeros((14,), device=device)
        cor_shape_bce2 = 0
        correct_shape_ce = 0
        class_correct = [0. for i in range(nclasses)]
        class_total = [0. for i in range(nclasses)]
        classes = [0. for i in range(nclasses)]
        confusion_matrix = np.zeros((nclasses, nclasses))
        with torch.no_grad():
            for batch_idx, (pix, shape_label, numb_label) in enumerate(loader):
                batch_size = pix.shape[0]
                hidden = rnn.initHidden(batch_size)
                hidden = hidden.to(device)
                this_shape_loss = 0
                # this_shape_r2 = torch.zeros((14,))
                for i in range(seq_len):
                    this_input = pix[:, i, :]
                    number, shape, hidden = model(this_input, hidden, bce)
                    ce_label = torch.argmax(shape_label[:, i, :7], dim=1)
                    bce_label = 1.0 * (shape_label[:, i, :7] > 0)
                    if 'shape' in use_loss:
                        if bce:
                            # shape_loss = crit_bce(shape[:, :7], bce_label)
                            shape_loss = crit_bce(shape[:, :7], shape_label[:, i, :7])
                        else:
                            shape_loss = criterion(shape, ce_label)
                        this_shape_loss += shape_loss.item()
                        # this_shape_r2 += r2_score(shape_label[:, i, :].detach().cpu(), shape.detach().cpu())
                        with torch.no_grad():
                            sigshape = torch.sigmoid(shape[:, :7])
                            # this_shape_r2 += r2score(sigshape, shape_label[:, i, :]) #torchmetrics
                            shape_pred = torch.round(sigshape)
                            # bin_shape_label = 1 * (shape_label[:, i, :] > 0)
                            # correct_shape_bce += shape_pred.eq(bin_shape_label).sum(dim=0)
                            cor_shape2 = shape_pred.eq(bce_label.view_as(sigshape))
                            # in order to count a glimpse as correctly labeled, the predicted glimpse vector must be approximately equal glimpse label
                            cor_shape_bce2 += sum([all(row) for row in cor_shape2])
                            pred = torch.argmax(shape[:, :7], dim=1, keepdim=True)
                            correct_shape_ce += pred.eq(ce_label.view_as(pred)).sum(dim=0).item()

                    else:
                        shape_loss = -1
                        this_shape_loss += shape_loss
                        # this_shape_r2 += -1

                numb_loss = criterion(number, numb_label)
                avg_shape_loss = this_shape_loss / seq_len
                if 'shape' in config.use_loss:
                    loss = numb_loss + avg_shape_loss
                else:
                    loss = numb_loss

                # Evaluate performance
                test_loss += loss.item()
                shape_test_loss += avg_shape_loss
                # shape_test_r2 += this_shape_r2/seq_len
                numb_test_loss += numb_loss.item()
                numb_local = number
                pred = numb_local.argmax(dim=1, keepdim=True)
                n_correct += pred.eq(numb_label.view_as(pred)).sum().item()

                # class-specific analysis and confusion matrix
                c = (pred.squeeze() == numb_label)
                for i in range(c.shape[0]):
                    label = numb_label[i]
                    class_correct[label-2] += c[i].item()
                    class_total[label-2] += 1
                    confusion_matrix[label-2, pred[i]-2] += 1
                if config.debug:
                    break
            # Calculate average performance
            test_loss /= len(loader)
            numb_test_loss /= len(loader)
            shape_test_loss /= len(loader)
            # shape_test_r2 /= len(loader)
            shape_acc_bce = 100. * (correct_shape_bce / (len(loader.dataset)*seq_len))
            shape_acc_bce = shape_acc_bce.cpu().numpy()
            shape_acc_bce2 = 100. * (cor_shape_bce2 / (len(loader.dataset)*seq_len))
            shape_acc_ce = 100. * (correct_shape_ce / (len(loader.dataset)*seq_len))
            numb_acc = 100. * (n_correct / len(loader.dataset))
        return test_loss, numb_test_loss, shape_test_loss, numb_acc, shape_acc_bce2, shape_acc_ce, confusion_matrix
    # Evaluate performance before training
    # train_losses_test, numb_losses_test, shape_losses_test, numb_accs_test, shape_accs_bce_test, shape_accs_ce_test, _ = test(train_loader)
    train_losses[0], numb_losses[0], shape_losses[0], numb_accs[0], shape_accs_bce[0], shape_accs_ce[0], _ = test(train_loader)
    valid_losses[0], valid_numb_losses[0], valid_shape_losses[0], valid_numb_accs[0], valid_shape_accs_bce[0], valid_shape_accs_ce[0], _ = test(valid_loader)
    test_losses[0], test_numb_losses[0], test_shape_losses[0], test_numb_accs[0], test_shape_accs_bce[0], test_shape_accs_ce[0], _ = test(test_loader)
    print('Performance before Training')
    print(f'Train Loss (Total/Num/Shape): {train_losses[0]:.6}/{numb_losses[0]:.6}/{shape_losses[0]:.6}, \t Train Performance (Num/Shape) {numb_accs[0]:.3}%/{shape_accs_bce[0]:.3}%')
    print(f'Valid Loss (Total/Num/Shape): {valid_losses[0]:.6}/{valid_numb_losses[0]:.6}/{valid_shape_losses[0]:.6}, \t Valid Performance (Num/Shape) {valid_numb_accs[0]:.3}%/{valid_shape_accs_bce[0]:.3}%')
    print(f'Test Loss (Total/Num/Shape): {test_losses[0]:.6}/{test_numb_losses[0]:.6}/{test_shape_losses[0]:.6}, \t Test Performance (Num/Shape) {test_numb_accs[0]:.3}%/{test_shape_accs_bce[0]:.3}%')

    for ep in range(1, n_epochs + 1):
        print(f'Epoch {ep}, Learning rate: {optimizer.param_groups[0]["lr"]}')

        train_loss, num_loss, shape_loss, train_num_acc, train_shape_acc_bce, train_shape_acc_ce = train(train_loader, ep)
        val_loss, val_num_loss, val_shape_loss, val_num_acc, val_shape_acc_bce, val_shape_acc_ce, val_conf_mat = test(valid_loader, which_set='valid')
        test_loss, test_num_loss, test_shape_loss, test_num_acc, te_shape_acc_bce, te_shape_acc_ce, test_conf_mat = test(test_loader, which_set='test')

        train_losses[ep] = train_loss
        numb_losses[ep] = num_loss
        shape_losses[ep] = shape_loss
        shape_accs_bce[ep] = train_shape_acc_bce
        shape_accs_ce[ep] = train_shape_acc_ce
        numb_accs[ep] = train_num_acc
        valid_losses[ep] = val_loss
        valid_numb_losses[ep] = val_num_loss
        valid_shape_losses[ep] = val_shape_loss
        valid_shape_accs_bce[ep] = val_shape_acc_bce
        valid_shape_accs_ce[ep] = val_shape_acc_ce
        valid_numb_accs[ep] = val_num_acc
        valid_conf_mats[ep, :, :] = val_conf_mat
        test_losses[ep] = test_loss
        test_numb_losses[ep] = test_num_loss
        test_shape_losses[ep] = test_shape_loss
        test_shape_accs_bce[ep] = te_shape_acc_bce
        test_shape_accs_ce[ep] = te_shape_acc_ce
        test_numb_accs[ep] = test_num_acc
        test_conf_mats[ep, :, :] = test_conf_mat
        pct_done = round(100. * (ep / n_epochs))
        if not ep % 5 or ep < 6:
            # Print and save performance
            # print(f'Progress {pct_done}% trained: \t Loss (Map/Numb): {map_loss.item():.6}/{number_loss.item():.6}, \t Accuracy: {accs[0]:.3}% {accs[1]:.3}% {accs[2]:.3}% {accs[3]:.3}% {accs[4]:.3}% {accs[5]:.3}% {accs[6]:.3}% \t Number {numb_acc:.3}% \t Mean Map Acc {accs.mean():.3}%')
            print(f'Epoch {ep}, Progress {pct_done}% trained')
            print(f'Train Loss (Total/Num/Shape): {train_losses[ep]:.6}/{numb_losses[ep]:.6}/{shape_losses[ep]:.6}, \t Train Performance (Num/Shape) {numb_accs[ep]:.3}%/{shape_accs_bce[ep]:.3}%')
            print(f'Valid Loss (Total/Num/Shape): {valid_losses[ep]:.6}/{valid_numb_losses[ep]:.6}/{valid_shape_losses[ep]:.6}, \t Valid Performance (Num/Shape) {valid_numb_accs[ep]:.3}%/{valid_shape_accs_bce[ep]:.3}%')
            print(f'Test Loss (Total/Num/Shape): {test_losses[ep]:.6}/{test_numb_losses[ep]:.6}/{test_shape_losses[ep]:.6}, \t Test Performance (Num/Shape) {test_numb_accs[ep]:.3}%/{test_shape_accs_bce[ep]:.3}%')

            np.savez('results/'+filename,
                     train_loss=train_losses,
                     numb_loss=numb_losses,
                     shape_loss=shape_losses,
                     shape_acc_bce=shape_accs_bce,
                     shape_acc_ce=shape_accs_ce,
                     numb_acc=numb_accs,
                     valid_loss=valid_losses,
                     valid_numb_loss=valid_numb_losses,
                     valid_shape_loss=valid_shape_losses,
                     valid_shape_acc_bce=valid_shape_accs_bce,
                     valid_shape_acc_ce=valid_shape_accs_ce,
                     valid_numb_acc=valid_numb_accs,
                     valid_conf_mat=valid_conf_mats,
                     test_loss=test_losses,
                     test_numb_loss=test_numb_losses,
                     test_shape_loss=test_shape_losses,
                     test_shape_acc_bce=test_shape_accs_bce,
                     test_shape_acc_ce=test_shape_accs_ce,
                     test_numb_acc=test_numb_accs,
                     test_conf_mat=test_conf_mats)
            torch.save(model, f'models/{filename}.pt')

            # Plot performance
            kwargs_train = {'alpha': 0.8, 'color': 'blue'}
            kwargs_val = {'alpha': 0.8, 'color': 'cyan'}
            kwargs_test = {'alpha': 0.8, 'color': 'red'}
            row, col = 2, 2
            fig, ax = plt.subplots(row, col, figsize=(6*col, 6*row))
            plt.suptitle(f'{config.model_version}, nonlin={config.rnn_act}, dr-rnn={config.drop_rnn}, dr-ro={config.drop_readout}, seq not shuffled  \n {preglimpsed}', fontsize=18)
            ax[0, 0].set_title('Number Loss')
            ax[0, 0].plot(test_numb_losses[:ep+1], label='test number loss', **kwargs_test)
            ax[0, 0].plot(valid_numb_losses[:ep+1], label='valid number loss', **kwargs_val)
            ax[0, 0].plot(numb_losses[:ep+1], label='train number loss', **kwargs_train)
            ax[0, 0].set_ylabel('Cross Entropy Loss')
            ax[0, 1].set_title('Number Accuracy')
            ax[0, 1].set_ylim([0, 101])
            ax[0, 1].plot(test_numb_accs[:ep+1], label='test number acc', **kwargs_test)
            ax[0, 1].plot(valid_numb_accs[:ep+1], label='valid number acc', **kwargs_val)
            ax[0, 1].plot(numb_accs[:ep+1], label='train number acc', **kwargs_train)

            ax[1, 0].set_title('Shape Loss')
            ax[1, 0].plot(test_shape_losses[:ep+1], label='test shape loss', **kwargs_test)
            ax[1, 0].plot(valid_shape_losses[:ep+1], label='valid shape loss', **kwargs_val)
            ax[1, 0].plot(shape_losses[:ep+1], label='train shape loss', **kwargs_train)
            # ax[1, 0].set_ylabel('Binary Cross Entropy Loss')
            ax[1, 1].set_title('Shape Accuracy')
            ax[1, 1].set_ylim([0, 101])
            # if bce:
                # n_eps = ep + 1
                # epochs = [i for i in range(n_eps)]
                # test_acc_data = {'accuracy':test_shape_accs_bce[:n_eps, :7].flatten(order='F'),
                #                  'epoch':epochs*7,
                #                  'shape':['arrow']*n_eps + ['circle']*n_eps + ['diamond']*n_eps + ['drop']*n_eps + ['halfmoon']*n_eps + ['heart']*n_eps + ['lightning']*n_eps,
                #                  'dataset': 'test'
                #                  }
                # df_test = pd.DataFrame(test_acc_data)
                # valid_acc_data = {'accuracy':valid_shape_accs_bce[:n_eps, :7].flatten(order='F'),
                #                  'epoch':epochs*7,
                #                  'shape':['arrow']*n_eps + ['circle']*n_eps + ['diamond']*n_eps + ['drop']*n_eps + ['halfmoon']*n_eps + ['heart']*n_eps + ['lightning']*n_eps,
                #                  'dataset': 'valid'
                #                  }
                # df_valid = pd.DataFrame(valid_acc_data)
                # train_acc_data = {'accuracy':shape_accs_bce[:n_eps, :7].flatten(order='F'),
                #                  'epoch':epochs*7,
                #                  'shape':['arrow']*n_eps + ['circle']*n_eps + ['diamond']*n_eps + ['drop']*n_eps + ['halfmoon']*n_eps + ['heart']*n_eps + ['lightning']*n_eps,
                #                  'dataset': 'train'
                #                  }
                # df_train = pd.DataFrame(train_acc_data)
                # df = pd.concat((df_test, df_valid, df_train), ignore_index=True)
                # sns.lineplot(ax=ax[1, 1], data=df, y='accuracy', x='epoch', hue='shape', style='dataset')
            ax[1, 1].plot(test_shape_accs_bce[:ep+1], label='test shape acc bce', **kwargs_test)
            ax[1, 1].plot(valid_shape_accs_bce[:ep+1], label='valid shape acc bce', **kwargs_val)
            ax[1, 1].plot(shape_accs_bce[:ep+1], label='train shape acc bce', **kwargs_train)
            # else:
            ax[1, 1].plot(test_shape_accs_ce[:ep+1], '--', label='test shape acc ce', **kwargs_test)
            ax[1, 1].plot(valid_shape_accs_ce[:ep+1], '--', label='valid shape acc ce', **kwargs_val)
            ax[1, 1].plot(shape_accs_ce[:ep+1], '--', label='train shape acc ce', **kwargs_train)

            for axes in ax.flatten():
                axes.legend()
                axes.grid(linestyle='--')
                axes.set_xlabel('Epochs')
                axes.set_xticks(np.arange(ep+1))
            plt.savefig(f'figures/{model_version}/{filename}_results.png', dpi=300)
            plt.close()

    print(f'Final performance:')
    print(f'Train Loss (Total/Num/Shape): {train_losses[ep]:.6}/{numb_losses[ep]:.6}/{shape_losses[ep]:.6}, \t Train Performance (Num/Shape) {numb_accs[ep]:.3}%/{shape_accs_ce[ep]:.3}')
    print(f'Valid Loss (Total/Num/Shape): {valid_losses[ep]:.6}/{valid_numb_losses[ep]:.6}/{valid_shape_losses[ep]:.6}, \t Valid Performance (Num/Shape) {valid_numb_accs[ep]:.3}%/{valid_shape_accs_ce[ep]:.3}')
    print(f'Test Loss (Total/Num/Shape): {test_losses[ep]:.6}/{test_numb_losses[ep]:.6}/{test_shape_losses[ep]:.6}, \t Test Performance (Num/Shape) {test_numb_accs[ep]:.3}%/{test_shape_accs_ce[ep]:.3}')

    np.savez('results/'+filename,
             train_loss=train_losses,
             numb_loss=numb_losses,
             shape_loss=shape_losses,
             shape_acc_bce=shape_accs_bce,
             shape_acc_ce=shape_accs_ce,
             numb_acc=numb_accs,
             valid_loss=valid_losses,
             valid_numb_loss=valid_numb_losses,
             valid_shape_loss=valid_shape_losses,
             valid_shape_acc_bce=valid_shape_accs_bce,
             valid_shape_acc_ce=valid_shape_accs_ce,
             valid_numb_acc=valid_numb_accs,
             valid_conf_mat=valid_conf_mats,
             test_loss=test_losses,
             test_numb_loss=test_numb_losses,
             test_shape_loss=test_shape_losses,
             test_shape_acc_bce=test_shape_accs_bce,
             test_shape_acc_ce=test_shape_accs_ce,
             test_numb_acc=test_numb_accs,
             test_conf_mat=test_conf_mats)
    torch.save(model, f'models/{filename}.pt')

    # Plot confusion matrices
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(val_conf_mat[:, :])
    ax1.set_title('Confusion Matrix (Validation)')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Estimated Class')
    ax1.set_xticks([0, 1, 2, 3, 4, 5], [2, 3, 4, 5, 6, 7])
    ax1.set_yticks([0, 1, 2, 3, 4, 5], [2, 3, 4, 5, 6, 7])
    # ax1.set_xticklabels([2, 3, 4, 5, 6, 7])
    # ax1.set_yticklabels([2, 3, 4, 5, 6, 7])
    ax2.imshow(test_conf_mat[:, :])
    ax2.set_title('Confusion Matrix (Test)')
    ax2.set_xlabel('Estimated Class')
    ax2.set_xticks([0, 1, 2, 3, 4, 5], [2, 3, 4, 5, 6, 7])
    ax2.set_yticks([0, 1, 2, 3, 4, 5], [2, 3, 4, 5, 6, 7])
    # ax2.set_xticklabels([2, 3, 4, 5, 6, 7])
    # ax2.set_yticklabels([2, 3, 4, 5, 6, 7])
    plt.savefig(f'figures/{filename}_confusion.png', dpi=300)



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
    clip_grad_norm = 2

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
        dir = 'preglimpsed_sequences/'
        num = np.load(f'{dir}number_{preglimpsed}.npy')
        num = torch.from_numpy(num).long()
        num = num.to(device)
        shape = np.load(f'{dir}shape_{preglimpsed}.npy')
        # shape = torch.from_numpy(shape).long()
        shape = torch.from_numpy(shape).float()
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
    if not os.path.isdir(f'figures/{model_version}'):
        os.mkdir(f'figures/{model_version}')
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
    kwargs = {'act': rnn_act, 'eye_weight': eye_weight, 'detached': detached,
              'drop_rnn': config.drop_rnn, 'drop_readout': config.drop_readout,
              'rotate': config.rotate, 'device': config.device}
    if config.train_on == 'pix':
        input_size = 1152
    elif config.train_on == 'loc':
        input_size = map_size
    elif config.train_on == 'both':
        input_size = map_size + 1152
    hidden_size = int(np.round(input_size*1.1))
    # hidden_size = map_size*2
    if model_version == 'two_step':
        model = models.TwoStepModel(input_size, hidden_size, map_size, numb_size, **kwargs)
    elif 'conv' in model_version:
        if 'big' in model_version:
            model = models.ConvReadoutMapNet(input_size, hidden_size, map_size, numb_size, big=True, **kwargs)
        else:
            model = models.ConvReadoutMapNet(input_size, hidden_size, map_size, numb_size, **kwargs)
    elif model_version == 'integrated':
        model = models.IntegratedModel(input_size, hidden_size, map_size, numb_size, **kwargs)
    elif model_version == 'three_step':
        model = models.ThreeStepModel(input_size, hidden_size, map_size, numb_size, **kwargs)
    elif model_version == 'content_gated':
        model = models.ContentGated_cheat(hidden_size, map_size, numb_size, **kwargs)
    elif model_version == 'two_step_ws':
        if 'detached' in use_loss:
            model = models.TwoStepModel_weightshare_detached(hidden_size, map_size, numb_size, init=False, **kwargs)
        else:
            model = models.TwoStepModel_weightshare(hidden_size, map_size, numb_size, init=False, **kwargs)
    elif model_version == 'two_step_ws_init':
        if 'detached' in use_loss:
            model = models.TwoStepModel_weightshare_detached(hidden_size, map_size, numb_size, init=True, **kwargs)
        else:
            model = models.TwoStepModel_weightshare(hidden_size, map_size, numb_size, init=True, **kwargs)
    elif 'one_step' in model_version:
        if use_loss == 'number':
            model = models.RNN(map_size, hidden_size, numb_size, **kwargs)
        elif use_loss == 'map':
            model = models.RNN(map_size, hidden_size, map_size, **kwargs)
    elif model_version == 'number_as_sum':
        model = models.NumberAsMapSum(map_size, hidden_size)
    elif model_version == 'cheat':
        model = models.DistinctiveCheat(**kwargs)
    elif model_version == 'cheat_small':
        model = models.DistinctiveCheat_small(**kwargs)
    elif model_version == 'distinctive':
        hidden_size = int(120*1.1)
        model = models.Distinctive(hidden_size, **kwargs)
    elif model_version == 'pixel+shape':
        hidden_size = int(120*1.1)
        model = models.PixelPlusShapeModel(hidden_size, **kwargs)
    elif model_version == 'shape':
        hidden_size = int(120*1.1)
        model = models.ShapeModel(hidden_size, **kwargs)
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
        start_lr = 0.05
        # Can decay lr quicker when starting with the rnn weights pretrained
        scale = 0.9955 if config.pretrained_map else 0.9978
        # 0.8 ** 200
        if 'detached' in use_loss:
            opt_rnn = torch.optim.SGD(model.rnn.parameters(), lr=start_lr, momentum=mom, weight_decay=wd_rnn)
            if 'two' in model_version or model_version == 'integrated':
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
            # scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda1)
            scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=config.n_epochs/10, gamma=0.5)
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
    elif 'cheat' in model_version:
        train_nomap_model(model, opt, config, scheduler)
    elif model_version == 'distinctive' or model_version == 'pixel+shape' or model_version == 'shape':
        train_shape_number(model, opt, config, scheduler)
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
    parser.add_argument('--drop_readout', type=float, default=0.5)
    parser.add_argument('--drop_rnn', type=float, default=0.1)
    parser.add_argument('--wd', type=float, default=0) # 1e-6
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--train_on', type=str, default='loc')  ## loc, pix, or both
    parser.add_argument('--n_epochs', type=int, default=2000)
    parser.add_argument('--pretrained_map', action='store_true', default=False)
    parser.add_argument('--rotate', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--bce', action='store_true', default=False)
    config = parser.parse_args()
    print(config)
    return config

if __name__ == '__main__':
    config = get_config()
    main(config)
