"""Train simple models on data synthesized from toy model."""
import os
import argparse
from itertools import product
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.io import savemat
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR
from prettytable import PrettyTable
import models_toy as mod
import ventral_models as vmod
from models import ConvNet
import toy_model_data as toy
from count_unique import Timer
import tetris as pseudo
import numeric
# from labml_nn.hypernetworks.hyper_lstm import HyperLSTM
# from hypernet import HyperLSTM
# torch.set_num_threads(15)

mom = 0.9
wd = 0
start_lr = 0.1
BATCH_SIZE = 500
TRAIN_SHAPES = [0,  2,  4,  5,  8,  9, 14, 15, 16]

device = torch.device("cuda")
# device = torch.device("cpu")
criterion = nn.CrossEntropyLoss()
criterion_noreduce = nn.CrossEntropyLoss(reduction='none')
criterion_mse = nn.MSELoss()
criterion_mse_noreduce = nn.MSELoss(reduction='none')


def train_model(rnn, optimizer, scheduler, loaders, config):
    base_name = config.base_name
    avg_num_objects = config.max_num - ((config.max_num-config.min_num)/2)
    n_locs = config.grid**2
    weight_full = (n_locs - avg_num_objects)/ (avg_num_objects+2) # 9 for 9 locations
    weight_count = (n_locs - avg_num_objects)/ avg_num_objects
    pos_weight_count = torch.ones([n_locs], device=device) * weight_count
    pos_weight_full = torch.ones([n_locs], device=device) * weight_full
    criterion_bce_full = nn.BCEWithLogitsLoss(pos_weight=pos_weight_full)
    criterion_bce_count = nn.BCEWithLogitsLoss(pos_weight=pos_weight_count)
    criterion_bce_full_noreduce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_full, reduction='none')
    criterion_bce_count_noreduce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_count, reduction='none')
    n_glimpses = config.n_glimpses
    # n_glimpses = config.max_num
    # if config.distract or config.distract_corner or config.random:
    # if config.challenge != '':
        # n_glimpses += 2
    n_epochs = config.n_epochs
    recurrent_iterations = config.n_iters
    cross_entropy = config.cross_entropy
    nonsymbolic = True if config.shape_input == 'parametric' else False
    learn_shape = config.learn_shape
    train_loader, test_loaders = loaders

    train_loss = np.zeros((n_epochs,))
    # train_map_loss = np.zeros((n_epochs,))
    train_count_map_loss = np.zeros((n_epochs,))
    train_dist_map_loss = np.zeros((n_epochs,))
    train_full_map_loss = np.zeros((n_epochs,))
    train_count_num_loss = np.zeros((n_epochs,))
    train_dist_num_loss = np.zeros((n_epochs,))
    train_all_num_loss = np.zeros((n_epochs,))
    train_sh_loss = np.zeros((n_epochs,))
    train_acc_count = np.zeros((n_epochs,))
    train_acc_dist = np.zeros((n_epochs,))
    train_acc_all = np.zeros((n_epochs,))
    n_test_sets = len(config.test_shapes) * len(config.lum_sets)
    test_loss = [np.zeros((n_epochs,)) for _ in range(n_test_sets)]
    # test_map_loss = [np.zeros((n_epochs,)) for _ in range(n_test_sets)]
    test_full_map_loss = [np.zeros((n_epochs,)) for _ in range(n_test_sets)]
    test_count_map_loss = [np.zeros((n_epochs,)) for _ in range(n_test_sets)]
    test_dist_map_loss = [np.zeros((n_epochs,)) for _ in range(n_test_sets)]
    test_count_num_loss = [np.zeros((n_epochs,)) for _ in range(n_test_sets)]
    test_dist_num_loss = [np.zeros((n_epochs,)) for _ in range(n_test_sets)]
    test_all_num_loss = [np.zeros((n_epochs,)) for _ in range(n_test_sets)]
    test_sh_loss = [np.zeros((n_epochs,)) for _ in range(n_test_sets)]
    test_acc_count = [np.zeros((n_epochs,)) for _ in range(n_test_sets)]
    test_acc_dist = [np.zeros((n_epochs,)) for _ in range(n_test_sets)]
    test_acc_all = [np.zeros((n_epochs,)) for _ in range(n_test_sets)]
    # columns = ['pass count', 'correct', 'predicted', 'true', 'loss', 'num loss', 'full map loss', 'count map loss', 'shape loss', 'epoch', 'train shapes', 'test shapes']

    def train_nosymbol(loader):
        rnn.train()
        correct = 0
        epoch_loss = 0
        num_epoch_loss = 0
        map_epoch_loss = 0
        shape_epoch_loss = 0
        for i, (xy, shape, target, locations, shape_label, _) in enumerate(loader):
            n_glimpses = xy.shape[1]
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
                    pred_num, map, pred_shape, hidden = rnn(xy[:, t, :], shape[:, t, :, :], hidden)

                    if learn_shape:
                        shape_loss = criterion_mse(pred_shape, shape_label[:, t, :])*10
                        shape_loss.backward(retain_graph=True)
                        shape_epoch_loss += shape_loss.item()
            # Calculate lossees
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
        accuracy = 100. * (correct/len(loader.dataset))
        epoch_loss /= len(loader)
        num_epoch_loss /= len(loader)
        map_epoch_loss /= len(loader)
        shape_epoch_loss /= (len(loader) * n_glimpses)
        return epoch_loss, num_epoch_loss, accuracy, map_epoch_loss, shape_epoch_loss

    def train(loader, ep):
        rnn.train()
        correct = 0
        epoch_loss = 0
        num_epoch_loss = 0
        # map_epoch_loss = 0
        count_map_epoch_loss = 0
        shape_epoch_loss = 0
        no_shuffle = ['recurrent_control', 'cnn', 'feedforward']
        for i, (input, target, num_dist, locations, shape_label, _) in enumerate(loader):
            assert all(locations.sum(dim=1) == target)
            n_glimpses = input.shape[1]

            if config.model_type not in no_shuffle:
                seq_len = input.shape[1]
                # Shuffle glimpse order on each batch
                for i, row in enumerate(input):
                    input[i, :, :] = row[torch.randperm(seq_len), :]
            input_dim = input.shape[0]

            rnn.zero_grad()
            if 'cnn' not in config.model_type and 'feedforward' not in config.model_type or 'pretrained' in config.model_type:
                hidden = rnn.initHidden(input_dim)
                if config.model_type != 'hyper':
                    hidden = hidden.to(device)
            # data = toy.generate_dataset(BATCH_SIZE)
            # # data.loc[0]
            # xy = torch.tensor(data['xy']).float()
            # shape = torch.tensor(data['shape']).float()
            # target = torch.tensor(data['numerosity']).long()

            if ('cnn' in config.model_type or 'feedforward' in config.model_type) and 'pretrained' not in config.model_type:
                pred_num, map = rnn(input)
            else:
                for _ in range(recurrent_iterations):
                    for t in range(n_glimpses):
                        if config.model_type == 'recurrent_control':
                            pred_num, map, hidden = rnn(input, hidden)
                        else:
                            pred_num, pred_shape, map, hidden, _, _ = rnn(input[:, t, :], hidden)
                            if learn_shape:
                                shape_loss_mse = criterion_mse(pred_shape, shape_label[:, t, :])*10
                                shape_loss_ce = criterion(pred_shape, shape_label[:, t, :])
                                shape_loss = shape_loss_mse + shape_loss_ce
                                shape_epoch_loss += shape_loss.item()
                                shape_loss.backward(retain_graph=True)
                            else:
                                shape_epoch_loss += -1
            if cross_entropy:
                num_loss = criterion(pred_num, target)
                pred = pred_num.argmax(dim=1, keepdim=True)
            else:
                num_loss = criterion_mse(torch.squeeze(pred_num), target)
                pred = torch.round(pred_num)

            def get_map_loss():
                map_loss = criterion_bce_full(map, locations)
                map_loss_to_add = map_loss.item()

                return map_loss, map_loss_to_add
            if config.use_loss == 'num':
                loss = num_loss
                # count_map_loss_to_add = -1
                # all_map_loss_to_add  = -1
                # map_loss = -1
                map_loss_to_add = -1
            elif config.use_loss == 'map':
                # map_loss, map_loss_to_add = get_map_loss()
                map_loss, map_loss_to_add = get_map_loss()
                loss = map_loss
            elif config.use_loss == 'both':
                # map_loss, map_loss_to_add = get_map_loss()
                map_loss, map_loss_to_add = get_map_loss()
                loss = num_loss + map_loss
            elif config.use_loss == 'map_then_both':
                # map_loss, map_loss_to_add = get_map_loss()
                map_loss, map_loss_to_add = get_map_loss()
                # map_loss_to_add = count_map_loss
                if ep < 100:
                    loss = map_loss
                else:
                    loss = num_loss + map_loss

            loss.backward()
            nn.utils.clip_grad_norm_(rnn.parameters(), 2)
            optimizer.step()

            correct += pred.eq(target.view_as(pred)).sum().item()
            epoch_loss += loss.item()
            num_epoch_loss += num_loss.item()
            # if not isinstance(map_loss_to_add, int):
                # map_loss_to_add = map_loss_to_add.item()
            count_map_epoch_loss += map_loss_to_add
            # count_map_epoch_loss += count_map_loss_to_add
        scheduler.step()
        accuracy = 100. * (correct/len(loader.dataset))
        epoch_loss /= len(loader)
        num_epoch_loss /= len(loader)
        # full_map_epoch_loss /= len(loader)
        count_map_epoch_loss /= len(loader)
        map_epoch_loss = (count_map_epoch_loss, -1)
        shape_epoch_loss /= len(loader) * n_glimpses
        return epoch_loss, num_epoch_loss, accuracy, shape_epoch_loss, map_epoch_loss

    def train2map(loader, ep):
        rnn.train()
        correct_count = 0
        correct_dist = 0
        correct_all = 0
        epoch_loss = 0
        count_num_epoch_loss = 0
        dist_num_epoch_loss = 0
        all_num_epoch_loss = 0
        full_map_epoch_loss = 0
        count_map_epoch_loss = 0
        dist_map_epoch_loss = 0
        shape_epoch_loss = 0
        no_shuffle = ['recurrent_control', 'cnn', 'feedforward']
        # for i, (input, target, all_loc, count_loc, shape_label, _) in enumerate(loader):
        for i, (input, count_num, dist_num, count_loc, dist_loc, shape_label, _) in enumerate(loader):
            n_glimpses = input.shape[1]
            # if type(locations) is tuple:
            #     (all_loc, count_loc) = locations
            # else:
            #     count_loc = locations
            if config.model_type not in no_shuffle:
                seq_len = input.shape[1]
                # Shuffle glimpse order on each batch
                for i, row in enumerate(input):
                    input[i, :, :] = row[torch.randperm(seq_len), :]
            input_dim = input.shape[0]

            rnn.zero_grad()
            if 'cnn' not in config.model_type:
                hidden = rnn.initHidden(input_dim)
                if config.model_type != 'hyper':
                    hidden = hidden.to(device)
            # data = toy.generate_dataset(BATCH_SIZE)
            # # data.loc[0]
            # xy = torch.tensor(data['xy']).float()
            # shape = torch.tensor(data['shape']).float()
            # target = torch.tensor(data['numerosity']).long()
            if 'cnn' in config.model_type:
                pred_num, map = rnn(input)
            else:
                for _ in range(recurrent_iterations):
                    for t in range(n_glimpses):
                        if config.model_type == 'recurrent_control':
                            pred_num, map, hidden = rnn(input, hidden)
                        else:
                            pred_num, pred_shape, map, hidden = rnn(input[:, t, :], hidden)
                            if learn_shape:
                                shape_loss = criterion_mse(pred_shape, shape_label[:, t, :])*10
                                shape_epoch_loss += shape_loss.item()
                                shape_loss.backward(retain_graph=True)
                            else:
                                shape_epoch_loss += -1
            if type(pred_num) is tuple:
                (pred_count_num, pred_dist_num, pred_all_num) = pred_num
            else:
                pred_num_d = None
                pred_all_num = None
            if type(map) is tuple:
                # (all_map, count_map) = map
                (pred_count_map, pred_dist_map, pred_all_map) = map
            else:
                pred_count_map = map
                pred_dist_map = None
                pred_all_map = None
            if cross_entropy:
                count_num_loss = criterion(pred_count_num, count_num)
                dist_num_loss = criterion(pred_dist_num, dist_num)
                all_num_loss = criterion(pred_all_num, torch.add(count_num, dist_num))
                num_loss = count_num_loss + dist_num_loss + all_num_loss
                pred_count = pred_count_num.argmax(dim=1, keepdim=True)
                pred_dist = pred_dist_num.argmax(dim=1, keepdim=True)
                pred_all = pred_all_num.argmax(dim=1, keepdim=True)
            else:
                num_loss = criterion_mse(torch.squeeze(pred_num), target)
                pred_count = torch.round(pred_num)

            def get_map_loss():
                count_map_loss = criterion_bce_count(pred_count_map, count_loc)
                map_loss_to_add = count_map_loss
                map_loss = count_map_loss
                if pred_dist_map is not None:
                    dist_map_loss = criterion_bce_full(pred_dist_map, dist_loc)
                    map_loss += dist_map_loss
                if pred_all_map is not None:
                    all_map_loss = criterion_bce_full(pred_all_map, dist_loc)
                    map_loss += all_map_loss
                return count_map_loss, dist_map_loss, all_map_loss, map_loss

            if config.use_loss == 'num':
                loss = num_loss
                count_map_loss_to_add = -1
                all_map_loss_to_add  = -1
            elif config.use_loss == 'map':
                # map_loss, map_loss_to_add = get_map_loss()
                count_map_loss, dist_map_loss, all_map_loss, map_loss = get_map_loss()
                count_map_loss_to_add = count_map_loss.item()
                dist_map_loss_to_add = dist_map_loss.item()
                all_map_loss_to_add = all_map_loss.item()
                loss = map_loss
            elif config.use_loss == 'both':
                # map_loss, map_loss_to_add = get_map_loss()
                count_map_loss, dist_map_loss, all_map_loss, map_loss = get_map_loss()
                count_map_loss_to_add = count_map_loss.item()
                dist_map_loss_to_add = dist_map_loss.item()
                all_map_loss_to_add = all_map_loss.item()
                loss = num_loss + map_loss
            elif config.use_loss == 'map_then_both':
                # map_loss, map_loss_to_add = get_map_loss()
                count_map_loss, dist_map_loss, all_map_loss, map_loss = get_map_loss()
                count_map_loss_to_add = count_map_loss.item()
                dist_map_loss_to_add = dist_map_loss.item()
                all_map_loss_to_add = all_map_loss.item()
                # map_loss_to_add = count_map_loss
                if ep < 100:
                    loss = map_loss
                else:
                    loss = num_loss + map_loss

            loss.backward()
            nn.utils.clip_grad_norm_(rnn.parameters(), 2)
            optimizer.step()

            # correct += pred.eq(target.view_as(pred)).sum().item()
            correct_count += pred_count.eq(count_num.view_as(pred_count)).sum().item()
            correct_dist += pred_dist.eq(dist_num.view_as(pred_dist)).sum().item()
            all_num = torch.add(count_num, dist_num)
            correct_all += pred_all.eq(all_num.view_as(pred_all)).sum().item()
            epoch_loss += loss.item()
            count_num_epoch_loss += count_num_loss.item()
            dist_num_epoch_loss += dist_num_loss.item()
            all_num_epoch_loss += all_num_loss.item()
            # if not isinstance(map_loss_to_add, int):
                # map_loss_to_add = map_loss_to_add.item()
            full_map_epoch_loss += all_map_loss_to_add
            count_map_epoch_loss += count_map_loss_to_add
            dist_map_epoch_loss += dist_map_loss_to_add
        scheduler.step()
        accuracy_count = 100. * (correct_count/len(loader.dataset))
        accuracy_dist = 100. * (correct_dist/len(loader.dataset))
        accuracy_all = 100. * (correct_all/len(loader.dataset))
        accuracy = (accuracy_count, accuracy_dist, accuracy_all)
        epoch_loss /= len(loader)
        count_num_epoch_loss /= len(loader)
        dist_num_epoch_loss /= len(loader)
        all_num_epoch_loss /= len(loader)
        num_epoch_loss = (count_num_epoch_loss, dist_num_epoch_loss, all_num_epoch_loss)
        count_map_epoch_loss /= len(loader)
        dist_map_epoch_loss /= len(loader)
        full_map_epoch_loss /= len(loader)
        map_epoch_loss = (count_map_epoch_loss, dist_map_epoch_loss, full_map_epoch_loss)
        shape_epoch_loss /= len(loader) * n_glimpses
        return epoch_loss, num_epoch_loss, accuracy, shape_epoch_loss, map_epoch_loss

    def train_glimpse(loader, ep):
        rnn.train()
        correct = 0
        epoch_loss = 0
        num_epoch_loss = 0
        # map_epoch_loss = 0
        count_map_epoch_loss = 0
        shape_epoch_loss = 0
        for i, (image, saliency, target, num_dist, locations, shape_label, _) in enumerate(loader):
            assert all(locations.sum(dim=1) == target)
            input_dim = image.shape[0]

            rnn.zero_grad()
            if 'cnn' not in config.model_type and 'feedforward' not in config.model_type:
                hidden = rnn.initHidden(input_dim)
                if config.model_type != 'hyper':
                    hidden = hidden.to(device)
            # data = toy.generate_dataset(BATCH_SIZE)
            # # data.loc[0]
            # xy = torch.tensor(data['xy']).float()
            # shape = torch.tensor(data['shape']).float()
            # target = torch.tensor(data['numerosity']).long()

            if 'cnn' in config.model_type or 'feedforward' in config.model_type:
                pred_num, map = rnn(input)
            else:
                for _ in range(recurrent_iterations):
                    for t in range(n_glimpses):
                        if config.model_type == 'recurrent_control':
                            pred_num, map, hidden = rnn(input, hidden)
                        else:
                            pred_num, pred_shape, map, hidden = rnn(image, saliency, hidden)
                            if learn_shape:
                                shape_loss_mse = criterion_mse(pred_shape, shape_label[:, t, :])*10
                                shape_loss_ce = criterion(pred_shape, shape_label[:, t, :])
                                shape_loss = shape_loss_mse + shape_loss_ce
                                shape_epoch_loss += shape_loss.item()
                                shape_loss.backward(retain_graph=True)
                            else:
                                shape_epoch_loss += -1
            if cross_entropy:
                num_loss = criterion(pred_num, target)
                pred = pred_num.argmax(dim=1, keepdim=True)
            else:
                num_loss = criterion_mse(torch.squeeze(pred_num), target)
                pred = torch.round(pred_num)

            def get_map_loss():
                map_loss = criterion_bce_full(map, locations)
                map_loss_to_add = map_loss.item()

                return map_loss, map_loss_to_add
            if config.use_loss == 'num':
                loss = num_loss
                # count_map_loss_to_add = -1
                # all_map_loss_to_add  = -1
                # map_loss = -1
                map_loss_to_add = -1
            elif config.use_loss == 'map':
                # map_loss, map_loss_to_add = get_map_loss()
                map_loss, map_loss_to_add = get_map_loss()
                loss = map_loss
            elif config.use_loss == 'both':
                # map_loss, map_loss_to_add = get_map_loss()
                map_loss, map_loss_to_add = get_map_loss()
                loss = num_loss + map_loss
            elif config.use_loss == 'map_then_both':
                # map_loss, map_loss_to_add = get_map_loss()
                map_loss, map_loss_to_add = get_map_loss()
                # map_loss_to_add = count_map_loss
                if ep < 100:
                    loss = map_loss
                else:
                    loss = num_loss + map_loss

            loss.backward()
            nn.utils.clip_grad_norm_(rnn.parameters(), 2)
            optimizer.step()

            correct += pred.eq(target.view_as(pred)).sum().item()
            epoch_loss += loss.item()
            num_epoch_loss += num_loss.item()
            # if not isinstance(map_loss_to_add, int):
                # map_loss_to_add = map_loss_to_add.item()
            count_map_epoch_loss += map_loss_to_add
            # count_map_epoch_loss += count_map_loss_to_add
        scheduler.step()
        accuracy = 100. * (correct/len(loader.dataset))
        epoch_loss /= len(loader)
        num_epoch_loss /= len(loader)
        # full_map_epoch_loss /= len(loader)
        count_map_epoch_loss /= len(loader)
        map_epoch_loss = (count_map_epoch_loss, -1)
        shape_epoch_loss /= len(loader) * n_glimpses
        return epoch_loss, num_epoch_loss, accuracy, shape_epoch_loss, map_epoch_loss

    def test_nosymbol(loader, epoch):
        rnn.eval()
        n_correct = 0
        epoch_loss = 0
        num_epoch_loss = 0
        map_epoch_loss = 0
        test_results = pd.DataFrame()
        for i, (xy, shape, target, locations, _, pass_count) in enumerate(loader):

            input_dim = xy.shape[0]
            n_glimpses = xy.shape[1]
            batch_results = pd.DataFrame()
            hidden = rnn.initHidden(input_dim)
            hidden = hidden.to(device)

            for _ in range(recurrent_iterations):
                if config.model_type == 'hyper':
                    pred_num, map, hidden = rnn(input, hidden)
                else:
                    for t in range(n_glimpses):
                        pred_num, map, _, hidden = rnn(xy[:, t, :], shape[:, t, :, :], hidden)

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

        accuracy = 100. * (n_correct/len(loader.dataset))
        epoch_loss /= len(loader)
        num_epoch_loss /= len(loader)
        map_epoch_loss /= len(loader)
        return epoch_loss, num_epoch_loss, accuracy, map_epoch_loss, test_results

    @torch.no_grad()
    def test(loader, epoch):
        rnn.eval()
        n_correct = 0
        epoch_loss = 0
        num_epoch_loss = 0
        count_map_epoch_loss = 0
        # count_map_epoch_loss = 0
        shape_epoch_loss = 0
        nclasses = rnn.output_size
        confusion_matrix = np.zeros((3, nclasses-config.min_num, nclasses-config.min_num))
        test_results = pd.DataFrame()
        # for i, (input, target, locations, shape_label, pass_count) in enumerate(loader):
        for i, (input, target, num_dist, all_loc, shape_label, pass_count) in enumerate(loader):
            n_glimpses = input.shape[1]
            if nonsymbolic:
                xy, shape = input
                input_dim = xy.shape[0]
            else:
                input_dim = input.shape[0]
            batch_results = pd.DataFrame()
            if 'cnn' not in config.model_type and 'feedforward' not in config.model_type or 'pretrained' in config.model_type:
                hidden = rnn.initHidden(input_dim)
                if config.model_type != 'hyper':
                    hidden = hidden.to(device)
            # data = toy.generate_dataset(BATCH_SIZE)
            # # data.loc[0]
            # xy = torch.tensor(data['xy']).float()
            # shape = torch.tensor(data['shape']).float()
            # target = torch.tensor(data['numerosity']).long()
            if ('cnn' in config.model_type or 'feedforward' in config.model_type) and 'pretrained' not in config.model_type:
                pred_num, map = rnn(input)
            else:
                for _ in range(recurrent_iterations):
                    if config.model_type == 'hyper':
                        pred_num, map, hidden = rnn(input, hidden)
                    else:
                        for t in range(n_glimpses):
                            if nonsymbolic:
                                pred_num, map, hidden = rnn(xy[:, t, :], shape[:, t, :, :], hidden)
                            elif config.model_type == 'recurrent_control':
                                pred_num, map, hidden = rnn(input, hidden)
                            else:
                                pred_num, pred_shape, map, hidden, _, _ = rnn(input[:, t, :], hidden)
                                if learn_shape:
                                    shape_loss_mse = criterion_mse(pred_shape, shape_label[:, t, :])*10
                                    shape_loss_ce = criterion(pred_shape, shape_label[:, t, :])
                                    shape_loss = shape_loss_mse + shape_loss_ce
                                    shape_epoch_loss += shape_loss.item()
                                else:
                                    shape_epoch_loss += -1

            if cross_entropy:
                num_loss = criterion_noreduce(pred_num, target)
                pred = pred_num.argmax(dim=1, keepdim=True)
            else:
                num_loss = criterion_mse_noreduce(torch.squeeze(pred_num), target)
                pred = torch.round(pred_num)

            def get_map_loss():
                all_map_loss = criterion_bce_full_noreduce(map, all_loc)
                map_loss = all_map_loss.mean(axis=1)
                map_loss_to_add = map_loss.sum().item()
                return map_loss, map_loss_to_add

            if config.use_loss == 'num':
                loss = num_loss
                # map_loss = criterion_bce_noreduce(map, locations)
                map_loss_to_add = -1
                # full_map_loss_to_add = -1
                # count_map_loss_to_add = -1
                # map_loss = -1
                map_loss_to_add = -1
            elif config.use_loss == 'map':
                # Average over map locations, sum over instances
                # map_loss = criterion_bce_noreduce(map, locations)
                # map_loss_to_add = map_loss.mean(axis=1).sum()
                # map_loss, map_loss_to_add = get_map_loss()
                # loss = map_loss.mean(axis=1)
                # map_loss, map_loss_to_add = get_map_loss()
                map_loss, map_loss_to_add = get_map_loss()
                loss = map_loss

            elif config.use_loss == 'both':
                # map_loss = criterion_bce_noreduce(map, locations)
                # # map_loss_reduce = criterion_bce(map, locations)
                # map_loss_to_add = map_loss.mean(axis=1).sum()
                # map_loss, map_loss_to_add = get_map_loss()
                map_loss, map_loss_to_add = get_map_loss()
                # loss = num_loss + map_loss.mean(axis=1)
                loss = num_loss + map_loss
            elif config.use_loss == 'map_then_both':
                # map_loss = criterion_bce_noreduce(map, locations)

                # map_loss, map_loss_to_add = get_map_loss()
                map_loss, map_loss_to_add = get_map_loss()

                if ep < 150:
                    loss = map_loss
                else:
                    loss = num_loss + map_loss
            correct = pred.eq(target.view_as(pred))
            batch_results['pass count'] = pass_count.detach().cpu().numpy()
            batch_results['correct'] = correct.cpu().numpy()
            batch_results['predicted'] = pred.detach().cpu().numpy()
            batch_results['true'] = target.detach().cpu().numpy()
            batch_results['loss'] = loss.detach().cpu().numpy()
            try:
                # Somehow before it was like just the first column was going
                # into batch_results, so just map loss for the first out of
                # nine location, instead of the average over all locations.
                # batch_results['map loss'] = map_loss.mean(axis=1).detach().cpu().numpy()
                # batch_results['map loss'] = map_loss.mean(axis=1).detach().cpu().numpy()
                batch_results['full map loss'] = map_loss.detach().cpu().numpy()
                # batch_results['count map loss'] = count_map_loss.detach().cpu().numpy()
            except:
                # batch_results['map loss'] = np.ones(loss.shape) * -1
                batch_results['full map loss'] = np.ones(loss.shape) * -1
                batch_results['count map loss'] = np.ones(loss.shape) * -1
            batch_results['num loss'] = num_loss.detach().cpu().numpy()
            batch_results['shape loss'] = shape_loss.detach().cpu().numpy() if learn_shape else -1
            batch_results['epoch'] = epoch
            test_results = pd.concat((test_results, batch_results))

            n_correct += pred.eq(target.view_as(pred)).sum().item()
            epoch_loss += loss.mean().item()
            num_epoch_loss += num_loss.mean().item()
            # if not isinstance(map_loss_to_add, int):
            #     map_loss_to_add = map_loss_to_add.item()
            count_map_epoch_loss += map_loss_to_add
            # class-specific analysis and confusion matrix
            # c = (pred.squeeze() == target)
            for dist in [0, 1, 2]:
                ind = num_dist == dist
                target_subset = target[ind]
                pred_subset = pred[ind]
                for j in range(target_subset.shape[0]):
                    label = target_subset[j]
                    confusion_matrix[dist, label-config.min_num, pred_subset[j]-config.min_num] += 1
        # These two lines should be the same
        # map_epoch_loss / len(loader.dataset)
        # test_results['map loss'].mean()

        accuracy = 100. * (n_correct/len(loader.dataset))
        epoch_loss /= len(loader)
        num_epoch_loss /= len(loader)
        # map_epoch_loss /= len(loader)
        if config.use_loss == 'num':
            # map_epoch_loss /= len(loader)
            count_map_epoch_loss /= len(loader)
        else:
            count_map_epoch_loss /= len(loader.dataset)
        map_epoch_loss = (count_map_epoch_loss, -1)
        shape_epoch_loss /= len(loader) * n_glimpses

        return (epoch_loss, num_epoch_loss, accuracy, shape_epoch_loss,
                map_epoch_loss, test_results, confusion_matrix)

    @torch.no_grad()
    def test2map(loader, epoch):
        rnn.eval()
        n_correct = 0
        epoch_loss = 0
        count_num_epoch_loss = 0
        dist_num_epoch_loss = 0
        all_num_epoch_loss = 0
        full_map_epoch_loss = 0
        count_map_epoch_loss = 0
        dist_map_epoch_loss = 0
        shape_epoch_loss = 0
        nclasses = rnn.output_size
        confusion_matrix = np.zeros((nclasses-config.min_num, nclasses-config.min_num))
        test_results = pd.DataFrame()
        # for i, (input, target, locations, shape_label, pass_count) in enumerate(loader):
        for i, (input, count_num, dist_num, count_loc, dist_loc, shape_label, pass_count) in enumerate(loader):
            n_glimpses = input.shape[1]
            if nonsymbolic:
                xy, shape = input
                input_dim = xy.shape[0]
            else:
                input_dim = input.shape[0]
            batch_results = pd.DataFrame()
            if 'cnn' not in config.model_type:
                hidden = rnn.initHidden(input_dim)
                if config.model_type != 'hyper':
                    hidden = hidden.to(device)
            # data = toy.generate_dataset(BATCH_SIZE)
            # # data.loc[0]
            # xy = torch.tensor(data['xy']).float()
            # shape = torch.tensor(data['shape']).float()
            # target = torch.tensor(data['numerosity']).long()
            if 'cnn' in config.model_type:
                pred_num, map = rnn(input)
            else:
                for _ in range(recurrent_iterations):
                    if config.model_type == 'hyper':
                        pred_num, map, hidden = rnn(input, hidden)
                    else:
                        for t in range(n_glimpses):
                            if nonsymbolic:
                                pred_num, map, hidden = rnn(xy[:, t, :], shape[:, t, :, :], hidden)
                            elif config.model_type == 'recurrent_control':
                                pred_num, map, hidden = rnn(input, hidden)
                            else:
                                pred_num, pred_shape, map, hidden = rnn(input[:, t, :], hidden)
                                shape_loss = criterion_mse(pred_shape, shape_label[:, t, :])*10
                                shape_epoch_loss += shape_loss.item()
            if type(pred_num) is tuple:
                (pred_count_num, pred_dist_num, pred_all_num) = pred_num
            else:
                pred_num_d = None
                pred_all_num = None
            if type(map) is tuple:
                # (all_map, count_map) = map
                (pred_count_map, pred_dist_map, pred_all_map) = map
            else:
                pred_count_map = map
                pred_dist_map = None
                pred_all_map = None
            if cross_entropy:
                count_num_loss = criterion(pred_count_num, count_num)
                dist_num_loss = criterion(pred_dist_num, dist_num)
                all_num_loss = criterion(pred_all_num, torch.add(count_num, dist_num))
                num_loss = count_num_loss + dist_num_loss + all_num_loss
                pred_count = pred_count_num.argmax(dim=1, keepdim=True)
                pred_dist = pred_dist_num.argmax(dim=1, keepdim=True)
                pred_all = pred_all_num.argmax(dim=1, keepdim=True)
                num_loss = count_num_loss + dist_num_loss + all_num_loss
            else:
                num_loss = criterion_mse(torch.squeeze(pred_num), target)
                pred_count = torch.round(pred_num)

            def get_map_loss():
                count_map_loss = criterion_bce_count_noreduce(pred_count_map, count_loc)
                count_map_loss = count_map_loss.mean(axis=1)
                # map_loss_to_add = count_map_loss.mean(axis=1).sum()
                map_loss = count_map_loss
                if pred_dist_map is not None:
                    dist_map_loss = criterion_bce_full_noreduce(pred_dist_map, dist_loc)
                    dist_map_loss = dist_map_loss.mean(axis=1)
                    map_loss += dist_map_loss
                if pred_all_map is not None:
                    all_loc = torch.add(count_loc, dist_loc)
                    all_map_loss = criterion_bce_full_noreduce(pred_all_map, all_loc)
                    all_map_loss = all_map_loss.mean(axis=1)
                    map_loss += all_map_loss
                return count_map_loss, dist_map_loss, all_map_loss, map_loss

            if config.use_loss == 'num':
                loss = num_loss
                # map_loss = criterion_bce_noreduce(map, locations)
                map_loss_to_add = -1
                full_map_loss_to_add = -1
                count_map_loss_to_add = -1
            elif config.use_loss == 'map':
                # Average over map locations, sum over instances
                # map_loss = criterion_bce_noreduce(map, locations)
                # map_loss_to_add = map_loss.mean(axis=1).sum()
                # map_loss, map_loss_to_add = get_map_loss()
                # loss = map_loss.mean(axis=1)
                # map_loss, map_loss_to_add = get_map_loss()
                count_map_loss, dist_map_loss, all_map_loss, map_loss = get_map_loss()
                count_map_loss_to_add = count_map_loss.sum().item()
                dist_map_loss_to_add = dist_map_loss.sum().item()
                full_map_loss_to_add = all_map_loss.sum().item()
                loss = map_loss

            elif config.use_loss == 'both':
                # map_loss = criterion_bce_noreduce(map, locations)
                # # map_loss_reduce = criterion_bce(map, locations)
                # map_loss_to_add = map_loss.mean(axis=1).sum()
                # map_loss, map_loss_to_add = get_map_loss()
                count_map_loss, dist_map_loss, all_map_loss, map_loss = get_map_loss()
                count_map_loss_to_add = count_map_loss.sum().item()
                dist_map_loss_to_add = dist_map_loss.sum().item()
                full_map_loss_to_add = all_map_loss.sum().item()
                # loss = num_loss + map_loss.mean(axis=1)
                loss = num_loss + map_loss
            elif config.use_loss == 'map_then_both':
                # map_loss = criterion_bce_noreduce(map, locations)

                # map_loss, map_loss_to_add = get_map_loss()
                count_map_loss, dist_map_loss, all_map_loss, map_loss = get_map_loss()
                count_map_loss_to_add = count_map_loss.sum().item()
                dist_map_loss_to_add = dist_map_loss.sum().item()
                full_map_loss_to_add = all_map_loss.sum().item()

                if ep < 150:
                    loss = map_loss
                else:
                    loss = num_loss + map_loss
            # correct = pred.eq(target.view_as(pred))
            correct_count = pred_count.eq(count_num.view_as(pred_count))
            correct_dist = pred_dist.eq(dist_num.view_as(pred_dist))
            all_num = torch.add(count_num, dist_num)
            correct_all = pred_all.eq(all_num.view_as(pred_all))
            batch_results['pass count'] = pass_count.detach().cpu().numpy()
            # batch_results['correct'] = correct.cpu().numpy()
            batch_results['correct_count'] = correct_count.cpu().numpy()
            batch_results['correct_dist'] = correct_dist.cpu().numpy()
            batch_results['correct_all'] = correct_all.cpu().numpy()
            # batch_results['predicted'] = pred.detach().cpu().numpy()
            batch_results['predicted_count'] = pred_count.detach().cpu().numpy()
            batch_results['predicted_dist'] = pred_dist.detach().cpu().numpy()
            batch_results['predicted_all'] = pred_all.detach().cpu().numpy()
            batch_results['true_count'] = count_num.detach().cpu().numpy()
            batch_results['true_dist'] = dist_num.detach().cpu().numpy()
            batch_results['true_all'] = all_num.detach().cpu().numpy()
            batch_results['loss'] = loss.detach().cpu().numpy()
            try:
                # Somehow before it was like just the first column was going
                # into batch_results, so just map loss for the first out of
                # nine location, instead of the average over all locations.
                # batch_results['map loss'] = map_loss.mean(axis=1).detach().cpu().numpy()
                # batch_results['map loss'] = map_loss.mean(axis=1).detach().cpu().numpy()
                batch_results['full map loss'] = all_map_loss.detach().cpu().numpy()
                batch_results['count map loss'] = count_map_loss.detach().cpu().numpy()
                batch_results['dist map loss'] = dist_map_loss.detach().cpu().numpy()
            except:
                # batch_results['map loss'] = np.ones(loss.shape) * -1
                batch_results['full map loss'] = np.ones(loss.shape) * -1
                batch_results['count map loss'] = np.ones(loss.shape) * -1
                batch_results['dist map loss'] = np.ones(loss.shape) * -1
            batch_results['count num loss'] = count_num_loss.detach().cpu().numpy()
            batch_results['dist num loss'] = dist_num_loss.detach().cpu().numpy()
            batch_results['all num loss'] = all_num_loss.detach().cpu().numpy()
            batch_results['shape loss'] = shape_loss.detach().cpu().numpy()
            batch_results['epoch'] = epoch
            test_results = pd.concat((test_results, batch_results))

            epoch_loss += loss.mean().item()
            # num_epoch_loss += num_loss.mean().item()

            count_num_epoch_loss += count_num_loss.mean().item()
            dist_num_epoch_loss += dist_num_loss.mean().item()
            all_num_epoch_loss += all_num_loss.mean().item()
            # if not isinstance(map_loss_to_add, int):
                # map_loss_to_add = map_loss_to_add.item()
            full_map_epoch_loss += full_map_loss_to_add
            count_map_epoch_loss += count_map_loss_to_add
            dist_map_epoch_loss += dist_map_loss_to_add
            # class-specific analysis and confusion matrix
            # c = (pred.squeeze() == target)
            for j in range(count_num.shape[0]):
                label = count_num[j]
                confusion_matrix[label-config.min_num, pred_count[j]-config.min_num] += 1
        # These two lines should be the same
        # map_epoch_loss / len(loader.dataset)
        # test_results['map loss'].mean()

        # accuracy = 100. * (n_correct/len(loader.dataset))
        accuracy_count = 100. * (correct_count/len(loader.dataset))
        accuracy_dist = 100. * (correct_dist/len(loader.dataset))
        accuracy_all = 100. * (correct_all/len(loader.dataset))
        accuracy = (accuracy_count, accuracy_dist, accuracy_all)
        epoch_loss /= len(loader)
        count_num_epoch_loss /= len(loader)
        dist_num_epoch_loss /= len(loader)
        all_num_epoch_loss /= len(loader)
        num_epoch_loss = (count_num_epoch_loss, dist_num_epoch_loss, all_num_epoch_loss)
        count_map_epoch_loss /= len(loader)
        dist_map_epoch_loss /= len(loader)
        full_map_epoch_loss /= len(loader)
        map_epoch_loss = (count_map_epoch_loss, dist_map_epoch_loss, full_map_epoch_loss)

        shape_epoch_loss /= len(loader) * n_glimpses

        return (epoch_loss, num_epoch_loss, accuracy, shape_epoch_loss,
                map_epoch_loss, test_results, confusion_matrix)

    @torch.no_grad()
    def test_glimpse(loader, epoch):
        rnn.eval()
        n_correct = 0
        epoch_loss = 0
        num_epoch_loss = 0
        count_map_epoch_loss = 0
        # count_map_epoch_loss = 0
        shape_epoch_loss = 0
        nclasses = rnn.output_size
        confusion_matrix = np.zeros((3, nclasses-config.min_num, nclasses-config.min_num))
        test_results = pd.DataFrame()
        # for i, (input, target, locations, shape_label, pass_count) in enumerate(loader):
        for i, (image, saliency, target, num_dist, all_loc, shape_label, pass_count) in enumerate(loader):

            batch_results = pd.DataFrame()
            if 'cnn' not in config.model_type and 'feedforward' not in config.model_type:
                hidden = rnn.initHidden(input_dim)
                if config.model_type != 'hyper':
                    hidden = hidden.to(device)
            # data = toy.generate_dataset(BATCH_SIZE)
            # # data.loc[0]
            # xy = torch.tensor(data['xy']).float()
            # shape = torch.tensor(data['shape']).float()
            # target = torch.tensor(data['numerosity']).long()
            if 'cnn' in config.model_type or 'feedforward' in config.model_type:
                pred_num, map = rnn(input)
            else:
                for _ in range(recurrent_iterations):
                    if config.model_type == 'hyper':
                        pred_num, map, hidden = rnn(input, hidden)
                    else:
                        for t in range(n_glimpses):
                            if nonsymbolic:
                                pred_num, map, hidden = rnn(xy[:, t, :], shape[:, t, :, :], hidden)
                            elif config.model_type == 'recurrent_control':
                                pred_num, map, hidden = rnn(input, hidden)
                            else:
                                pred_num, pred_shape, map, hidden = rnn(image, saliency, hidden)
                                if learn_shape:
                                    shape_loss_mse = criterion_mse(pred_shape, shape_label[:, t, :])*10
                                    shape_loss_ce = criterion(pred_shape, shape_label[:, t, :])
                                    shape_loss = shape_loss_mse + shape_loss_ce
                                    shape_epoch_loss += shape_loss.item()
                                else:
                                    shape_epoch_loss += -1

            if cross_entropy:
                num_loss = criterion_noreduce(pred_num, target)
                pred = pred_num.argmax(dim=1, keepdim=True)
            else:
                num_loss = criterion_mse_noreduce(torch.squeeze(pred_num), target)
                pred = torch.round(pred_num)

            def get_map_loss():
                all_map_loss = criterion_bce_full_noreduce(map, all_loc)
                map_loss = all_map_loss.mean(axis=1)
                map_loss_to_add = map_loss.sum().item()
                return map_loss, map_loss_to_add

            if config.use_loss == 'num':
                loss = num_loss
                # map_loss = criterion_bce_noreduce(map, locations)
                map_loss_to_add = -1
                # full_map_loss_to_add = -1
                # count_map_loss_to_add = -1
                # map_loss = -1
                map_loss_to_add = -1
            elif config.use_loss == 'map':
                # Average over map locations, sum over instances
                # map_loss = criterion_bce_noreduce(map, locations)
                # map_loss_to_add = map_loss.mean(axis=1).sum()
                # map_loss, map_loss_to_add = get_map_loss()
                # loss = map_loss.mean(axis=1)
                # map_loss, map_loss_to_add = get_map_loss()
                map_loss, map_loss_to_add = get_map_loss()
                loss = map_loss

            elif config.use_loss == 'both':
                # map_loss = criterion_bce_noreduce(map, locations)
                # # map_loss_reduce = criterion_bce(map, locations)
                # map_loss_to_add = map_loss.mean(axis=1).sum()
                # map_loss, map_loss_to_add = get_map_loss()
                map_loss, map_loss_to_add = get_map_loss()
                # loss = num_loss + map_loss.mean(axis=1)
                loss = num_loss + map_loss
            elif config.use_loss == 'map_then_both':
                # map_loss = criterion_bce_noreduce(map, locations)

                # map_loss, map_loss_to_add = get_map_loss()
                map_loss, map_loss_to_add = get_map_loss()

                if ep < 150:
                    loss = map_loss
                else:
                    loss = num_loss + map_loss
            correct = pred.eq(target.view_as(pred))
            batch_results['pass count'] = pass_count.detach().cpu().numpy()
            batch_results['correct'] = correct.cpu().numpy()
            batch_results['predicted'] = pred.detach().cpu().numpy()
            batch_results['true'] = target.detach().cpu().numpy()
            batch_results['loss'] = loss.detach().cpu().numpy()
            try:
                # Somehow before it was like just the first column was going
                # into batch_results, so just map loss for the first out of
                # nine location, instead of the average over all locations.
                # batch_results['map loss'] = map_loss.mean(axis=1).detach().cpu().numpy()
                # batch_results['map loss'] = map_loss.mean(axis=1).detach().cpu().numpy()
                batch_results['full map loss'] = map_loss.detach().cpu().numpy()
                # batch_results['count map loss'] = count_map_loss.detach().cpu().numpy()
            except:
                # batch_results['map loss'] = np.ones(loss.shape) * -1
                batch_results['full map loss'] = np.ones(loss.shape) * -1
                batch_results['count map loss'] = np.ones(loss.shape) * -1
            batch_results['num loss'] = num_loss.detach().cpu().numpy()
            batch_results['shape loss'] = shape_loss.detach().cpu().numpy() if learn_shape else -1
            batch_results['epoch'] = epoch
            test_results = pd.concat((test_results, batch_results))

            n_correct += pred.eq(target.view_as(pred)).sum().item()
            epoch_loss += loss.mean().item()
            num_epoch_loss += num_loss.mean().item()
            # if not isinstance(map_loss_to_add, int):
            #     map_loss_to_add = map_loss_to_add.item()
            count_map_epoch_loss += map_loss_to_add
            # class-specific analysis and confusion matrix
            # c = (pred.squeeze() == target)
            for dist in [0, 1, 2]:
                ind = num_dist == dist
                target_subset = target[ind]
                pred_subset = pred[ind]
                for j in range(target_subset.shape[0]):
                    label = target_subset[j]
                    confusion_matrix[dist, label-config.min_num, pred_subset[j]-config.min_num] += 1
        # These two lines should be the same
        # map_epoch_loss / len(loader.dataset)
        # test_results['map loss'].mean()

        accuracy = 100. * (n_correct/len(loader.dataset))
        epoch_loss /= len(loader)
        num_epoch_loss /= len(loader)
        # map_epoch_loss /= len(loader)
        if config.use_loss == 'num':
            # map_epoch_loss /= len(loader)
            count_map_epoch_loss /= len(loader)
        else:
            count_map_epoch_loss /= len(loader.dataset)
        map_epoch_loss = (count_map_epoch_loss, -1)
        shape_epoch_loss /= len(loader) * n_glimpses

        return (epoch_loss, num_epoch_loss, accuracy, shape_epoch_loss,
                map_epoch_loss, test_results, confusion_matrix)
    if config.save_act:
        print('Saving untrained activations...')
        save_activations(rnn, test_loaders, base_name + '_init', config)
    test_results = pd.DataFrame()
    for ep in range(n_epochs):
        if ep == 50 and config.save_act:
            print('Saving midway activations...')
            save_activations(rnn, test_loaders, base_name + '_midway', config)
        epoch_timer = Timer()
        if nonsymbolic:
            train_f = train_nosymbol
            test_f = test_nosymbol
            perf_f = plot_performance
        elif '2map' in config.model_type:
            train_f = train2map
            test_f = test2map
            perf_f = plot_performance_2map
        elif 'glimpsing' in config.model_type:
            train_f = train_glimpse
            test_f = test_glimpse
            perf_f = plot_performance
        else:
            train_f = train
            test_f = test
            perf_f = plot_performance

        # Train
        ep_tr_loss, ep_tr_num_loss, tr_accuracy, ep_tr_sh_loss, ep_tr_map_loss = train_f(train_loader, ep)
        if type(ep_tr_num_loss) is tuple:
            (ep_tr_count_num_loss, ep_tr_dist_num_loss, ep_tr_all_num_loss) = ep_tr_num_loss
            train_count_num_loss[ep] = ep_tr_count_num_loss
            train_dist_num_loss[ep] = ep_tr_dist_num_loss
            train_all_num_loss[ep] = ep_tr_all_num_loss
            (tr_acc_count, tr_acc_dist, tr_acc_all) = tr_accuracy
            train_acc_count[ep] = tr_acc_count
            train_acc_dist[ep] = tr_acc_dist
            train_acc_all[ep] = tr_acc_all
            (ep_tr_count_map_loss, ep_tr_dist_map_loss, ep_tr_all_map_loss) = ep_tr_map_loss
            train_count_map_loss[ep] = ep_tr_count_map_loss
            train_dist_map_loss[ep] = ep_tr_dist_map_loss
            train_full_map_loss[ep] = ep_tr_all_map_loss
        else:
            train_count_num_loss[ep] = ep_tr_num_loss
            train_acc_count[ep] = tr_accuracy
            train_count_map_loss[ep], train_full_map_loss[ep] = ep_tr_map_loss

        # optimized loss
        train_loss[ep] = ep_tr_loss
        train_sh_loss[ep] = ep_tr_sh_loss
        # train_map_loss[ep] = ep_tr_map_loss
        confs = [None for _ in test_loaders]
        # Test
        shape_lum = product(config.test_shapes, config.lum_sets)
        for ts, (test_loader, (test_shapes, lums)) in enumerate(zip(test_loaders, shape_lum)):
            epoch_te_loss, epoch_te_num_loss, te_accuracy, epoch_te_sh_loss, epoch_te_map_loss, epoch_df, conf = test_f(test_loader, ep)

            epoch_df['train shapes'] = str(config.train_shapes)
            epoch_df['test shapes'] = str(test_shapes)
            epoch_df['test lums'] = str(lums)
            epoch_df['repetition'] = config.rep
            test_results = pd.concat((test_results, epoch_df), ignore_index=True)


            if type(epoch_te_num_loss) is tuple:
                (ep_te_count_num_loss, ep_te_dist_num_loss, ep_te_all_num_loss)= epoch_te_num_loss
                test_count_num_loss[ts][ep] = ep_te_count_num_loss
                test_dist_num_loss[ts][ep] = ep_te_dist_num_loss
                test_all_num_loss[ts][ep] = ep_te_all_num_loss
                (ep_te_count_map_loss, ep_te_dist_map_loss, ep_te_all_map_loss)= epoch_te_map_loss
                test_count_map_loss[ts][ep] = ep_te_count_map_loss
                test_dist_map_loss[ts][ep] = ep_te_dist_map_loss
                test_full_map_loss[ts][ep] = ep_te_all_map_loss

            else:
                test_count_num_loss[ts][ep] = epoch_te_num_loss
                test_acc_count[ts][ep] = te_accuracy
                test_count_map_loss[ts][ep], test_full_map_loss[ts][ep] = epoch_te_map_loss

            # test_map_loss[ts][ep] = epoch_te_map_loss
            test_loss[ts][ep] = epoch_te_loss
            test_sh_loss[ts][ep] = epoch_te_sh_loss
            confs[ts] = conf
            # base_name_test = base_name + f'_test-shapes-{test_shapes}_lums-{lums}'
            base_name_test = base_name

        if not ep % 50 or ep == n_epochs - 1 or ep==1:
            train_num_losses = (train_count_num_loss, train_dist_num_loss, train_all_num_loss)
            train_map_losses = (train_count_map_loss, train_dist_map_loss, train_full_map_loss)
            train_accs = (train_acc_count, train_acc_dist, train_acc_all)
            train_losses = (train_num_losses, train_map_losses, train_sh_loss)
            perf_f(test_results, train_losses, train_accs, confs, ep, config)
        epoch_timer.stop_timer()
        if isinstance(test_loss, list):
            print(f'Epoch {ep}. LR={optimizer.param_groups[0]["lr"]:.4}')
            print(f'Train (Count/Dist/All) Num Loss={train_count_num_loss[ep]:.4}/{train_dist_num_loss[ep]:.4}/{train_all_num_loss[ep]:.4} \t Accuracy={train_acc_count[ep]:.3}%/{train_acc_dist[ep]:.3}%/{train_acc_all[ep]:.3}')
             # Shape loss: {train_sh_loss[ep]:.4}')
            print(f'Train (Count/Dist/All) Map Loss={train_count_map_loss[ep]:.4}/{train_dist_map_loss[ep]:.4}/{train_full_map_loss[ep]:.4}')
            print(f'Test (Count/Dist/All) Num Loss={test_count_num_loss[-1][ep]:.4}/{test_dist_num_loss[-1][ep]:.4}/{test_all_num_loss[-1][ep]:.4} \t Accuracy={test_acc_count[-1][ep]:.3}%/{test_acc_dist[-1][ep]:.3}%/{test_acc_all[-1][ep]:.3}')
            print(f'Test (Count/Dist/All) Map Loss={test_count_map_loss[-1][ep]:.4}/{test_dist_map_loss[-1][ep]:.4}/{test_full_map_loss[-1][ep]:.4}')
        # else:
        #     print(f'Epoch {ep}. LR={optimizer.param_groups[0]["lr"]:.4} \t (Train/Test) Num Loss={train_num_loss[ep]:.4}/{test_num_loss[ep]:.4}/ \t Accuracy={train_acc[ep]:.3}%/{test_acc[ep]:.3}% \t Shape loss: {train_sh_loss[ep]:.5} \t Map loss: {train_map_loss[ep]:.5}')
    
    # Save network activations
    if config.save_act:
        print('Saving activations...')
        save_activations(rnn, test_loaders, base_name + '_trained', config)

    train_num_losses = (train_count_num_loss, train_dist_num_loss, train_all_num_loss)
    train_map_losses = (train_count_map_loss, train_dist_map_loss, train_full_map_loss)
    train_losses = (train_num_losses, train_map_losses, train_sh_loss)
    train_accs = (train_acc_count, train_acc_dist, train_acc_all)
    test_num_losses = (test_count_num_loss, test_dist_num_loss, test_all_num_loss)
    test_map_losses = (test_count_map_loss, test_dist_map_loss, test_full_map_loss)
    test_losses = (test_num_losses, test_map_losses, test_sh_loss)
    test_accs = (test_acc_count, test_acc_dist, test_acc_all)

    # res_tr  = [train_loss, train_acc, train_num_loss, train_sh_loss, train_full_map_loss, train_count_map_loss]
    # res_te = [test_loss, test_acc, test_num_loss, test_sh_loss, test_full_map_loss, test_count_map_loss, confs, test_results]
    res_tr = [train_losses, train_accs]
    res_te = [test_losses, test_accs,  confs, test_results]
    results_list = res_tr + res_te
    return rnn, results_list

@torch.no_grad()
def save_activations(model, test_loaders, basename, config):
    model.eval()
    shape_lum = product(config.test_shapes, config.lum_sets)
    n_glimpses = config.n_glimpses
    test_names = ['validation', 'new-luminances', 'new-shapes', 'new_both']
    # 
    
    # for ts, (test_loader, (test_shapes, lums)) in enumerate(zip(test_loaders, shape_lum)):
    # only save new-both test set for now
    ts = 3
    test_loader = test_loaders[-1]
    start = 0
    test_size = len(test_loader.dataset)
    hidden_act = np.zeros((test_size, n_glimpses, config.h_size))
    premap_act = np.zeros((test_size, n_glimpses, config.h_size))
    penult_act = np.zeros((test_size, n_glimpses, config.grid**2))
    numerosity = np.zeros((test_size,))
    dist_num = np.zeros((test_size,))
    predicted_num = np.zeros((test_size,))
    correct = np.zeros((test_size,))
    for i, (input, target, num_dist, all_loc, shape_label, pass_count) in enumerate(test_loader):
        batch_size = input.shape[0]
        hidden = model.initHidden(batch_size).to(device)
        numerosity[start: start + batch_size] = target.cpu().detach().numpy()
        dist_num[start: start + batch_size] = num_dist.cpu().detach().numpy()
        for t in range(n_glimpses):
            pred_num, _, _, hidden, premap, penult = model(input[:, t, :], hidden)
            hidden_act[start: start + batch_size, t] = hidden.cpu().detach().numpy()
            premap_act[start: start + batch_size, t] = premap.cpu().detach().numpy()
            penult_act[start: start + batch_size, t] = penult.cpu().detach().numpy()
        pred = pred_num.argmax(dim=1, keepdim=True)
        predicted_num[start: start + batch_size] = pred.cpu().detach().numpy().squeeze()
        correct[start: start + batch_size] = pred.eq(target.view_as(pred)).cpu().detach().numpy().squeeze()

        start += batch_size
    # Save to file
    to_save = {'numerosity':numerosity, 'num_distractor':dist_num, 
                'act_hidden':hidden_act, 'act_premap':premap_act, 
                'act_penult':penult_act, 'predicted_num':predicted_num, 'correct':correct}
    savename = f'activations/{basename}_test-{test_names[ts]}'
    # MATLAB
    savemat(savename + '.mat', to_save)
    # Compressed numpy
    np.savez(savename, numerosity=numerosity, num_distractor=dist_num, 
                act_hidden=hidden_act, act_premap=premap_act, 
                act_penult=penult_act)



        



def plot_performance(test_results, train_losses, train_acc, confs, ep, config):
    ticks = list(range(config.max_num - config.min_num + 1))
    ticklabels = [str(tick + config.min_num) for tick in ticks]
    base_name = config.base_name
    test_results['accuracy'] = test_results['correct'].astype(int)*100
    # data = test_results[test_results['test shapes'] == str(test_shapes) and test_results['test lums'] == str(lums)]
    data = test_results
    # max_pass = max(data['pass count'].max(), 6)
    # data = data[data['pass count'] < max_pass]
    make_loss_plot(data, train_losses, ep, config)
    # sns.countplot(data=test_results[test_results['correct']==True], x='epoch', hue='pass count')
    # plt.savefig(f'figures/toy/test_correct_{base_name}.png', dpi=300)
    # plt.close()

    (train_acc_count, _, _) = train_acc
    # accuracy = data.groupby(['epoch', 'pass count']).mean()
    accuracy = data.groupby(['epoch', 'test shapes', 'test lums', 'pass count']).mean(numeric_only=True)

    plt.plot(train_acc_count[:ep + 1], ':', color='green', label='training accuracy')
    sns.lineplot(data=accuracy, x='epoch', hue='test shapes',
                 style='test lums', y='accuracy', alpha=0.7)
    plt.legend()
    plt.grid()
    title = f'{config.model_type} trainon-{config.train_on} train_shapes-{config.train_shapes}'
    plt.title(title)
    plt.ylim([0, 102])
    plt.ylabel('Accuracy on number task')
    plt.savefig(f'figures/toy/letters/accuracy_{base_name}.png', dpi=300)
    plt.close()

    # by integration score plot
    # accuracy = accuracy[]
    # plt.plot(train_acc_count[:ep + 1], ':', color='green', label='training accuracy')
    # sns.lineplot(data=accuracy, x='epoch', hue='pass count',
    #              y='accuracy', alpha=0.7)
    # plt.legend()
    # plt.grid()
    # plt.title(title)
    # plt.ylim([0, 102])
    # plt.ylabel('Accuracy on number task')
    # plt.savefig(f'figures/toy/letters/accuracy_byintegration_{base_name}.png', dpi=300)
    # plt.close()

    # acc_on_difficult = accuracy.loc[ep, 5.0]['accuracy']
    # print(f'Testset {ts}, Accuracy on level 5 difficulty: {acc_on_difficult}')

    fig, axs = plt.subplots(3, 4, figsize=(19, 16))
    
    maxes = [mat.max() for mat in confs]
    vmax = max(maxes)
    # axs = axs.flatten()
    for dist in [0, 1, 2]:
        shape_lum = product(config.test_shapes, config.lum_sets)
        for i, (shape, lum) in enumerate(shape_lum):
    # for i, (ax, (shape, lum)) in enumerate(zip(axs, shape_lum)):
            axs[dist, i].matshow(confs[i][dist, :, :], cmap='Greys', vmin=0, vmax=vmax)
            axs[dist, i].set_aspect('equal', adjustable='box')
            axs[dist, i].set_title(f'dist={dist} shapes={shape} lums={lum}')
            axs[dist, i].set_xticks(ticks, ticklabels)
            axs[dist, i].set_xlabel('Predicted Class')
            axs[dist, i].set_ylabel('True Class')
            axs[dist, i].set_yticks(ticks, ticklabels)
        # ax2 = ax.twinx()
        # ax2.set_yticks(ticks, np.sum(confs[i], axis=1))
    fig.tight_layout()
    plt.savefig(f'figures/toy/letters/confusion_{base_name}.png', dpi=300)
    plt.close()


def make_loss_plot(data, train_losses, ep, config):
    # train_num_loss, train_full_map_loss, _, train_sh_loss = train_losses
    (train_num_losses, train_map_losses, train_sh_loss) = train_losses
    (train_num_loss, _, _) = train_num_losses
    (train_count_map_loss, _, _) = train_map_losses
    ## PLOT LOSS FOR BOTH OBJECTIVES
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=[9,9], sharex=True)
    # sns.lineplot(data=data, x='epoch', y='num loss', hue='pass count', ax=ax1)
    ax1.plot(train_num_loss[:ep + 1], ':', color='green', label='training loss')
    sns.lineplot(data=data, x='epoch', y='num loss', hue='test shapes',
                 style='test lums', ax=ax1, legend=False, alpha=0.7)

    # ax1.legend(title='Integration Difficulty')
    ax1.set_ylabel('Number Loss')
    mt = config.model_type #+ '-nosymbol' if nonsymbolic else config.model_type
    # title = f'{mt} trainon-{config.train_on} train_shapes-{config.train_shapes} \n test_shapes-{test_shapes} useloss-{config.use_loss} lums-{lums}'
    title = f'{mt} trainon-{config.train_on} train_shapes-{config.train_shapes}'
    ax1.set_title(title)
    ylim = ax1.get_ylim()
    ax1.set_ylim([-0.05, ylim[1]])
    ax1.grid()
    # plt.savefig(f'figures/toy/test_num-loss_{base_name_test}.png', dpi=300)
    # plt.close()

    # sns.lineplot(data=data, x='epoch', y='map loss', hue='pass count', ax=ax2, estimator='mean')
    ax2.plot(train_count_map_loss[:ep + 1], ':', color='green', label='training loss')
    sns.lineplot(data=data, x='epoch', y='full map loss', hue='test shapes',
                 style='test lums', ax=ax2, estimator='mean', legend=False, alpha=0.7)
    ax2.set_ylabel('Count Map Loss')
    # plt.title(title)
    ylim = ax2.get_ylim()
    ax2.set_ylim([-0.05, ylim[1]])
    ax2.grid()

    ax3.plot(train_sh_loss[:ep + 1], ':', color='green', label='training loss')
    sns.lineplot(data=data, x='epoch', y='shape loss', hue='test shapes',
                 style='test lums', ax=ax3, estimator='mean', alpha=0.7)
    ax3.set_ylabel('Shape Loss')
    # plt.title(title)
    ylim = ax3.get_ylim()
    ax3.set_ylim([-0.05, ylim[1]])
    ax3.grid()
    fig.tight_layout()
    # ax2.legend(title='Integration Difficulty')
    ax3.legend()
    plt.savefig(f'figures/toy/letters/loss_{config.base_name}.png', dpi=300)
    plt.close()

def plot_performance_2map(test_results, train_losses, train_accs, confs, ep, config):
    (train_num_losses, train_map_losses, train_sh_loss) = train_losses
    (train_acc_count, train_acc_dist, train_acc_all) = train_accs
    ticks = list(range(config.max_num - config.min_num + 1))
    ticklabels = [str(tick + config.min_num) for tick in ticks]
    base_name = config.base_name
    test_results['accuracy_count'] = test_results['correct_count'].astype(int)*100
    test_results['accuracy_dist'] = test_results['correct_dist'].astype(int)*100
    test_results['accuracy_all'] = test_results['correct_all'].astype(int)*100
    # data = test_results[test_results['test shapes'] == str(test_shapes) and test_results['test lums'] == str(lums)]
    data = test_results
    # max_pass = max(data['pass count'].max(), 6)
    # data = data[data['pass count'] < max_pass]
    make_loss_plot_2map(data, train_losses, ep, config)
    # sns.countplot(data=test_results[test_results['correct']==True], x='epoch', hue='pass count')
    # plt.savefig(f'figures/toy/test_correct_{base_name}.png', dpi=300)
    # plt.close()

    # accuracy = data.groupby(['epoch', 'pass count']).mean()

    accuracy = data.groupby(['epoch', 'test shapes', 'test lums', 'pass count']).mean()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
    # count
    ax1.plot(train_acc_count[:ep + 1], ':', color='green', label='training accuracy')
    sns.lineplot(data=accuracy, x='epoch', hue='test shapes',
                 style='test lums', y='accuracy_count', alpha=0.7, ax=ax1)
    ax1.legend()
    ax1.grid()
    title = f'{config.model_type} trainon-{config.train_on} train_shapes-{config.train_shapes}'
    ax1.set_title(title)
    ax1.set_ylim([0, 102])
    ax1.set_ylabel('Accuracy on countA task')
    # distractor
    ax2.plot(train_acc_dist[:ep + 1], ':', color='green', label='training accuracy')
    sns.lineplot(data=accuracy, x='epoch', hue='test shapes',
                 style='test lums', y='accuracy_dist', alpha=0.7, ax=ax2)
    ax2.grid()
    ax2.set_ylim([0, 102])
    ax2.set_ylabel('Accuracy on countB task')
    # all
    ax3.plot(train_acc_all[:ep + 1], ':', color='green', label='training accuracy')
    sns.lineplot(data=accuracy, x='epoch', hue='test shapes',
                 style='test lums', y='accuracy_all', alpha=0.7, ax=ax3)
    ax3.grid()
    ax3.set_ylim([0, 102])
    ax3.set_ylabel('Accuracy on count all task')
    plt.savefig(f'figures/toy/letters/accuracy_{base_name}.png', dpi=300)
    plt.close()

    # plt.plot(train_acc[:ep + 1], ':', color='green', label='training accuracy')
    # sns.lineplot(data=accuracy, x='epoch', hue='pass count',
    #              y='accuracy', alpha=0.7)
    # plt.legend()
    # plt.grid()
    # plt.title(title)
    # plt.ylim([0, 102])
    # plt.ylabel('Accuracy on number task')
    # plt.savefig(f'figures/toy/letters/accuracy_byintegration_{base_name_test}.png', dpi=300)
    # plt.close()

    # acc_on_difficult = accuracy.loc[ep, 5.0]['accuracy']
    # print(f'Testset {ts}, Accuracy on level 5 difficulty: {acc_on_difficult}')

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    shape_lum = product(config.test_shapes, config.lum_sets)
    axs = axs.flatten()
    for i, (ax, (shape, lum)) in enumerate(zip(axs, shape_lum)):
        ax.matshow(confs[i])
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f'test shapes={shape} lums={lum}')
        ax.set_xticks(ticks, ticklabels)
        ax.set_xlabel('Predicted Class')
        ax.set_ylabel('True Class')
        ax.set_yticks(ticks, ticklabels)
        # ax2 = ax.twinx()
        # ax2.set_yticks(ticks, np.sum(confs[i], axis=1))
    fig.tight_layout()
    plt.savefig(f'figures/toy/letters/confusion_{base_name}.png', dpi=300)
    plt.close()


def make_loss_plot_2map(data, train_losses, ep, config):
    (train_num_losses, train_map_losses, train_sh_loss) = train_losses
    (train_count_num_loss, train_dist_num_loss, train_all_num_loss) = train_num_losses
    (train_count_map_loss, train_dist_map_loss, train_all_map_loss) = train_map_losses
    ## PLOT LOSS FOR BOTH OBJECTIVES
    fig, ax = plt.subplots(3, 3, figsize=[12,9], sharex=True)
    # First column: number losses
    # sns.lineplot(data=data, x='epoch', y='num loss', hue='pass count', ax=ax1)
    ax[0,0].plot(train_count_num_loss[:ep + 1], ':', color='green', label='training loss')
    sns.lineplot(data=data, x='epoch', y='count num loss', hue='test shapes',
                 style='test lums', ax=ax[0,0], legend=False, alpha=0.7)

    # ax1.legend(title='Integration Difficulty')
    ax[0,0].set_ylabel('Count Number Loss')
    mt = config.model_type #+ '-nosymbol' if nonsymbolic else config.model_type
    # title = f'{mt} trainon-{config.train_on} train_shapes-{config.train_shapes} \n test_shapes-{test_shapes} useloss-{config.use_loss} lums-{lums}'
    title = f'{mt} trainon-{config.train_on} train_shapes-{config.train_shapes}'
    ax[0,0].set_title(title)
    ylim = ax[0,0].get_ylim()
    ax[0,0].set_ylim([-0.05, ylim[1]])
    ax[0,0].grid()

    ax[1,0].plot(train_dist_num_loss[:ep + 1], ':', color='green', label='training loss')
    sns.lineplot(data=data, x='epoch', y='dist num loss', hue='test shapes',
                 style='test lums', ax=ax[1,0], legend=False, alpha=0.7)
    ax[1,0].set_ylabel('Distractor Number Loss')
    ylim = ax[1,0].get_ylim()
    ax[1,0].set_ylim([-0.05, ylim[1]])
    ax[1,0].grid()

    ax[2,0].plot(train_all_num_loss[:ep + 1], ':', color='green', label='training loss')
    sns.lineplot(data=data, x='epoch', y='all num loss', hue='test shapes',
                 style='test lums', ax=ax[2,0], legend=False, alpha=0.7)
    ax[2,0].set_ylabel('All Number Loss')
    ylim = ax[2,0].get_ylim()
    ax[2,0].set_ylim([-0.05, ylim[1]])
    ax[2,0].grid()

    # Second column: Map loss
    # sns.lineplot(data=data, x='epoch', y='map loss', hue='pass count', ax=ax2, estimator='mean')
    ax[0, 1].plot(train_count_map_loss[:ep + 1], ':', color='green', label='training loss')
    sns.lineplot(data=data, x='epoch', y='count map loss', hue='test shapes',
                 style='test lums', ax=ax[0, 1], estimator='mean', legend=False, alpha=0.7)
    ax[0, 1].set_ylabel('Count Map Loss')
    ylim = ax[0, 1].get_ylim()
    ax[0, 1].set_ylim([-0.05, ylim[1]])
    ax[0, 1].grid()

    ax[1, 1].plot(train_dist_map_loss[:ep + 1], ':', color='green', label='training loss')
    sns.lineplot(data=data, x='epoch', y='dist map loss', hue='test shapes',
                 style='test lums', ax=ax[1, 1], estimator='mean', legend=False, alpha=0.7)
    ax[1, 1].set_ylabel('Distractor Map Loss')
    ylim = ax[1, 1].get_ylim()
    ax[1, 1].set_ylim([-0.05, ylim[1]])
    ax[1, 1].grid()

    ax[2, 1].plot(train_all_map_loss[:ep + 1], ':', color='green', label='training loss')
    sns.lineplot(data=data, x='epoch', y='full map loss', hue='test shapes',
                 style='test lums', ax=ax[2, 1], estimator='mean', legend=False, alpha=0.7)
    ax[2, 1].set_ylabel('All Map Loss')
    ylim = ax[2, 1].get_ylim()
    ax[2, 1].set_ylim([-0.05, ylim[1]])
    ax[2, 1].grid()


    ax[0, 2].plot(train_sh_loss[:ep + 1], ':', color='green', label='training loss')
    sns.lineplot(data=data, x='epoch', y='shape loss', hue='test shapes',
                 style='test lums', ax=ax[0, 2], estimator='mean', alpha=0.7)
    ax[0, 2].set_ylabel('Shape Loss')
    # plt.title(title)
    ylim = ax[0, 2].get_ylim()
    ax[0, 2].set_ylim([-0.05, ylim[1]])
    ax[0, 2].grid()

    ax[1, 2].axis("off")
    ax[2, 2].axis("off")
    fig.tight_layout()
    # ax2.legend(title='Integration Difficulty')
    plt.savefig(f'figures/toy/letters/loss_{config.base_name}.png', dpi=300)
    plt.close()

def save_dataset(fname, noise_level, size, pass_count_range, num_range, shapes_set, same):
    "Depreceated. Datasets should be generated in advance."
    n_shapes = 10
    data = toy.generate_dataset(noise_level, size, pass_count_range, num_range, shapes_set, n_shapes, same)
    data.to_pickle(fname)
    return data

def get_dataset(size, shapes_set, config, lums, solarize):
    """If specified dataset already exists, load it. Otherwise, create it.

    Datasets are always saved with the same time, irrespective of whether the
    dataframe contains the tetris or numeric glimpses or neither.
    """
    noise_level = config.noise_level
    min_pass_count = config.min_pass
    max_pass_count = config.max_pass
    pass_count_range = (min_pass_count, max_pass_count)
    min_num = config.min_num
    max_num = config.max_num
    num_range = (min_num, max_num)
    shape_input = config.shape_input
    same = config.same
    shapes = ''.join([str(i) for i in shapes_set])
    if 'glimpsing' in config.model_type:
        n_glimpses = 'nogl'
    elif config.n_glimpses is not None:
        n_glimpses = f'{config.n_glimpses}_' 
    else:
        n_glimpses = ''
    # solarize = config.solarize

    # fname = f'toysets/toy_dataset_num{min_num}-{max_num}_nl-{noise_level}_diff{min_pass_count}-{max_pass_count}_{shapes_set}_{size}{tet}.pkl'
    # fname_notet = f'toysets/toy_dataset_num{min_num}-{max_num}_nl-{noise_level}_diff{min_pass_count}-{max_pass_count}_{shapes_set}_{size}'
    samee = 'same' if same else ''
    # if config.distract:
    #     if '2channel' in config.shape_input:
    #         challenge = '_distract2ch'
    #     else:
    #         challenge = '_distract'
    # elif config.distract_corner:
    #     challenge = '_distract_corner'
    # elif config.random:
    #     challenge = '_random'
    # else:
    #     challenge = ''
    if config.challenge:
        challenge = '_' + config.challenge
    # distract = '_distract' if config.distract else ''
    solar = 'solarized_' if solarize else ''
    # fname = f'toysets/toy_dataset_num{min_num}-{max_num}_nl-{noise_level}_diff{min_pass_count}-{max_pass_count}_{shapes}{samee}{challenge}_grid{config.grid}_{solar}{n_glimpses}{size}.pkl'
    fname_gw = f'toysets/toy_dataset_num{min_num}-{max_num}_nl-{noise_level}_diff{min_pass_count}-{max_pass_count}_{shapes}{samee}{challenge}_grid{config.grid}_lum{lums}_gw6_{solar}{n_glimpses}{size}.pkl'
    if os.path.exists(fname_gw):
        print(f'Loading saved dataset {fname_gw}')
        data = pd.read_pickle(fname_gw)
    # elif os.path.exists(fname):
    #     print(f'Loading saved dataset {fname}')
    #     data = pd.read_pickle(fname)
    else:
        print(f'{fname_gw} does not exist. Exiting.')
        raise FileNotFoundError
        # print('Generating new dataset')
        # data = save_dataset(fname_gw, noise_level, size, pass_count_range, num_range, shapes_set, same)

    # Add pseudoimage glimpses if needed but not present
    if shape_input == 'tetris' and 'tetris glimpse pixels' not in data.columns:
        data = pseudo.add_tetris(fname_gw)
    elif shape_input == 'char' and 'char glimpse pixels' not in data.columns:
        data = numeric.add_chars(fname_gw)

    return data

def get_loader(dataset, config):
    """Prepare a torch DataLoader for the provided dataset.

    Other input arguments control what the input features should be and what
    datatype the target should be, depending on what loss function will be used.
    The outer argument appends the flattened outer product of the two input
    vectors (xy and shape) to the input tensor. This is hypothesized to help
    enable the network to rely on an integration of the two streams
    """
    train_on = config.train_on
    cross_entropy_loss = config.cross_entropy
    outer = config.outer
    shape_format = config.shape_input
    model_type = config.model_type
    target_type = config.target_type
    # Create shape and or xy tensors
    dataset['shape1'] = dataset['shape']
    shape_array = np.stack(dataset['shape'], axis=0)
    if config.sort:
        shape_arrayA = shape_array[:, :, 0] # Distractor
        shape_array_rest = shape_array[:, :, 1:] # Everything else
        shape_array_rest.sort(axis=-1) # ascending order
        shape_array_rest = shape_array_rest[:, :, ::-1] # descending order
        shape_array =  np.concatenate((np.expand_dims(shape_arrayA, axis=2), shape_array_rest), axis=2)
    shape_label = torch.tensor(shape_array).float().to(device)
    if train_on == 'both' or train_on =='shape':
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
        if shape_format == 'parametric':
            converted = dataset['shape1'].apply(convert)
            shape_input = torch.tensor(converted).float().to(device)
            # shape_label = torch.tensor(dataset['shape']).float().to(device)
            # shape = [torch.tensor(glimpse).float().to(device) for glimpse in converted]
        elif shape_format == 'tetris':
            print('Tetris pixel inputs.')
            # shape_label = torch.tensor(dataset['shape']).float().to(device)
            shape_input = torch.tensor(dataset['tetris glimpse pixels']).float().to(device)
        elif shape_format == 'solarized':
            if 'cnn' in model_type:
                image_array = np.stack(dataset['solarized image'], axis=0)
                shape_input = torch.tensor(image_array).float().to(device)
                shape_input = torch.unsqueeze(shape_input, 1)  # 1 channel
            elif model_type == 'recurrent_control':
                image_array = np.stack(dataset['solarized image'], axis=0)
                nex, w, h = image_array.shape
                image_array = image_array.reshape(nex, -1)
                shape_input = torch.tensor(image_array).float().to(device)
            else:
                glimpse_array = np.stack(dataset['sol glimpse pixels'], axis=0)
                shape_input = torch.tensor(glimpse_array).float().to(device)

        elif 'noise' in shape_format:
            if ('cnn' in model_type and 'pretrained' not in model_type) or 'glimpsing' in model_type:
                image_array = np.stack(dataset['noised image'], axis=0)
                shape_input = torch.tensor(image_array).float().to(device)
                shape_input = torch.unsqueeze(shape_input, 1)  # 1 channel
                if 'glimpsing' in model_type:
                    salience = np.stack(dataset['saliency'], axis=0)
                    # standardize
                    salience /= salience.max()
                    salience = torch.tensor(salience).float().to(device)
            elif model_type == 'recurrent_control' or model_type == 'feedforward':
                image_array = np.stack(dataset['noised image'], axis=0)
                nex, w, h = image_array.shape
                image_array = image_array.reshape(nex, -1)
                shape_input = torch.tensor(image_array).float().to(device)
            else:
                if ('2channel' in shape_format):
                    assert 'dist noi glimpse pixels' in dataset.columns
                    tar_glimpse_array = np.stack(dataset['target noi glimpse pixels'], axis=0)
                    dist_glimpse_array = np.stack(dataset['dist noi glimpse pixels'], axis=0)
                    glimpse_array = np.concatenate((tar_glimpse_array, dist_glimpse_array), axis=-1)
                else:
                    glimpse_array = np.stack(dataset['noi glimpse pixels'], axis=0)
                glimpse_array -= glimpse_array.min()
                glimpse_array /= glimpse_array.max()
                print(f'pixel range: {glimpse_array.min()}-{glimpse_array.max()}')
                # if 'cnn' in config.ventral:
                #     glimpse_array = glimpse_array.reshape(-1, config.n_glimpses, 6, 6)
                # we're going to reshape later instead

                shape_input = torch.tensor(glimpse_array).float().to(device)
            # shape_label = torch.tensor(shape_array).float().to(device)
        elif shape_format == 'pixel_std':
            glimpse_array = np.stack(dataset['char glimpse pixels'], axis=0)
            glimpse_array = np.std(glimpse_array, axis=-1) / 0.4992277987669841  # max std in training
            shape_input = torch.tensor(glimpse_array).unsqueeze(-1).float().to(device)
        elif shape_format == 'pixel_mn+std':
            glimpse_array = np.stack(dataset['char glimpse pixels'], axis=0)
            std = np.std(glimpse_array, axis=-1) / 0.4992277987669841  # max std in training
            mn = np.mean(glimpse_array, axis=-1) #/ max mean is 1 in training set
            glimpse_array = np.stack((mn, std), axis=-1)
            shape_input = torch.tensor(glimpse_array).float().to(device)
        elif shape_format == 'pixel_count':
            glimpse_array = np.stack(dataset['char glimpse pixels'], axis=0)
            n, s, _ = glimpse_array.shape
            all_counts = np.zeros((n, s, 1))
            for i, seq in enumerate(glimpse_array):
                for j, glimpse in enumerate(seq):
                    unique, counts = np.unique(glimpse, return_counts=True)
                    all_counts[i, j, 0] = counts.min()/36
            # unique, counts = np.unique(glimpse_array[0], return_counts=True, axis=0)
            shape_input = torch.tensor(all_counts).float().to(device)
        elif 'symbolic' in shape_format: # symbolic shape input
            # shape_array = np.stack(dataset['shape'], axis=0)
            # shape_input = torch.tensor(dataset['shape']).float().to(device)
            # shape_label = torch.tensor(shape_array).float().to(device)
            if 'ghost' in shape_format:
                # remove distractor shape
                shape_array[:, :, 0] = 0
            shape_input = torch.tensor(shape_array).float().to(device)

    if train_on == 'both' or train_on == 'xy':
        xy_array = np.stack(dataset['xy'], axis=0)
        # xy_array = np.stack(dataset['glimpse coords'], axis=0)
        # norm_xy_array = xy_array/20
        # xy should now already be the original scaled xy between 0 and 1. No need to rescale (since alphabetic)
        norm_xy_array = xy_array * 1.2
        # norm_xy_array = xy_array / 21
        # xy = torch.tensor(dataset['xy']).float().to(device)
        xy = torch.tensor(norm_xy_array).float().to(device)

    # Create merged input (or not)
    if train_on == 'xy':
        input = xy
    elif train_on == 'shape':
        input = shape_input
    elif train_on == 'both' and 'glimpsing' not in model_type:
        if outer:
            assert shape_format != 'parametric'  # not implemented outer with nonsymbolic
            # dataset['shape.t'] = dataset['shape'].apply(lambda x: np.transpose(x))
            # kernel = np.outer(sh, xy) for sh, xy in zip
            def get_outer(xy, shape):
                return [np.outer(x,s).flatten() for x, s in zip(xy, shape)]
            dataset['kernel'] = dataset.apply(lambda x: get_outer(x.xy, x.shape1), axis=1)
            kernel = torch.tensor(dataset['kernel']).float().to(device)
            input = torch.cat((xy, shape_input, kernel), dim=-1)
        else:
            input = torch.cat((xy, shape_input), dim=-1)

    if cross_entropy_loss:
        # better to do this in the training code where we can get all as the sum of the two others
        if target_type == 'all':
            total_num = dataset['locations'].apply(sum)
            target = torch.tensor(total_num).long().to(device)
            count_num = target
        else:
            # target = torch.tensor(dataset['numerosity']).long().to(device)
            try:
                count_num = torch.tensor(dataset['numerosity_count']).long().to(device)
                dist_num = torch.tensor(dataset['numerosity_dist']).long().to(device)
            except:
                count_num = torch.tensor(dataset['numerosity']).long().to(device)
    else:
        count_num = torch.tensor(dataset['numerosity']).float().to(device)
    pass_count = torch.tensor(dataset['pass count']).float().to(device)
    if 'numerosity_dist' in dataset.columns:
        dist_num = torch.tensor(dataset['numerosity_dist']).long().to(device)
    # true_loc = torch.tensor(dataset['locations']).float().to(device)
    if 'locations_count' in dataset.columns:
        count_loc = torch.tensor(dataset['locations_count']).float().to(device)
        dist_loc = torch.tensor(dataset['locations_distract']).float().to(device)
        all_loc = torch.tensor(dataset['locations']).float().to(device)
        true_loc = (all_loc, count_loc)
    elif 'locations_to_count' in dataset.columns:
        count_loc = torch.tensor(dataset['locations_to_count']).float().to(device)
        all_loc = torch.tensor(dataset['locations']).float().to(device)
    else:
        all_loc = torch.tensor(dataset['locations']).float().to(device)
    
    if target_type == 'all':
        count_loc = all_loc

    if shape_format == 'parametric':
        dset = TensorDataset(xy, shape_input, target, all_loc, shape_label, pass_count)
    if '2map' in model_type:
        dset = TensorDataset(input, count_num, dist_num, count_loc, dist_loc, shape_label, pass_count)
    elif 'ghost' in shape_format:
        dset = TensorDataset(input, count_num, count_loc, shape_label, pass_count)
    elif 'glimpsing' in model_type:
        dset = TensorDataset(shape_input, salience, count_num, dist_num, count_loc, shape_label, pass_count)
    else:
        dset = TensorDataset(input, count_num, dist_num, count_loc, shape_label, pass_count)
        # dset = TensorDataset(input, count_num, dist_num, count_loc, shape_label, pass_count)
        # dset = TensorDataset(input, target, all_loc, shape_label, pass_count)
        # dset = TensorDataset(input, target, true_loc, None, shape_label, pass_count)
    loader = DataLoader(dset, batch_size=BATCH_SIZE, shuffle=True)
    return loader

def get_model(model_type, **mod_args):
    hidden_size = mod_args['h_size']
    output_size = mod_args['n_classes'] + 1
    shape_format = mod_args['format']
    train_on = mod_args['train_on']
    grid = mod_args['grid']
    n_shapes = mod_args['n_shapes']
    grid_to_im_shape = {3:[27, 24], 6:[48, 42], 9:[69, 60]}
    height, width = grid_to_im_shape[grid]
    map_size = grid**2
    xy_sz = 2
    if shape_format == 'tetris':
        sh_sz = 4*4
    elif 'symbolic' in shape_format:
        # sh_sz = 64 # 8 * 8
        # sh_sz = 9
        sh_sz = n_shapes#20#25
    elif 'pixel' in shape_format:
        sh_sz = 2 if '+' in shape_format else 1

        sh_sz = (6 * 6) * 2
    else:
        sh_sz = 6 * 6
    if '2channel' in shape_format:
        sh_sz *= 2
    in_sz = xy_sz if train_on=='xy' else sh_sz if train_on =='shape' else sh_sz + xy_sz
    if train_on == 'both' and mod_args['outer']:
        in_sz += xy_sz * sh_sz

    # Initialize the selected model class
    if 'num_as_mapsum' in model_type:
        if shape_format == 'parametric':  #no_symbol:
            model = mod.NumAsMapsum_nosymbol(in_sz, hidden_size, map_size, output_size, **mod_args).to(device)
        elif '2stream' in model_type:
            model = mod.NumAsMapsum2stream(sh_sz, hidden_size, map_size, output_size, **mod_args).to(device)
        else:
            model = mod.NumAsMapsum(in_sz, hidden_size, output_size, **mod_args).to(device)
    elif 'glimpsing' in model_type:
        salience_size = (42, 36)
        model = mod.Glimpsing(salience_size, hidden_size, map_size, output_size, **mod_args).to(device)
    elif 'feedforward' in model_type:
        in_size = height * width
        model = mod.FeedForward(in_size, hidden_size, map_size, output_size, **mod_args).to(device)
    elif 'pretrained_ventral' in model_type:
        model = mod.PretrainedVentral(sh_sz, hidden_size, map_size, output_size, **mod_args).to(device)
    elif 'gated' in model_type:
        if 'map' in model_type:
            if '2' in model_type:
                model = mod.MapGated2RNN(sh_sz, hidden_size, map_size, output_size, **mod_args).to(device)
            else:
                model = mod.MapGatedSymbolicRNN(sh_sz, hidden_size, map_size, output_size, **mod_args).to(device)
        else:
            model = mod.GatedSymbolicRNN(sh_sz, hidden_size, map_size, output_size, **mod_args).to(device)
    elif 'rnn_classifier' in model_type:
        if model_type == 'rnn_classifier_par':
            # Model with two parallel streams at the level of the map. Only one
            # stream is optimized to match the map. The other of the same size
            # is free, only influenced by the num loss.
            mod_args['parallel'] = True
        elif '2stream' in model_type:
            if '2map' in model_type:
                model = mod.RNNClassifier2stream2map(sh_sz, hidden_size, map_size, output_size, **mod_args).to(device)
            else:
                model = mod.RNNClassifier2stream(sh_sz, hidden_size, map_size, output_size, **mod_args).to(device)
        elif shape_format == 'parametric':  #no_symbol:
            model = mod.RNNClassifier_nosymbol(in_sz, hidden_size, output_size, **mod_args).to(device)
        else:
            model = mod.RNNClassifier(in_sz, hidden_size, map_size, output_size, **mod_args).to(device)
    elif model_type == 'recurrent_control':
        # in_sz = 21 * 27  # Size of the images
        in_size = height * width
        model = mod.RNNClassifier(in_sz, hidden_size, output_size, **mod_args).to(device)
    elif model_type == 'rnn_regression':
        model = mod.RNNRegression(in_sz, hidden_size, map_size, output_size, **mod_args).to(device)
    elif model_type == 'mult':
        model = mod.MultiplicativeModel(in_sz, hidden_size, output_size, **mod_args).to(device)
    elif model_type == 'hyper':
        model = mod.HyperModel(in_sz, hidden_size, output_size).to(device)
    elif 'cnn' in model_type:
        # width = 24 #21
        # height = 27
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

def get_config():
    parser = argparse.ArgumentParser(description='PyTorch network settings')
    parser.add_argument('--model_type', type=str, default='num_as_mapsum', help='rnn_classifier rnn_regression num_as_mapsum cnn')
    parser.add_argument('--target_type', type=str, default='multi', help='all or notA ')
    parser.add_argument('--train_on', type=str, default='xy', help='xy, shape, or both')
    parser.add_argument('--noise_level', type=float, default=1.6)
    parser.add_argument('--train_size', type=int, default=100000)
    parser.add_argument('--test_size', type=int, default=5000)
    parser.add_argument('--grid', type=int, default=9)
    parser.add_argument('--n_iters', type=int, default=1, help='how many times the rnn should loop through sequence')
    parser.add_argument('--rotate', action='store_true', default=False)  # not implemented
    parser.add_argument('--small_weights', action='store_true', default=False)  # not implemented
    parser.add_argument('--n_epochs', type=int, default=500)
    parser.add_argument('--use_loss', type=str, default='both', help='num, map or both')
    parser.add_argument('--ventral', type=str, default=None)  
    parser.add_argument('--outer', action='store_true', default=False)
    parser.add_argument('--h_size', type=int, default=25)
    parser.add_argument('--min_pass', type=int, default=0)
    parser.add_argument('--max_pass', type=int, default=6)
    parser.add_argument('--min_num', type=int, default=2)
    parser.add_argument('--max_num', type=int, default=7)
    parser.add_argument('--act', type=str, default=None)
    parser.add_argument('--alt_rnn', action='store_true', default=False)
    # parser.add_argument('--no_symbol', action='store_true', default=False)
    parser.add_argument('--train_shapes', type=list, default=[0, 1, 2, 3, 5, 6, 7, 8], help='Can either be a string of numerals 0123 or letters ABCD.')
    parser.add_argument('--test_shapes', nargs='*', type=list, default=[[0, 1, 2, 3, 5, 6, 7, 8], [4]])
    parser.add_argument('--detach', action='store_true', default=False)
    parser.add_argument('--learn_shape', action='store_true', default=False, help='for the parametric shape rep, whether to additional train to produce symbolic shape labels')
    parser.add_argument('--shape_input', type=str, default='symbolic', help='Which format to use for what pathway (symbolic, parametric, tetris, or char)')
    parser.add_argument('--same', action='store_true', default=False)
    # parser.add_argument('--distract', action='store_true', default=False)
    # parser.add_argument('--distract_corner', action='store_true', default=False)
    # parser.add_argument('--random', action='store_true', default=False)
    parser.add_argument('--challenge', type=str, default='')
    parser.add_argument('--solarize', action='store_true', default=False)
    parser.add_argument('--n_glimpses', type=int, default=None)
    parser.add_argument('--rep', type=int, default=0)
    parser.add_argument('--opt', type=str, default='SGD')
    # parser.add_argument('--tetris', action='store_true', default=False)
    # parser.add_argument('--no_cuda', action='store_true', default=False)
    # parser.add_argument('--preglimpsed', type=str, default=None)
    # parser.add_argument('--use_schedule', action='store_true', default=False)
    parser.add_argument('--dropout', type=float, default=0.0)
    # parser.add_argument('--drop_rnn', type=float, default=0.1)
    # parser.add_argument('--wd', type=float, default=0) # 1e-6
    # parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--save_act', action='store_true', default=False)
    parser.add_argument('--sort', action='store_true', default=False, help='whether ventral stream was trained on sorted shape labels')
    parser.add_argument('--no_pretrain', action='store_true', default=False)
    config = parser.parse_args()
    # Convert string input argument into a list of indices
    if config.train_shapes[0].isnumeric():
        config.shapestr = config.train_shapes.copy()
        config.testshapestr = config.test_shapes.copy()
        config.train_shapes = [int(i) for i in config.train_shapes]
        for j, test_set in enumerate(config.test_shapes):
            config.test_shapes[j] = [int(i) for i in test_set]
    elif config.train_shapes[0].isalpha():
        letter_map = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7,
                      'J':8, 'K':9, 'N':10, 'O':11, 'P':12, 'R':13, 'S':14,
                      'U':15, 'Z':16}
        config.shapestr = config.train_shapes.copy()
        config.testshapestr = config.test_shapes.copy()
        config.train_shapes = [letter_map[i] for i in config.train_shapes]
        for j, test_set in enumerate(config.test_shapes):
            config.test_shapes[j] = [letter_map[i] for i in test_set]
    if 'pretrained_ventral' in config.model_type and config.no_pretrain:
        assert 'finetune' in config.model_type  # otherwise the params in the ventral module will never be trained!
    print(config)
    return config

def main():
    # Process input arguments
    config = get_config()
    model_type = config.model_type
    target_type = config.target_type
    # if model_type == 'num_as_mapsum' or model_type == 'rnn_regression':
    if model_type == 'rnn_regression':
        config.cross_entropy = False
    else:
        config.cross_entropy = True

    train_on = config.train_on
    if model_type == 'recurrent_control' and train_on != 'shape':
        print('Recurrent control requires --train_on=shape (pixel inputs only)')
        exit()
    noise_level = config.noise_level
    train_size = config.train_size
    test_size = config.test_size
    n_iters = config.n_iters
    n_epochs = config.n_epochs
    min_pass = config.min_pass
    max_pass = config.max_pass
    pass_range = (min_pass, max_pass)
    min_num = config.min_num
    max_num = config.max_num
    num_range = (min_num, max_num)
    use_loss = config.use_loss
    drop = config.dropout

    # Prepare base file name for results files
    kernel = '-kernel' if config.outer else ''
    act = '-' + config.act if config.act is not None else ''
    alt_rnn = '2'
    n_glimpses = f'{config.n_glimpses}_' if config.n_glimpses is not None else ''
    detach = '-detach' if config.detach else ''
    pretrain = '-nopretrain' if config.no_pretrain else ''
    model_desc = f'{model_type}{detach}{act}{pretrain}_hsize-{config.h_size}_input-{train_on}{kernel}_{config.shape_input}'
    same = 'same' if config.same else ''
    # if config.distract:
    #     if '2channel' in config.shape_input:
    #         challenge = '_distract2ch'
    #     else:
    #         challenge = '_distract'
    # elif config.distract_corner:
    #     challenge = '_distract_corner'
    # elif config.random:
    #     challenge = '_random'
    # else:
    #     challenge = ''
    # distract = '_distract' if config.distract else ''
    challenge = config.challenge
    solar = 'solarized_' if config.solarize else ''
    shapes = ''.join([str(i) for i in config.shapestr])
    sort = 'sort_' if config.sort else '_notsort'
    data_desc = f'num{min_num}-{max_num}_nl-{noise_level}_diff-{min_pass}-{max_pass}_grid{config.grid}_trainshapes-{shapes}{same}_{challenge}_gw6_{solar}{n_glimpses}{train_size}'
    # train_desc = f'loss-{use_loss}_niters-{n_iters}_{n_epochs}eps'
    withshape = '+shape' if config.learn_shape else ''
    train_desc = f'loss-{use_loss}{withshape}_opt-{config.opt}_drop{drop}_{sort}count-{target_type}_{n_epochs}eps_rep{config.rep}'
    base_name = f'{model_desc}_{data_desc}_{train_desc}'
    if config.small_weights:
        base_name += '_small'
    config.base_name = base_name

    # make sure all results directories exist
    model_dir = 'models/toy/letters'
    results_dir = 'results/toy/letters'
    fig_dir = 'figures/toy/letters'
    dir_list = [model_dir, results_dir, fig_dir]
    for directory in dir_list:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Prepare datasets and torch dataloaders
    #
    # try:
        # config.lum_sets = [[0.1, 0.5, 0.9], [0.2, 0.4, 0.6, 0.8]]
    config.lum_sets = [[0.1, 0.4, 0.7], [0.3, 0.6, 0.9]]
    trainset = get_dataset(train_size, config.shapestr, config, [0.1, 0.4, 0.7], solarize=config.solarize)
    testsets = [get_dataset(test_size, test_shapes, config, lums, solarize=config.solarize) for test_shapes, lums in product(config.testshapestr, config.lum_sets)]
    # except:
    #     config.lum_sets = [[0.0, 0.5, 1.0], [0.1, 0.3, 0.7, 0.9]]
    #     trainset = get_dataset(train_size, config.shapestr, config, [0.0, 0.5, 1.0], solarize=config.solarize)
    #     testsets = [get_dataset(test_size, test_shapes, config, lums, solarize=config.solarize) for test_shapes, lums in product(config.testshapestr, config.lum_sets)]
    # train_loader = get_loader(trainset, config.train_on, config.cross_entropy, config.outer, config.shape_input, model_type, target_type)
    # test_loaders = [get_loader(testset, config.train_on, config.cross_entropy, config.outer, config.shape_input, model_type, target_type) for testset in testsets]
    train_loader = get_loader(trainset, config)
    test_loaders = [get_loader(testset, config) for testset in testsets]
    
    loaders = [train_loader, test_loaders]
    # if config.distract and target_type == 'all':
    if 'distract' in challenge and target_type == 'all':
        max_num += 2
        config.max_num += 2

    # Prepare model and optimizer
    no_symbol = True if config.shape_input == 'parametric' else False
    n_classes = max_num
    n_shapes = 25 # 20 or 25
    ventral = model_dir + '/ventral/' + config.ventral
    finetune = True if 'finetune' in model_type else False
    mod_args = {'h_size': config.h_size, 'act': config.act,
                'small_weights': config.small_weights, 'outer':config.outer,
                'detach': config.detach, 'format':config.shape_input,
                'n_classes':n_classes, 'dropout': drop, 'grid': config.grid,
                'n_shapes':n_shapes, 'ventral':ventral, 'train_on':train_on,
                'finetune': finetune, 'device':device, 'sort':config.sort,
                'no_pretrain': config.no_pretrain}
    model = get_model(model_type, **mod_args)
    # opt = SGD(model.parameters(), lr=start_lr, momentum=mom, weight_decay=wd)
    if config.opt == 'SGD':
        opt = SGD(model.parameters(), lr=start_lr, momentum=mom, weight_decay=wd)
    elif config.opt == 'Adam':
        opt = Adam(model.parameters(), weight_decay=wd, amsgrad=True)
    # scheduler = StepLR(opt, step_size=n_epochs/10, gamma=0.7)
    scheduler = StepLR(opt, step_size=n_epochs/20, gamma=0.7)


    # Train model and save trained model
    model, results = train_model(model, opt, scheduler, loaders, config)
    print('Saving trained model and results files...')
    torch.save(model.state_dict(), f'{model_dir}/toy_model_{base_name}_ep-{n_epochs}.pt')

    # Organize and save results
    train_losses, train_accs, test_losses, test_accs, confs, test_results = results
    (train_num_losses, train_map_losses, train_shape_loss) = train_losses
    (train_acc_count, train_acc_dist, train_acc_all) = train_accs
    (train_count_num_loss, train_dist_num_loss, train_all_num_loss) = train_num_losses
    (train_count_map_loss, train_dist_map_loss, train_full_map_loss) = train_map_losses
    (test_num_losses, test_map_losses, test_shape_loss) = test_losses
    (test_acc_count, test_acc_dist, test_acc_all) = test_accs
    (test_count_num_loss, test_dist_num_loss, test_all_num_loss) = test_num_losses
    (test_count_map_loss, test_dist_map_loss, test_full_map_loss) = test_map_losses

    # train_loss, train_acc, train_num_loss, train_shape_loss, train_full_map_loss, train_count_map_loss, test_loss, test_acc, test_num_loss, test_shape_loss, test_full_map_loss, test_count_map_loss, conf, test_results = results
    test_results.to_pickle(f'{results_dir}/detailed_test_results_{base_name}.pkl')
    df_train = pd.DataFrame()
    df_test_list = [pd.DataFrame() for _ in range(len(testsets))]
    # df_train['loss'] = train_loss
    # df_train['map loss'] = train_map_loss
    df_train['full map loss'] = train_full_map_loss
    df_train['dist map loss'] = train_dist_map_loss
    df_train['count map loss'] = train_count_map_loss
    df_train['count num loss'] = train_count_num_loss
    df_train['dist num loss'] = train_dist_num_loss
    df_train['all num loss'] = train_all_num_loss
    df_train['shape loss'] = train_shape_loss
    df_train['accuracy count'] = train_acc_count
    df_train['accuracy dist'] = train_acc_dist
    df_train['accuracy all'] = train_acc_all
    df_train['epoch'] = np.arange(n_epochs)
    df_train['rnn iterations'] = n_iters
    df_train['dataset'] = 'train'
    for ts, (test_shapes, test_lums) in enumerate(product(config.test_shapes, config.lum_sets)):
        # df_test_list[ts]['loss'] = test_loss[ts]
        df_test_list[ts]['count num loss'] = test_count_num_loss[ts]
        df_test_list[ts]['dist num loss'] = test_dist_num_loss[ts]
        df_test_list[ts]['all num loss'] = test_all_num_loss[ts]
        df_test_list[ts]['shape loss'] = test_shape_loss[ts]
        # df_test_list[ts]['map loss'] = test_map_loss[ts]
        df_test_list[ts]['full map loss'] = test_full_map_loss[ts]
        df_test_list[ts]['count map loss'] = test_count_map_loss[ts]
        df_test_list[ts]['dist map loss'] = test_count_map_loss[ts]
        df_test_list[ts]['accuracy count'] = test_acc_count[ts]
        df_test_list[ts]['accuracy dist'] = test_acc_dist[ts]
        df_test_list[ts]['accuracy all'] = test_acc_all[ts]
        df_test_list[ts]['dataset'] = f'test {test_shapes} {test_lums}'
        df_test_list[ts]['test shapes'] = str(test_shapes)
        df_test_list[ts]['test lums'] = str(test_lums)
        df_test_list[ts]['epoch'] = np.arange(n_epochs)

    np.save(f'{results_dir}/confusion_{base_name}', confs)

    df_test = pd.concat(df_test_list)
    df_test['rnn iterations'] = n_iters
    df = pd.concat((df_train, df_test))
    df.to_pickle(f'{results_dir}/toy_results_{base_name}.pkl')


if __name__ == '__main__':
    main()


# Eventually the plot we want to make is
# sns.countplot(data=correct, x='pass count', hue='rnn iterations')
