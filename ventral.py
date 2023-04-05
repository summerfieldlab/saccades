import os
import argparse
from argparse import Namespace
from itertools import product
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR
from functools import partial
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
# from ray import tune
# from ray.tune import CLIReporter
# from ray.tune.schedulers import ASHAScheduler
# from prettytable import PrettyTable
import ventral_models as mod
from models_old import ConvNet
import toy_model_data as toy
from count_unique import Timer

mom = 0.9
wd = 0.0001
start_lr = 0.01
BATCH_SIZE = 512
TRAIN_SHAPES = [0,  2,  4,  5,  8,  9, 14, 15, 16]

# device = torch.device("cuda")
# device = torch.device("cpu")
criterion_ce = nn.CrossEntropyLoss()
criterion_noreduce = nn.CrossEntropyLoss(reduction='none')
criterion_mse = nn.MSELoss()
criterion_mse_noreduce = nn.MSELoss(reduction='none')
criterion_bce = nn.BCEWithLogitsLoss()


def train_ventral(config, cla, n_epochs):
    # net = Net(config["l1"], config["l2"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net = get_model(config, device)
    net.to(device)

    optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)

    trainset, testset = load_data(cla)

    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs])

    trainloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=0)
    valloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=0)

    for epoch in range(n_epochs):  # loop over the dataset multiple times
        running_loss_ce = 0.
        running_loss_mse = 0.0
        epoch_steps = 0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss_mse = criterion_mse(outputs, labels)
            loss_ce = criterion(outputs, labels)
            loss = 100*loss_mse + loss_ce
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            argmax_labels = torch.argmax(labels, 1)
            total += labels.size(0)
            correct += (predicted == argmax_labels).sum().item()

            # print statistics
            running_loss_ce += loss_ce.item()
            running_loss_mse += loss_mse.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                tr_loss_ce = running_loss_ce / epoch_steps
                tr_loss_mse = running_loss_mse / epoch_steps
                print("[%d, %5d] CEloss: %.3f MSEloss*100: %.3f" % (epoch + 1, i + 1,
                                                                tr_loss_ce,
                                                                tr_loss_mse*100))
                running_loss_ce = 0.0
                running_loss_mse = 0.0
                epoch_steps = 0
        tr_accuracy = correct / total

        # Validation loss
        val_loss_mse = 0.0
        val_loss_ce = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                argmax_labels = torch.argmax(labels, 1)
                total += labels.size(0)
                correct += (predicted == argmax_labels).sum().item()

                loss_mse = criterion_mse(outputs, labels)
                loss_ce = criterion(outputs, labels)
                val_loss_mse += loss_mse.cpu().numpy()
                val_loss_ce += loss_ce.cpu().numpy()
                val_steps += 1
        if np.isnan(val_loss_mse):
            val_loss_mse = 100
        tune.report(val_loss_ce=(val_loss_ce / val_steps),
                    val_loss_mse=(val_loss_mse / val_steps),
                    val_accuracy=correct / total,
                    tr_loss_ce=tr_loss_ce,
                    tr_loss_mse=tr_loss_mse,
                    tr_accuracy=tr_accuracy)
    print("Finished Training")


def load_data(config, device):
    # Prepare datasets and torch dataloaders
    train_size = config.train_size
    test_size = config.test_size
    model_type = config.model_type
    target_type = config.target_type
    sort = config.sort
    # lums = [0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9]
    lums = config.lums
    trainframe = get_dataframe(train_size, config.shapestr, config, lums, solarize=True)
    trainset = get_dataset(trainframe, model_type, target_type, device, sort)
    testframe = get_dataframe(test_size, config.testshapestr[0], config, lums, solarize=True)
    testset = get_dataset(testframe, model_type, target_type, device, sort)
    # try:
    #     config.lum_sets = [[0.1, 0.5, 0.9], [0.2, 0.4, 0.6, 0.8]]
    #     trainframe = get_dataframe(train_size, config.shapestr, config, [0.1, 0.5, 0.9], solarize=config.solarize)
    #     trainset = get_dataset(trainframe, model_type, target_type)
    #     testframes = [get_dataframe(test_size, test_shapes, config, lums, solarize=config.solarize) for test_shapes, lums in product(config.testshapestr, config.lum_sets)]
    #     testsets = [get_dataset(frame, model_type, target_type) for frame in testframes]
    # except:
    #     config.lum_sets = [[0.0, 0.5, 1.0], [0.1, 0.3, 0.7, 0.9]]
    #     trainframe = get_dataframe(train_size, config.shapestr, config, [0.0, 0.5, 1.0], solarize=config.solarize)
    #     trainset = get_dataset(trainframe, model_type, target_type)
    #     testframes = [get_dataframe(test_size, test_shapes, config, lums, solarize=config.solarize) for test_shapes, lums in product(config.testshapestr, config.lum_sets)]
    #     testsets = [get_dataset(frame, model_type, target_type) for frame in testframes]
    return trainset, testset

def te_accuracy(net, cla, device="cpu"):
    trainset, testset = load_data(cla)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=0)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total

def train_model(model, optimizer, scheduler, loaders, config):
    train_loader, test_loader = loaders
    tr_loss_mse = np.zeros((config.n_epochs+1,))
    tr_loss_ce = np.zeros((config.n_epochs+1,))
    tr_loss = np.zeros((config.n_epochs+1,))
    tr_acc = np.zeros((config.n_epochs+1,))
    te_loss_mse = np.zeros((config.n_epochs+1,))
    te_loss_ce = np.zeros((config.n_epochs+1,))
    te_loss = np.zeros((config.n_epochs+1,))
    te_acc = np.zeros((config.n_epochs+1,))

    tr_loss_mse[0], tr_loss_ce[0], tr_loss[0], tr_acc[0] = test(train_loader, model, config.loss, config.sort)
    te_loss_mse[0], te_loss_ce[0], te_loss[0], te_acc[0] = test(test_loader, model, config.loss, config.sort)
    print('Before training')
    print(f'Train: {tr_loss_mse[0]:.4}/{tr_loss_ce[0]:.4}/{tr_loss[0]:.4}/{tr_acc[0]:.3}%')
    print(f'Test: {te_loss_mse[0]:.4}/{te_loss_ce[0]:.4}/{te_loss[0]:.4}/{te_acc[0]:.3}%')
    for ep in range(config.n_epochs):
        tr_res = train_one_epoch(train_loader, model, optimizer, config.loss, config.sort)
        tr_loss_mse[ep+1], tr_loss_ce[ep+1], tr_loss[ep+1], tr_acc[ep+1] = tr_res
        te_res = test(test_loader, model, config.loss, config.sort)
        te_loss_mse[ep+1], te_loss_ce[ep+1], te_loss[ep+1], te_acc[ep+1] = te_res
        scheduler.step()
        print(f'Epoch {ep+1}. LR={optimizer.param_groups[0]["lr"]:.4}')
        print(f'Train: {tr_loss_mse[ep+1]:.4}/{tr_loss_ce[ep+1]:.4}/{tr_loss[ep+1]:.4}/{tr_acc[ep+1]:.4}%')
        print(f'Test: {te_loss_mse[ep+1]:.4}/{te_loss_ce[ep+1]:.4}/{te_loss[ep+1]:.4}/{te_acc[ep+1]:.4}%')
        # if not ep % 10 or ep < 2:
            # plot_performance()
    tr_results = (tr_loss_mse, tr_loss_ce, tr_loss, tr_acc)
    te_results = (te_loss_mse, te_loss_ce, te_loss, te_acc)
    columns = ['loss_mse', 'loss_ce', 'loss', 'accuracy']
    df_train = pd.DataFrame(np.column_stack(tr_results), columns=columns)
    df_train['epoch'] = np.arange(config.n_epochs + 1)
    df_train['dataset'] = 'Train'
    df_test = pd.DataFrame(np.column_stack(te_results), columns=columns)
    df_test['dataset'] = 'Test'
    df_test['epoch'] = np.arange(config.n_epochs + 1)
    results = pd.concat((df_train, df_test))
    return results



def train_one_epoch(train_loader, model, optimizer, which_loss, sort):
    """Iterate through all mini-batches for one epoch of training."""
    model.train()
    optimizer.zero_grad()
    mse_loss = 0
    ce_loss = 0
    tot_loss = 0
    correct = 0
    n = 0
    for (input, target) in train_loader:
        pred, _ = model(input)
        if sort:
            # 0th output for bce loss - detect As
            # 1st and 2nd outputs for MSE loss
            mse = criterion_mse(pred[:, :2], target[:, :2])
            ce = criterion_ce(pred[:, :2], target[:, :2])
        else:
            mse = criterion_mse(pred[:, TRAIN_SHAPES], target[:, TRAIN_SHAPES])
            ce = criterion_ce(pred[:, TRAIN_SHAPES], target[:, TRAIN_SHAPES])
        
        # ce = criterion(pred, target)
        # ce = criterion_bce(pred[:, 0], target[:, 0])
        # bce = criterion_bce(pred[:, TRAIN_SHAPES], target[:, TRAIN_SHAPES])

        # total = (mse*10) + (ce/10)
        # total = (mse*10) + (bce)
        total = mse
        # total = ce
        # loss = mse
        if which_loss=='mse':
            loss = mse
        elif which_loss == 'ce':
            loss = ce

        if sort:
            argmax_labels = torch.argmax(target[:, :2], 1)
            argmax_pred = torch.argmax(pred[:, :2], 1)
        else:
            argmax_labels = torch.argmax(target[:, TRAIN_SHAPES], 1)
            argmax_pred = torch.argmax(pred[:, TRAIN_SHAPES], 1)
        correct += (argmax_pred == argmax_labels).sum().item()
        # elif which_loss == 'bce':
        #     loss = bce
        #     labels = torch.ceil(target[:, TRAIN_SHAPES])
        #     pred = torch.round(torch.sigmoid(pred[:, TRAIN_SHAPES]))
        #     correct += ((pred == labels) * 1.0).mean(dim=1).sum().item()
        # else:
        #     loss = total
        #     A_labels = torch.ceil(target[:, 0])
        #     A_pred = torch.round(torch.sigmoid(pred[:, 0]))
        #     correct += (A_pred == A_labels).sum().item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        optimizer.step()
        n += target.size(0)
        mse_loss += mse.item()
        # bce_loss += bce.item()
        ce_loss += ce.item()
        tot_loss += total.item()
    acc = 100 * (correct/n)
    mse_loss /= len(train_loader)
    ce_loss /= len(train_loader)
    tot_loss /= len(train_loader)
    return mse_loss, ce_loss, tot_loss, acc


@torch.no_grad()
def test(loader, model, which_loss, sort):
    model.eval()
    mse_loss = 0
    ce_loss = 0
    loss = 0
    correct = 0
    n = 0
    
    for (input, target) in loader:
        pred, _ = model(input)
        if sort:
            mse = criterion_mse(pred[:, :2], target[:, :2])
            ce = criterion_ce(pred[:, :2], target[:, :2])
        else:
            mse = criterion_mse(pred[:, TRAIN_SHAPES], target[:, TRAIN_SHAPES])
            ce = criterion_ce(pred[:, TRAIN_SHAPES], target[:, TRAIN_SHAPES])
        # mse = criterion_mse(pred[:, 1:3], target[:, :2])
        # mse = criterion_mse(pred[:, TRAIN_SHAPES], target[:, TRAIN_SHAPES])
        # bce = criterion_bce(pred[:, TRAIN_SHAPES], target[:, TRAIN_SHAPES])
        # ce = criterion_bce(pred[:, 0], target[:, 0])
        # total = (mse*10) + (ce/10)
        # total = (mse*10) + (bce)
        total = mse
        # total = ce
        # if which_loss=='mse':

        if sort:
            argmax_labels = torch.argmax(target[:, :2], 1)
            argmax_pred = torch.argmax(pred[:, :2], 1)
        else:
            argmax_labels = torch.argmax(target[:, TRAIN_SHAPES], 1)
            argmax_pred = torch.argmax(pred[:, TRAIN_SHAPES], 1)
        correct += (argmax_pred == argmax_labels).sum().item()
        # elif which_loss == 'bce':
        #     labels = torch.ceil(target[:, TRAIN_SHAPES])
        #     pred = torch.round(torch.sigmoid(pred[:, TRAIN_SHAPES]))
        #     correct += ((pred == labels) * 1.0).mean(dim=1).sum().item()
        # else:
        #     A_labels = torch.ceil(target[:, 0])
        #     A_pred = torch.round(torch.sigmoid(pred[:, 0]))
        #     correct += (A_pred == A_labels).sum().item()        
        n += target.size(0)
        mse_loss += mse.item()
        # bce_loss += bce.item()
        ce_loss += ce.item()
        loss += total.item()
    print(f'{pred.max()} {pred.min()}')
    acc = 100 * (correct/n)
    mse_loss /= len(loader)
    ce_loss /= len(loader)
    loss /= len(loader)
    return mse_loss, ce_loss, loss, acc



# def plot_performance():
#     pass


def get_dataframe(size, shapes_set, config, lums, solarize):
    """ Load specified dataset

    Args:
        size (int): Number of observations in dataset to be loaded
        shapes_set (list): indicator
        config (dict): _description_
        lums (_type_): _description_
        solarize (_type_): _description_

    Raises:
        FileNotFoundError: _description_

    Returns:
        DataFrame: _description_
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
    # solarize = config.solarize

    # fname = f'toysets/toy_dataset_num{min_num}-{max_num}_nl-{noise_level}_diff{min_pass_count}-{max_pass_count}_{shapes_set}_{size}{tet}.pkl'
    # fname_notet = f'toysets/toy_dataset_num{min_num}-{max_num}_nl-{noise_level}_diff{min_pass_count}-{max_pass_count}_{shapes_set}_{size}'
    samee = 'same' if same else ''
    if config.distract:
        challenge = '_distract'
    elif config.distract_corner:
        challenge = '_distract_corner'
    elif config.random:
        challenge = '_random'
    else:
        challenge = ''
    # distract = '_distract' if config.distract else ''
    solar = 'solarized_' if solarize else ''
    home = '/mnt/jessica/data0/Dropbox/saccades/rnn_tests'
    fname = f'{home}/toysets/toy_dataset_num{min_num}-{max_num}_nl-{noise_level}_diff{min_pass_count}-{max_pass_count}_{shapes}{samee}{challenge}_grid{config.grid}_{solar}{size}.pkl'
    fname_gw = f'{home}/toysets/toy_dataset_num{min_num}-{max_num}_nl-{noise_level}_diff{min_pass_count}-{max_pass_count}_{shapes}{samee}{challenge}_grid{config.grid}_lum{lums}_gw6_{solar}{size}.pkl'
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


    return data


def get_dataset(dataframe, model_type, target_type, device, sort):
    """_summary_

    Args:
        dataset (DataFrame): _description_
        shape_format (str): _description_
        model_type (str): _description_
        target_type (str): _description_

    Returns:
        DataLoader: _description_
    """
    dataframe['shape1'] = dataframe['shape']
    shape_array = np.stack(dataframe['shape'], axis=0)
    if sort:
        shape_arrayA = shape_array[:, :, 0]
        shape_array_rest = shape_array[:, :, 1:]
        shape_array_rest.sort(axis=-1)
        shape_array_rest = shape_array_rest[:, :, ::-1]
        shape_array =  np.concatenate((np.expand_dims(shape_arrayA, axis=2), shape_array_rest), axis=2)
    print(f'label range: {shape_array.min()}-{shape_array.max()}')
    shape_label = torch.tensor(shape_array).float().to(device)
    # if 'cnn' in model_type:
    #     image_array = np.stack(dataframe['noised image'], axis=0)
    #     shape_input = torch.tensor(image_array).float()
    #     shape_input = torch.unsqueeze(shape_input, 1)  # 1 channel
    # else:

    glimpse_array = np.stack(dataframe['noi glimpse pixels'], axis=0)
    glimpse_array -= glimpse_array.min()
    glimpse_array /= glimpse_array.max()
    print(f'pixel range: {glimpse_array.min()}-{glimpse_array.max()}')

    shape_input = torch.tensor(glimpse_array).float().to(device)

    if target_type == 'A_vs_notA':
        pass
    else:
        target = shape_label

    # collapse over glimpses
    shape_input = shape_input.view((-1, 36))
    if 'cnn' in model_type:
        # reshape for conv layers
        shape_input = shape_input.view((-1, 6, 6)).unsqueeze(1)
    shape_label = shape_label.view((-1, 25))

    if not(torch.isfinite(shape_input).all() and torch.isfinite(shape_label).all()):
        print('Found NaNs in the inputs or targets.')
        exit()
    dset = TensorDataset(shape_input, shape_label)
    # loader = DataLoader(dset, batch_size=BATCH_SIZE, shuffle=True)
    return dset

def get_model(config, device):
    input_size = 36
    layer_width = 1024 # config["layer_width"]
    n_layers = 2 # config["n_layers"]
    output_size = 25
    drop = 0.5
    if config.model_type == 'mlp':
        model = mod.MLP(input_size, layer_width, n_layers, output_size, drop)
    elif config.model_type == 'cnn':
        width = 6; height = 6
        penult_size = 4
        model = mod.ConvNet(width, height, penult_size, output_size, dropout=drop)
    model.to(device)
    return model


def get_config():
    parser = argparse.ArgumentParser(description='PyTorch network settings')
    parser.add_argument('--model_type', type=str, default='mlp', help='mlp or cnn')
    parser.add_argument('--target_type', type=str, default='multi', help='all or notA ')

    parser.add_argument('--noise_level', type=float, default=1.6)
    parser.add_argument('--train_size', type=int, default=100000)
    parser.add_argument('--test_size', type=int, default=5000)
    parser.add_argument('--grid', type=int, default=9)

    parser.add_argument('--h_size', type=int, default=25)
    parser.add_argument('--min_pass', type=int, default=0)
    parser.add_argument('--max_pass', type=int, default=6)
    parser.add_argument('--min_num', type=int, default=2)
    parser.add_argument('--max_num', type=int, default=7)
    parser.add_argument('--act', type=str, default=None)
    parser.add_argument('--loss', type=str, default='mse')
    parser.add_argument('--opt', type=str, default='SGD')
    # parser.add_argument('--no_symbol', action='store_true', default=False)
    parser.add_argument('--train_shapes', type=list, default=[0, 1, 2, 3, 5, 6, 7, 8], help='Can either be a string of numerals 0123 or letters ABCD.')
    parser.add_argument('--test_shapes', nargs='*', type=list, default=[[0, 1, 2, 3, 5, 6, 7, 8], [4]])
    parser.add_argument('--lums', nargs='*', type=float, default=[0, 0.5, 1], help='at least two values between 0 and 1')

    parser.add_argument('--shape_input', type=str, default='symbolic', help='Which format to use for what pathway (symbolic, parametric, tetris, or char)')
    parser.add_argument('--same', action='store_true', default=False)
    parser.add_argument('--distract', action='store_true', default=False)
    parser.add_argument('--distract_corner', action='store_true', default=False)
    parser.add_argument('--random', action='store_true', default=False)
    parser.add_argument('--solarize', action='store_true', default=False)
    parser.add_argument('--rep', type=int, default=0)
    parser.add_argument('--n_epochs', type=int, default=10)
    # parser.add_argument('--tetris', action='store_true', default=False)
    parser.add_argument('--no_cuda', action='store_true', default=False)
    # parser.add_argument('--preglimpsed', type=str, default=None)
    # parser.add_argument('--use_schedule', action='store_true', default=False)
    parser.add_argument('--dropout', type=float, default=0.0)
    # parser.add_argument('--drop_rnn', type=float, default=0.1)
    # parser.add_argument('--wd', type=float, default=0) # 1e-6
    # parser.add_argument('--lr', type=float, default=0.01)
    # parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--sort', action='store_true', default=False)
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
    print(config)
    return config


def raymain(num_samples=10, max_num_epochs=10, gpus_per_trial=0.5):
    # Process input arguments
    config = get_config()
    # model_type = config.model_type
    # target_type = config.target_type
    # noise_level = config.noise_level
    # train_size = config.train_size
    # test_size = config.test_size
    # n_iters = config.n_iters
    # n_epochs = config.n_epochs
    # min_pass = config.min_pass
    # max_pass = config.max_pass
    # pass_range = (min_pass, max_pass)
    # min_num = config.min_num
    # max_num = config.max_num
    # num_range = (min_num, max_num)
    # use_loss = config.use_loss
    # drop = config.dropout

    # # Prepare base file name for results files
    # act = '-' + config.act if config.act is not None else ''
    # alt_rnn = '2'
    # detach = '-detach' if config.detach else ''
    # model_desc = f'ventral_{model_type}{alt_rnn}{detach}{act}_hsize-{config.h_size}_{config.shape_input}'
    # same = 'same' if config.same else ''
    # if config.distract:
    #     challenge = '_distract'
    # elif config.distract_corner:
    #     challenge = '_distract_corner'
    # elif config.random:
    #     challenge = '_random'
    # else:
    #     challenge = ''
    # solar = 'solarized_' if config.solarize else ''
    # shapes = ''.join([str(i) for i in config.shapestr])
    # data_desc = f'num{min_num}-{max_num}_nl-{noise_level}_diff-{min_pass}-{max_pass}_grid{config.grid}_trainshapes-{shapes}{same}{challenge}_gw6_{solar}{train_size}'
    # withshape = '+shape' if config.learn_shape else ''
    # train_desc = f'loss-{use_loss}{withshape}_drop{drop}_count-{target_type}_{n_epochs}eps_rep{config.rep}'
    # base_name = f'{model_desc}_{data_desc}_{train_desc}'
    # if config.small_weights:
    #     base_name += '_small'
    # config.base_name = base_name

    # # make sure all results directories exist
    # model_dir = 'models/toy/letters/ventral'
    # results_dir = 'results/toy/letters/ventral'
    # fig_dir = 'figures/toy/letters/ventral'
    # dir_list = [model_dir, results_dir, fig_dir]
    # for directory in dir_list:
    #     if not os.path.exists(directory):
    #         os.makedirs(directory)

    # # Prepare datasets and torch dataloaders
    # try:
    #     config.lum_sets = [[0.1, 0.5, 0.9], [0.2, 0.4, 0.6, 0.8]]
    #     trainset = get_dataset(train_size, config.shapestr, config, [0.1, 0.5, 0.9], solarize=config.solarize)
    #     testsets = [get_dataset(test_size, test_shapes, config, lums, solarize=config.solarize) for test_shapes, lums in product(config.testshapestr, config.lum_sets)]
    # except:
    #     config.lum_sets = [[0.0, 0.5, 1.0], [0.1, 0.3, 0.7, 0.9]]
    #     trainset = get_dataset(train_size, config.shapestr, config, [0.0, 0.5, 1.0], solarize=config.solarize)
    #     testsets = [get_dataset(test_size, test_shapes, config, lums, solarize=config.solarize) for test_shapes, lums in product(config.testshapestr, config.lum_sets)]
    # train_loader = get_loader(trainset, config.shape_input, model_type, target_type)
    # test_loaders = [get_loader(testset, config.shape_input, model_type, target_type) for testset in testsets]
    # loaders = [train_loader, test_loaders]
    # if config.distract and target_type == 'all':
    #     max_num += 2
    #     config.max_num += 2

    # # Prepare model and optimizer
    # n_classes = max_num
    # n_shapes = 25 # 20 or 25
    # mod_args = {'h_size': config.h_size, 'act': config.act,
    #             'small_weights': config.small_weights, 'outer':config.outer,
    #             'detach': config.detach, 'format':config.shape_input,
    #             'n_classes':n_classes, 'dropout': drop, 'grid': config.grid,
    #             'n_shapes':n_shapes}
    # model = get_model(model_type, **mod_args)
    # opt = SGD(model.parameters(), lr=start_lr, momentum=mom, weight_decay=wd)
    # # scheduler = StepLR(opt, step_size=n_epochs/10, gamma=0.7)
    # scheduler = StepLR(opt, step_size=n_epochs/20, gamma=0.7)


    # # Train model and save trained model
    # model, results = train_model(model, opt, scheduler, loaders, config)
    # print('Saving trained model and results files...')
    # torch.save(model.state_dict(), f'{model_dir}/toy_model_{base_name}_ep-{n_epochs}.pt')

    ray_config = {
        "n_layers": tune.choice([0, 1, 2, 3]),
        "layer_width": tune.sample_from(lambda _: 2 ** np.random.randint(5, 11)),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([16, 32, 64, 128, 256])
    }
    # config = Namespace(**vars(config), **ray_config)
    scheduler = ASHAScheduler(
        metric="val_loss_mse",
        mode="min",
        max_t=max_num_epochs,
        grace_period=5,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["tr_loss_ce", "tr_loss_mse", "val_loss_ce", "val_loss_mse", "tr_accuracy", "val_accuracy", "training_iteration"])
    result = tune.run(
        partial(train_ventral, cla=config, n_epochs=max_num_epochs),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=ray_config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    # best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])
    # best_trained_model = get_model(best_trial.config)
    # device = "cpu"
    # if torch.cuda.is_available():
    #     device = "cuda:0"
    #     if gpus_per_trial > 1:
    #         best_trained_model = nn.DataParallel(best_trained_model)
    # best_trained_model.to(device)

    # best_checkpoint_dir = best_trial.checkpoint.value
    # model_state, optimizer_state = torch.load(os.path.join(
    #     best_checkpoint_dir, "checkpoint"))
    # best_trained_model.load_state_dict(model_state)

    # te_acc = te_accuracy(best_trained_model, cla, device)
    # print("Best trial test set accuracy: {}".format(te_acc))

def main():
    config = get_config()
    device = "cpu"
    if torch.cuda.is_available() and not config.no_cuda:
        device = "cuda:0"
    # Prepare base file name for results files
    model_type = config.model_type
    noise_level = config.noise_level
    train_size = config.train_size
    n_epochs = config.n_epochs
    min_pass = config.min_pass
    max_pass = config.max_pass
    min_num = config.min_num
    max_num = config.max_num
    drop = config.dropout
    act = '-' + config.act if config.act is not None else ''

    model_desc = f'ventral_{model_type}{act}_hsize-{config.h_size}_{config.shape_input}'
    same = 'same' if config.same else ''
    if config.distract:
        challenge = '_distract'
    elif config.random:
        challenge = '_random'
    else:
        challenge = ''
    solar = 'solarized_' if config.solarize else ''
    sort = 'sort_' if config.sort else ''
    shapes = ''.join([str(i) for i in config.shapestr])
    data_desc = f'num{min_num}-{max_num}_nl-{noise_level}_diff-{min_pass}-{max_pass}_grid{config.grid}_lum-{config.lums}_trainshapes-{shapes}{same}{challenge}_gw6_{train_size}'
    train_desc = f'loss-{config.loss}_opt-{config.opt}_drop{drop}_{sort}{n_epochs}eps_rep{config.rep}'
    base_name = f'{model_desc}_{data_desc}_{train_desc}'
    config.base_name = base_name

    # make sure all results directories exist
    model_dir = 'models/toy/letters/ventral'
    results_dir = 'results/toy/letters/ventral'
    fig_dir = 'figures/toy/letters/ventral'
    dir_list = [model_dir, results_dir, fig_dir]
    for directory in dir_list:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Prepare datasets
    trainset, testset = load_data(config, device)
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=int(BATCH_SIZE),
        shuffle=True,
        num_workers=0)
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=int(BATCH_SIZE),
        shuffle=True,
        num_workers=0)
    loaders = [trainloader, testloader]

    # Prepare model and optimizer

    model = get_model(config, device)
    if config.opt == 'SGD':
        opt = SGD(model.parameters(), lr=start_lr, momentum=mom, weight_decay=wd)
    elif config.opt == 'Adam':
        opt = Adam(model.parameters(), weight_decay=wd, amsgrad=True)
    scheduler = StepLR(opt, step_size=config.n_epochs/10, gamma=0.7)
    # scheduler = StepLR(opt, step_size=config.n_epochs/20, gamma=0.7)

    # Train model
    results = train_model(model, opt, scheduler, loaders, config)

    print('Saving trained model and results files...')
    print(f'{model_dir}/ventral_{base_name}_ep-{n_epochs}.pt')
    torch.save(model.state_dict(), f'{model_dir}/ventral_{base_name}_ep-{n_epochs}.pt')
    results.to_csv(f'{results_dir}/ventral_{base_name}_ep-{n_epochs}.csv')



if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    # raymain(num_samples=100, max_num_epochs=500, gpus_per_trial=0.5)

    main()