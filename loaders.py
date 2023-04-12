import os
from itertools import product
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader


def get_dataset(size, shapes_set, config, lums, solarize):
    """If specified dataset already exists, load it.
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
    challenge = '_' + config.challenge if config.challenge != '' else ''
    # distract = '_distract' if config.distract else ''
    solar = 'solarized_' if solarize else ''
    policy = config.policy
    # fname = f'toysets/toy_dataset_num{min_num}-{max_num}_nl-{noise_level}_diff{min_pass_count}-{max_pass_count}_{shapes}{samee}{challenge}_grid{config.grid}_{solar}{n_glimpses}{size}.pkl'
    # fname_gw = f'toysets/toy_dataset_num{min_num}-{max_num}_nl-{noise_level}_diff{min_pass_count}-{max_pass_count}_{shapes}{samee}{challenge}_grid{config.grid}_lum{lums}_gw6_{solar}{n_glimpses}{size}.pkl'
    fname_gw = f'toysets/num{min_num}-{max_num}_nl-{noise_level}_{shapes}{samee}{challenge}_grid{config.grid}_policy-{policy}_lum{lums}_gw6_{solar}{n_glimpses}{size}.pkl'

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


def get_loader(dataset, config, batch_size=None):
    train_on = config.train_on
    cross_entropy_loss = config.cross_entropy
    outer = config.outer
    shape_format = config.shape_input
    model_type = config.model_type
    target_type = config.target_type
    convolutional = True if config.model_type in ['cnn', 'bigcnn'] else False

    ### NUMBER LABEL ###
    if target_type == 'all':
        total_num = dataset['locations'].apply(sum)
        target = torch.tensor(total_num).long().to(config.device)
        count_num = target
        if 'numerosity_dist' in dataset.columns:
            dist_num = torch.tensor(dataset['numerosity_dist']).long().to(config.device)
        else:
            notA = dataset['locations_to_count'].apply(sum)
            dist_num = torch.tensor(count_num - notA).long().to(config.device)
    elif 'unique' in config.challenge:
        count_num = torch.tensor(dataset['num_unique']).long().to(config.device)
        dist_num = torch.zeros_like(count_num).long().to(config.device)
    else:
        if 'numerosity_dist' in dataset.columns:
            count_num = torch.tensor(dataset['numerosity_target']).long().to(config.device)
            dist_num = torch.tensor(dataset['numerosity_dist']).long().to(config.device)
        else:
            count_num = torch.tensor(dataset['numerosity']).long().to(config.device)
            dist_num = torch.zeros_like(count_num).long().to(config.device)

    ### INTEGRATION SCORE ###
    pass_count = torch.tensor(dataset['pass_count']).float().to(config.device)
    
    ### SHAPE LABEL ###
    dataset['shape1'] = dataset['shape']
    shape_array = np.stack(dataset['shape'], axis=0)
    if config.sort:
        shape_arrayA = shape_array[:, :, 0] # Distractor
        shape_array_rest = shape_array[:, :, 1:] # Everything else
        shape_array_rest.sort(axis=-1) # ascending order
        shape_array_rest = shape_array_rest[:, :, ::-1] # descending order
        shape_array =  np.concatenate((np.expand_dims(shape_arrayA, axis=2), shape_array_rest), axis=2)
    shape_label = torch.tensor(shape_array).float().to(config.device)

    ### MAP LABEL ###
    # true_loc = torch.tensor(dataset['locations']).float().to(config.device)
    if 'locations_count' in dataset.columns:
        count_loc = torch.tensor(dataset['locations_count']).float().to(config.device)
        dist_loc = torch.tensor(dataset['locations_distract']).float().to(config.device)
        all_loc = torch.tensor(dataset['locations']).float().to(config.device)
        true_loc = (all_loc, count_loc)
    elif 'locations_to_count' in dataset.columns:
        count_loc = torch.tensor(dataset['locations_to_count']).float().to(config.device)
        all_loc = torch.tensor(dataset['locations']).float().to(config.device)
    else:
        all_loc = torch.tensor(dataset['locations']).float().to(config.device)
        count_loc = all_loc
    
    if target_type == 'all':
        count_loc = all_loc

    ### IMAGE INPUT ###
    if config.whole_image:
        if convolutional:
            image_array = np.stack(dataset['noised_image'], axis=0)
            image_input = torch.tensor(image_array).float().to(config.device)
            image_input = torch.unsqueeze(image_input, 1)  # 1 channel
        else:  # flatten
            image_array = np.stack(dataset['noised_image'], axis=0)
            nex, w, h = image_array.shape
            image_array = image_array.reshape(nex, -1)
            image_input = torch.tensor(image_array).float().to(config.device)

    ### PIXEL/SHAPE INPUT ###
    if train_on == 'both' or train_on =='shape':
        if shape_format == 'noise':
            try:
                glimpse_array = np.stack(dataset['noi_glimpse_pixels'], axis=0)
            except:
                glimpse_array = np.stack(dataset['noi glimpse pixels'], axis=0)
            glimpse_array -= glimpse_array.min()
            glimpse_array /= glimpse_array.max()
            print(f'pixel range: {glimpse_array.min()}-{glimpse_array.max()}')
            shape_input = torch.tensor(glimpse_array).float().to(config.device)
        elif 'symbolic' in shape_format: # symbolic shape input
            if 'ghost' in shape_format:
                # remove distractor shape
                shape_array[:, :, 0] = 0
            shape_input = torch.tensor(shape_array).float().to(config.device)
    
    ### XY LOCATION INPUT ###
    if train_on == 'both' or train_on == 'xy':
        xy_array = np.stack(dataset['xy'], axis=0)
        # xy_array = np.stack(dataset['glimpse coords'], axis=0)
        # norm_xy_array = xy_array/20
        # xy should now already be the original scaled xy between 0 and 1. No need to rescale (since alphabetic)
        norm_xy_array = xy_array * 1.2
        # norm_xy_array = xy_array / 21
        # xy = torch.tensor(dataset['xy']).float().to(config.device)
        xy = torch.tensor(norm_xy_array).float().to(config.device)
    
    # Create merged input (or not)
    if config.whole_image:
        input = image_input
    elif train_on == 'xy':
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
            kernel = torch.tensor(dataset['kernel']).float().to(config.device)
            input = torch.cat((xy, shape_input, kernel), dim=-1)
        else:
            input = torch.cat((xy, shape_input), dim=-1)

    ### PREPARE LOADER ###
    if config.whole_image:    
        dset = TensorDataset(input, count_num, dist_num, count_loc, pass_count)
    else:  
        
        dset = TensorDataset(input, count_num, dist_num, count_loc, shape_label, pass_count)
        # dset = TensorDataset(input, count_num, dist_num, count_loc, shape_label, pass_count)
        # dset = TensorDataset(input, target, all_loc, shape_label, pass_count)
        # dset = TensorDataset(input, target, true_loc, None, shape_label, pass_count)
    bs = config.batch_size if batch_size is None else batch_size
    loader = DataLoader(dset, batch_size=bs, shuffle=True)
    return loader



# def get_loader(dataset, config, batch_size=None):
#     """Prepare a torch DataLoader for the provided dataset.

#     Other input arguments control what the input features should be and what
#     datatype the target should be, depending on what loss function will be used.
#     The outer argument appends the flattened outer product of the two input
#     vectors (xy and shape) to the input tensor. This is hypothesized to help
#     enable the network to rely on an integration of the two streams
#     """
#     train_on = config.train_on
#     cross_entropy_loss = config.cross_entropy
#     outer = config.outer
#     shape_format = config.shape_input
#     model_type = config.model_type
#     target_type = config.target_type
#     # Create shape and or xy tensors
#     dataset['shape1'] = dataset['shape']
#     shape_array = np.stack(dataset['shape'], axis=0)
#     if config.sort:
#         shape_arrayA = shape_array[:, :, 0] # Distractor
#         shape_array_rest = shape_array[:, :, 1:] # Everything else
#         shape_array_rest.sort(axis=-1) # ascending order
#         shape_array_rest = shape_array_rest[:, :, ::-1] # descending order
#         shape_array =  np.concatenate((np.expand_dims(shape_arrayA, axis=2), shape_array_rest), axis=2)
#     shape_label = torch.tensor(shape_array).float().to(config.device)
#     ### PIXEL/SHAPE INPUT ###
#     if train_on == 'both' or train_on =='shape':
#         def convert(symbolic):
#             """return array of 4 lists of nonsymbolic"""
#             # dataset size x n glimpses x n shapes (100 x 4 x 9)
#             # want to convert to 100 x 4 x n_shapes in this glimpse x 3)
#             coords = [(x,y) for (x,y) in product([0.2, 0.5, 0.8], [0.2, 0.5, 0.8])]
#             indexes = np.arange(9)
#             # [word for sentence in text for  word in sentence]
#             # nonsymbolic = [(glimpse[idx], coords[idx][0], coords[idx][1]) for glimpse in symbolic for idx in glimpse.nonzero()[0]]
#             nonsymbolic = [[],[],[],[]]
#             for i, glimpse in enumerate(symbolic):
#                 np.random.shuffle(indexes)
#                 nonsymbolic[i] = [(glimpse[idx], coords[idx][0], coords[idx][1]) for idx in indexes]
#             return nonsymbolic
#         if shape_format == 'parametric':
#             converted = dataset['shape1'].apply(convert)
#             shape_input = torch.tensor(converted).float().to(config.device)
#             # shape_label = torch.tensor(dataset['shape']).float().to(config.device)
#             # shape = [torch.tensor(glimpse).float().to(config.device) for glimpse in converted]
#         elif shape_format == 'tetris':
#             print('Tetris pixel inputs.')
#             # shape_label = torch.tensor(dataset['shape']).float().to(config.device)
#             shape_input = torch.tensor(dataset['tetris glimpse pixels']).float().to(config.device)
#         elif shape_format == 'solarized':
#             if 'cnn' in model_type:
#                 image_array = np.stack(dataset['solarized image'], axis=0)
#                 shape_input = torch.tensor(image_array).float().to(config.device)
#                 shape_input = torch.unsqueeze(shape_input, 1)  # 1 channel
#             elif model_type == 'recurrent_control':
#                 image_array = np.stack(dataset['solarized image'], axis=0)
#                 nex, w, h = image_array.shape
#                 image_array = image_array.reshape(nex, -1)
#                 shape_input = torch.tensor(image_array).float().to(config.device)
#             else:
#                 glimpse_array = np.stack(dataset['sol glimpse pixels'], axis=0)
#                 shape_input = torch.tensor(glimpse_array).float().to(config.device)

#         elif 'noise' in shape_format:
#             if ('cnn' in model_type and 'ventral' not in model_type) or 'whole' in model_type:
#                 image_array = np.stack(dataset['noised_image'], axis=0)
#                 shape_input = torch.tensor(image_array).float().to(config.device)
#                 shape_input = torch.unsqueeze(shape_input, 1)  # 1 channel
#             elif 'glimpsing' in model_type:
#                 image_array = np.stack(dataset['noised_image'], axis=0)
#                 shape_input = torch.tensor(image_array).float().to(config.device)
#                 shape_input = torch.unsqueeze(shape_input, 1)  # 1 channel
#                 salience = np.stack(dataset['saliency'], axis=0)
#                 # standardize
#                 salience /= salience.max()
#                 salience = torch.tensor(salience).float().to(config.device)
#             elif model_type == 'recurrent_control' or model_type == 'feedforward':
#                 image_array = np.stack(dataset['noised_image'], axis=0)
#                 nex, w, h = image_array.shape
#                 image_array = image_array.reshape(nex, -1)
#                 shape_input = torch.tensor(image_array).float().to(config.device)
#             else:
#                 if ('2channel' in shape_format):
#                     assert 'dist noi glimpse pixels' in dataset.columns
#                     tar_glimpse_array = np.stack(dataset['target noi glimpse pixels'], axis=0)
#                     dist_glimpse_array = np.stack(dataset['dist noi glimpse pixels'], axis=0)
#                     glimpse_array = np.concatenate((tar_glimpse_array, dist_glimpse_array), axis=-1)
#                 else:
#                     try:
#                         glimpse_array = np.stack(dataset['noi_glimpse_pixels'], axis=0)
#                     except:
#                         glimpse_array = np.stack(dataset['noi glimpse pixels'], axis=0)
#                 glimpse_array -= glimpse_array.min()
#                 glimpse_array /= glimpse_array.max()
#                 print(f'pixel range: {glimpse_array.min()}-{glimpse_array.max()}')
#                 # if 'cnn' in config.ventral:
#                 #     glimpse_array = glimpse_array.reshape(-1, config.n_glimpses, 6, 6)
#                 # we're going to reshape later instead

#                 shape_input = torch.tensor(glimpse_array).float().to(config.device)
#             # shape_label = torch.tensor(shape_array).float().to(config.device)
#         elif shape_format == 'pixel_std':
#             glimpse_array = np.stack(dataset['char glimpse pixels'], axis=0)
#             glimpse_array = np.std(glimpse_array, axis=-1) / 0.4992277987669841  # max std in training
#             shape_input = torch.tensor(glimpse_array).unsqueeze(-1).float().to(config.device)
#         elif shape_format == 'pixel_mn+std':
#             glimpse_array = np.stack(dataset['char glimpse pixels'], axis=0)
#             std = np.std(glimpse_array, axis=-1) / 0.4992277987669841  # max std in training
#             mn = np.mean(glimpse_array, axis=-1) #/ max mean is 1 in training set
#             glimpse_array = np.stack((mn, std), axis=-1)
#             shape_input = torch.tensor(glimpse_array).float().to(config.device)
#         elif shape_format == 'pixel_count':
#             glimpse_array = np.stack(dataset['char glimpse pixels'], axis=0)
#             n, s, _ = glimpse_array.shape
#             all_counts = np.zeros((n, s, 1))
#             for i, seq in enumerate(glimpse_array):
#                 for j, glimpse in enumerate(seq):
#                     unique, counts = np.unique(glimpse, return_counts=True)
#                     all_counts[i, j, 0] = counts.min()/36
#             # unique, counts = np.unique(glimpse_array[0], return_counts=True, axis=0)
#             shape_input = torch.tensor(all_counts).float().to(config.device)
#         elif 'symbolic' in shape_format: # symbolic shape input
#             # shape_array = np.stack(dataset['shape'], axis=0)
#             # shape_input = torch.tensor(dataset['shape']).float().to(config.device)
#             # shape_label = torch.tensor(shape_array).float().to(config.device)
#             if 'ghost' in shape_format:
#                 # remove distractor shape
#                 shape_array[:, :, 0] = 0
#             shape_input = torch.tensor(shape_array).float().to(config.device)
#     ### XY LOCATION INPUT ###
#     if train_on == 'both' or train_on == 'xy':
#         xy_array = np.stack(dataset['xy'], axis=0)
#         # xy_array = np.stack(dataset['glimpse coords'], axis=0)
#         # norm_xy_array = xy_array/20
#         # xy should now already be the original scaled xy between 0 and 1. No need to rescale (since alphabetic)
#         norm_xy_array = xy_array * 1.2
#         # norm_xy_array = xy_array / 21
#         # xy = torch.tensor(dataset['xy']).float().to(config.device)
#         xy = torch.tensor(norm_xy_array).float().to(config.device)

#     # Create merged input (or not)
#     if train_on == 'xy':
#         input = xy
#     elif train_on == 'shape':
#         input = shape_input
#     elif train_on == 'both' and 'glimpsing' not in model_type:
#         if outer:
#             assert shape_format != 'parametric'  # not implemented outer with nonsymbolic
#             # dataset['shape.t'] = dataset['shape'].apply(lambda x: np.transpose(x))
#             # kernel = np.outer(sh, xy) for sh, xy in zip
#             def get_outer(xy, shape):
#                 return [np.outer(x,s).flatten() for x, s in zip(xy, shape)]
#             dataset['kernel'] = dataset.apply(lambda x: get_outer(x.xy, x.shape1), axis=1)
#             kernel = torch.tensor(dataset['kernel']).float().to(config.device)
#             input = torch.cat((xy, shape_input, kernel), dim=-1)
#         else:
#             input = torch.cat((xy, shape_input), dim=-1)

#     ### NUMBER LABEL ###
#     if cross_entropy_loss:
#         # better to do this in the training code where we can get all as the sum of the two others
#         if target_type == 'all':
#             total_num = dataset['locations'].apply(sum)
#             target = torch.tensor(total_num).long().to(config.device)
#             count_num = target
#             if 'numerosity_dist' in dataset.columns:
#                 dist_num = torch.tensor(dataset['numerosity_dist']).long().to(config.device)
#             else:
#                 notA = dataset['locations_to_count'].apply(sum)
#                 dist_num = torch.tensor(count_num - notA).long().to(config.device)
#         elif 'unique' in config.challenge:
#             count_num = torch.tensor(dataset['num_unique']).long().to(config.device)
#             dist_num = torch.zeros_like(count_num).long().to(config.device)
#         else:
#             # target = torch.tensor(dataset['numerosity']).long().to(config.device)
#             if 'numerosity_dist' in dataset.columns:
#                 count_num = torch.tensor(dataset['numerosity_count']).long().to(config.device)
#                 dist_num = torch.tensor(dataset['numerosity_dist']).long().to(config.device)
#             else:
#                 count_num = torch.tensor(dataset['numerosity']).long().to(config.device)
#                 dist_num = torch.zeros_like(count_num).long().to(config.device)
#     else:
#         count_num = torch.tensor(dataset['numerosity']).float().to(config.device)
#     try:
#         pass_count = torch.tensor(dataset['pass_count']).float().to(config.device)
#     except:
#         pass_count = torch.tensor(dataset['pass count']).float().to(config.device)
    
#     ### MAP LABEL ###
#     # true_loc = torch.tensor(dataset['locations']).float().to(config.device)
#     if 'locations_count' in dataset.columns:
#         count_loc = torch.tensor(dataset['locations_count']).float().to(config.device)
#         dist_loc = torch.tensor(dataset['locations_distract']).float().to(config.device)
#         all_loc = torch.tensor(dataset['locations']).float().to(config.device)
#         true_loc = (all_loc, count_loc)
#     elif 'locations_to_count' in dataset.columns:
#         count_loc = torch.tensor(dataset['locations_to_count']).float().to(config.device)
#         all_loc = torch.tensor(dataset['locations']).float().to(config.device)
#     else:
#         all_loc = torch.tensor(dataset['locations']).float().to(config.device)
#         count_loc = all_loc
    
#     if target_type == 'all':
#         count_loc = all_loc

#     if shape_format == 'parametric':
#         dset = TensorDataset(xy, shape_input, target, all_loc, shape_label, pass_count)
#     if '2map' in model_type:
#         dset = TensorDataset(input, count_num, dist_num, count_loc, dist_loc, shape_label, pass_count)
#     elif 'ghost' in shape_format:
#         dset = TensorDataset(input, count_num, count_loc, shape_label, pass_count)
#     elif 'glimpsing' in model_type:
#         dset = TensorDataset(shape_input, salience, count_num, dist_num, count_loc, shape_label, pass_count)
#     else:
#         dset = TensorDataset(input, count_num, dist_num, count_loc, shape_label, pass_count)
#         # dset = TensorDataset(input, count_num, dist_num, count_loc, shape_label, pass_count)
#         # dset = TensorDataset(input, target, all_loc, shape_label, pass_count)
#         # dset = TensorDataset(input, target, true_loc, None, shape_label, pass_count)
#     bs = config.batch_size if batch_size is None else batch_size
#     loader = DataLoader(dset, batch_size=bs, shuffle=True)
#     return loader

def choose_loader(config):
    train_size = config.train_size
    test_size = config.test_size
    try:
        # config.lum_sets = [[0.1, 0.5, 0.9], [0.2, 0.4, 0.6, 0.8]]
        config.lum_sets = [[0.1, 0.4, 0.7], [0.3, 0.6, 0.9]]
        trainset = get_dataset(train_size, config.shapestr, config, [0.1, 0.4, 0.7], solarize=config.solarize)
        testsets = [get_dataset(test_size, test_shapes, config, lums, solarize=config.solarize) for test_shapes, lums in product(config.testshapestr, config.lum_sets)]
    except:
        config.lum_sets = [[0.0, 0.5, 1.0], [0.1, 0.3, 0.7, 0.9]]
        trainset = get_dataset(train_size, config.shapestr, config, [0.0, 0.5, 1.0], solarize=config.solarize)
        testsets = [get_dataset(test_size, test_shapes, config, lums, solarize=config.solarize) for test_shapes, lums in product(config.testshapestr, config.lum_sets)]
    
    # train_loader = get_loader(trainset, config.train_on, config.cross_entropy, config.outer, config.shape_input, model_type, target_type)
    # test_loaders = [get_loader(testset, config.train_on, config.cross_entropy, config.outer, config.shape_input, model_type, target_type) for testset in testsets]
    train_loader = get_loader(trainset, config)
    # Large batch size for test loader to test quicker
    test_loaders = [get_loader(testset, config, batch_size=2500) for testset in testsets]
    
    loaders = [train_loader, test_loaders]
    return loaders