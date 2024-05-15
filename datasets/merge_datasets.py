import os
import numpy as np
import pandas as pd
import xarray as xr
from itertools import product
from matplotlib import pyplot as plt
import argparse

# DATA_DIR = '/mnt/jessica/data0/Dropbox/saccades/rnn_tests/toysets/'
DATA_DIR = '/mnt/jessica/data0/Dropbox/saccades/rnn_tests/datasets/image_sets/'

### Training data
# E01file = 'num1-5_nl-0.9_ESUZsame_distract123_grid6_policy-humanlike_lum[0.1, 0.4, 0.7]_logpolar_12_10000.nc'
# F01file = 'num1-5_nl-0.9_FCKJsame_distract123_grid6_policy-humanlike_lum[0.1, 0.4, 0.7]_logpolar_12_10000.nc'
# E03file = 'num1-5_nl-0.9_ESUZsame_distract123_grid6_policy-humanlike_lum[0.3, 0.6, 0.9]_logpolar_12_10000.nc'
# F03file = 'num1-5_nl-0.9_FCKJsame_distract123_grid6_policy-humanlike_lum[0.3, 0.6, 0.9]_logpolar_12_10000.nc'
E01file = 'num1-5_nl-0.74_BCDEsame_distract012_grid6_policy-cheat+jitter_lum[0.1, 0.4, 0.7]_logpolar_12_10000.nc'
F01file = 'num1-5_nl-0.74_FGHJsame_distract012_grid6_policy-cheat+jitter_lum[0.1, 0.4, 0.7]_logpolar_12_10000.nc'
E03file = 'num1-5_nl-0.74_BCDEsame_distract012_grid6_policy-cheat+jitter_lum[0.3, 0.6, 0.9]_logpolar_12_10000.nc'
F03file = 'num1-5_nl-0.74_FGHJsame_distract012_grid6_policy-cheat+jitter_lum[0.3, 0.6, 0.9]_logpolar_12_10000.nc'
E01 = xr.open_dataset(DATA_DIR + E01file)
E03 = xr.open_dataset(DATA_DIR + E03file)
F01 = xr.open_dataset(DATA_DIR + F01file)
F03 = xr.open_dataset(DATA_DIR + F03file)
merged = xr.concat([E01, E03, F01, F03], dim='image')
merged.to_netcdf(DATA_DIR + 'num1-5_nl-0.74_BCDEFGHJsame_distract012_grid6_policy-cheat+jitter_lum[0.1, 0.4, 0.7, 0.3, 0.6, 0.9]_logpolar_12_40000.nc')


### Testing data
# E01file = 'num1-5_nl-0.9_ESUZsame_distract123_grid6_policy-humanlike_lum[0.1, 0.4, 0.7]_logpolar_12_1000.nc'
# F01file = 'num1-5_nl-0.9_FCKJsame_distract123_grid6_policy-humanlike_lum[0.1, 0.4, 0.7]_logpolar_12_1000.nc'
# E03file = 'num1-5_nl-0.9_ESUZsame_distract123_grid6_policy-humanlike_lum[0.3, 0.6, 0.9]_logpolar_12_1000.nc'
# F03file = 'num1-5_nl-0.9_FCKJsame_distract123_grid6_policy-humanlike_lum[0.3, 0.6, 0.9]_logpolar_12_1000.nc'
E01file = 'num1-5_nl-0.74_BCDEsame_distract012_grid6_policy-cheat+jitter_lum[0.1, 0.4, 0.7]_logpolar_12_1000.nc'
F01file = 'num1-5_nl-0.74_FGHJsame_distract012_grid6_policy-cheat+jitter_lum[0.1, 0.4, 0.7]_logpolar_12_1000.nc'
E03file = 'num1-5_nl-0.74_BCDEsame_distract012_grid6_policy-cheat+jitter_lum[0.3, 0.6, 0.9]_logpolar_12_1000.nc'
F03file = 'num1-5_nl-0.74_FGHJsame_distract012_grid6_policy-cheat+jitter_lum[0.3, 0.6, 0.9]_logpolar_12_1000.nc'
E01 = xr.open_dataset(DATA_DIR + E01file)
E03 = xr.open_dataset(DATA_DIR + E03file)
F01 = xr.open_dataset(DATA_DIR + F01file)
F03 = xr.open_dataset(DATA_DIR + F03file)
merged = xr.concat([E01, E03, F01, F03], dim='image')
merged.to_netcdf(DATA_DIR + 'num1-5_nl-0.74_BCDEFGHJsame_distract012_grid6_policy-cheat+jitter_lum[0.1, 0.4, 0.7, 0.3, 0.6, 0.9]_logpolar_12_4000.nc')


#### Training set
# E01file = 'num1-5_nl-0.9_ESUZsame_distract_grid6_policy-cheat+jitter_lum[0.1, 0.4, 0.7]_gw6_solarized_12_25000.pkl'
# F01file = 'num1-5_nl-0.9_FCKJsame_distract_grid6_policy-cheat+jitter_lum[0.1, 0.4, 0.7]_gw6_solarized_12_25000.pkl'
# E03file = 'num1-5_nl-0.9_ESUZsame_distract_grid6_policy-cheat+jitter_lum[0.3, 0.6, 0.9]_gw6_solarized_12_25000.pkl'
# F03file = 'num1-5_nl-0.9_FCKJsame_distract_grid6_policy-cheat+jitter_lum[0.3, 0.6, 0.9]_gw6_solarized_12_25000.pkl'
# E01file = 'num3-6_nl-0.9_ESUZ_unique_grid6_policy-cheat+jitter_lum[0.1, 0.4, 0.7]_gw6_solarized_12_25000.pkl'
# F01file = 'num3-6_nl-0.9_FCKJ_unique_grid6_policy-cheat+jitter_lum[0.1, 0.4, 0.7]_gw6_solarized_12_25000.pkl'
# E03file = 'num3-6_nl-0.9_ESUZ_unique_grid6_policy-cheat+jitter_lum[0.3, 0.6, 0.9]_gw6_solarized_12_25000.pkl'
# F03file = 'num3-6_nl-0.9_FCKJ_unique_grid6_policy-cheat+jitter_lum[0.3, 0.6, 0.9]_gw6_solarized_12_25000.pkl'

# E01file = 'toy_dataset_num1-5_nl-0.9_diff0-8_ESUZsame_distract_grid6_lum[0.1, 0.4, 0.7]_gw6_solarized_50000.pkl'
# F01file = 'toy_dataset_num1-5_nl-0.9_diff0-8_FCKJsame_distract_grid6_lum[0.1, 0.4, 0.7]_gw6_solarized_25000.pkl'
# E03file = 'toy_dataset_num1-5_nl-0.9_diff0-8_ESUZsame_distract_grid6_lum[0.3, 0.6, 0.9]_gw6_solarized_50000.pkl'
# F03file = 'toy_dataset_num1-5_nl-0.9_diff0-8_FCKJsame_distract_grid6_lum[0.3, 0.6, 0.9]_gw6_solarized_25000.pkl'

# E01file = 'num1-5_nl-0.9_ESUZsame_distract_grid6_policy-cheat+jitter_lum[0.1, 0.4, 0.7]_logpolar_12_25000.pkl'
# F01file = 'num1-5_nl-0.9_FCKJsame_distract_grid6_policy-cheat+jitter_lum[0.1, 0.4, 0.7]_logpolar_12_25000.pkl'
# E03file = 'num1-5_nl-0.9_ESUZsame_distract_grid6_policy-cheat+jitter_lum[0.3, 0.6, 0.9]_logpolar_12_25000.pkl'
# F03file = 'num1-5_nl-0.9_FCKJsame_distract_grid6_policy-cheat+jitter_lum[0.3, 0.6, 0.9]_logpolar_12_25000.pkl'
# print('Loading training data')
# E01 = pd.read_pickle(DATA_DIR + E01file)
# F01 = pd.read_pickle(DATA_DIR + F01file)
# E03 = pd.read_pickle(DATA_DIR + E03file)
# F03 = pd.read_pickle(DATA_DIR + F03file)
# print('Merging data')
# merged = pd.concat((E01, F01, E03, F03)).reset_index()
# # merged = pd.concat((E01, E03)).reset_index()
# # savename = 'num1-5_nl-0.9_ESUZFCKJsame_distract_grid6_lum[0.1, 0.4, 0.7, 0.3, 0.6, 0.9]_gw6_solarized_12_100000.pkl'
# # savename = 'num3-6_nl-0.9_ESUZFCKJ_unique_grid6_lum[0.1, 0.4, 0.7, 0.3, 0.6, 0.9]_gw6_solarized_12_100000.pkl'
# savename = 'num1-5_nl-0.9_ESUZFCKJsame_distract_grid6_policy-cheat+jitter_lum[0.1, 0.4, 0.7, 0.3, 0.6, 0.9]_logpolar_12_100000.pkl'

# print(f'Saving {savename}...')
# merged.to_pickle(DATA_DIR + savename)

# #### Test set
# # E01file = 'num1-5_nl-0.9_ESUZsame_distract_grid6_policy-cheat+jitter_lum[0.1, 0.4, 0.7]_gw6_solarized_12_1251.pkl'
# # F01file = 'num1-5_nl-0.9_FCKJsame_distract_grid6_policy-cheat+jitter_lum[0.1, 0.4, 0.7]_gw6_solarized_12_1251.pkl'
# # E03file = 'num1-5_nl-0.9_ESUZsame_distract_grid6_policy-cheat+jitter_lum[0.3, 0.6, 0.9]_gw6_solarized_12_1251.pkl'
# # F03file = 'num1-5_nl-0.9_FCKJsame_distract_grid6_policy-cheat+jitter_lum[0.3, 0.6, 0.9]_gw6_solarized_12_1251.pkl'
# # E01file = 'num3-6_nl-0.9_ESUZ_unique_grid6_policy-cheat+jitter_lum[0.1, 0.4, 0.7]_gw6_solarized_12_1251.pkl'
# # F01file = 'num3-6_nl-0.9_FCKJ_unique_grid6_policy-cheat+jitter_lum[0.1, 0.4, 0.7]_gw6_solarized_12_1251.pkl'
# # E03file = 'num3-6_nl-0.9_ESUZ_unique_grid6_policy-cheat+jitter_lum[0.3, 0.6, 0.9]_gw6_solarized_12_1251.pkl'
# # F03file = 'num3-6_nl-0.9_FCKJ_unique_grid6_policy-cheat+jitter_lum[0.3, 0.6, 0.9]_gw6_solarized_12_1251.pkl'
# E01file = 'num1-5_nl-0.9_ESUZsame_distract_grid6_policy-cheat+jitter_lum[0.1, 0.4, 0.7]_logpolar_12_1250.pkl'
# F01file = 'num1-5_nl-0.9_FCKJsame_distract_grid6_policy-cheat+jitter_lum[0.1, 0.4, 0.7]_logpolar_12_1250.pkl'
# E03file = 'num1-5_nl-0.9_ESUZsame_distract_grid6_policy-cheat+jitter_lum[0.3, 0.6, 0.9]_logpolar_12_1250.pkl'
# F03file = 'num1-5_nl-0.9_FCKJsame_distract_grid6_policy-cheat+jitter_lum[0.3, 0.6, 0.9]_logpolar_12_1250.pkl'
# print('Loading testing data')
# E01 = pd.read_pickle(DATA_DIR + E01file)
# F01 = pd.read_pickle(DATA_DIR + F01file)
# E03 = pd.read_pickle(DATA_DIR + E03file)
# F03 = pd.read_pickle(DATA_DIR + F03file)
# merged = pd.concat((E01, F01, E03, F03)).reset_index()
# # savename = 'num1-5_nl-0.9_ESUZFCKJsame_distract_grid6_lum[0.1, 0.4, 0.7, 0.3, 0.6, 0.9]_gw6_solarized_12_5004.pkl'
# # savename = 'num3-6_nl-0.9_ESUZFCKJ_unique_grid6_lum[0.1, 0.4, 0.7, 0.3, 0.6, 0.9]_gw6_solarized_12_5004.pkl'
# savename = 'num1-5_nl-0.9_ESUZFCKJsame_distract_grid6_policy-cheat+jitter_lum[0.1, 0.4, 0.7, 0.3, 0.6, 0.9]_logpolar_12_5000.pkl'
# print(f'Saving {savename}...')
# merged.to_pickle(DATA_DIR + savename)
