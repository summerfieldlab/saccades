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



