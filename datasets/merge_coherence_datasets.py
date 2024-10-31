import os
import numpy as np
import pandas as pd
import xarray as xr
from itertools import product
from matplotlib import pyplot as plt
import argparse


DATA_DIR = '/mnt/jessica/data0/Dropbox/saccades/rnn_tests/datasets/image_sets/'

### Training data

diff_file = 'num1-5_nl-0.74_BCDE_grid6_policy-cheat+jitter_lum[0.1, 0.4, 0.7]_logpolar_12_25000.nc'
same_file = 'num1-5_nl-0.74_BCDEsame_grid6_policy-cheat+jitter_lum[0.1, 0.4, 0.7]_logpolar_12_25000.nc'
diff3_file = 'num1-5_nl-0.74_BCDEdistinct-0.3_grid6_policy-cheat+jitter_lum[0.1, 0.4, 0.7]_logpolar_12_25000.nc'
diff6_file = 'num1-5_nl-0.74_BCDEdistinct-0.6_grid6_policy-cheat+jitter_lum[0.1, 0.4, 0.7]_logpolar_12_25000.nc'

E01 = xr.open_dataset(DATA_DIR + diff_file)
E03 = xr.open_dataset(DATA_DIR + same_file)
E02 = xr.open_dataset(DATA_DIR + diff3_file)
E04 = xr.open_dataset(DATA_DIR + diff6_file)

merged = xr.concat([E01, E02, E03, E04], dim='image')
merged.to_netcdf(DATA_DIR + 'num1-5_nl-0.74_BCDEmixed_grid6_policy-cheat+jitter_lum[0.1, 0.4, 0.7]_logpolar_12_100000.nc')

