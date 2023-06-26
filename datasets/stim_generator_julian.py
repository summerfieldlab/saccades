"""Generate stimuli for exytracking experiment conducted by Julian Sandbrink.

Generates (for each participant) 8 stimuli sets, one for each of the 4
conditions, duplicated with only half the stimuli for the practice trials. In
the ignore-distractors task, the number of distractors is now 1,2 or 3, so there
is always at least one distractor. The target numerosity range (for all 
conditions) is now 3-6. (previously the lowest numerosity was 2) These two tasks
are now 'matched on their output rather than their input'. So the images in the
count-all conditions contain no A's. The ignore-distractor images thus contain
more items than the count-all images, but the target numerosities (the range of
correct responses) are matched. The images all have constant contrast.
Random seed is set as a function of participant number. Rerunning this code
for the same participant number should generate exactly the same 8 sets of
images as were used in the experiment.

Random seed = [participant number] * 10 + [seed command line argument]


author: Jessica Thompson
date: 2023-06-13
"""
import os
import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.io import savemat
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial import distance
from skimage.transform import warp_polar
from itertools import product
import json
from math import isclose

import symbolic_model as solver
from letters import get_alphabet
from dataset_generator import DatasetGenerator

class DatasetGeneratorJulian(DatasetGenerator):
    def __init__(self, conf):
        super().__init__(conf)
        
    def add_logpolar_glimpses_pandas(self, data, dirname, conf):
        lums = conf.luminances
        glim_wid = 6
        half_glim = glim_wid//2
        chars = get_alphabet()
        image_size_nobord = [self.pixel_height, self.pixel_width]
        imsize_wbord = [self.pixel_height+glim_wid, self.pixel_width+glim_wid]
        data['glimpse_coords_image'] = None
        data['glimpse_coords_scaled'] = None
        data['luminances'] = None
        data['noised_image'] = None
        data['centre_fixation'] = None
        data['logpolar_pixels'] = None
        data['target_coords_scaled'] = None
        data['distract_coords_scaled'] = None
        for i in range(len(data)):
            filename = f'image_{i}.png'
            if not i % 10:
                print(f'Synthesizing image {i}', end='\r')
            row = data.iloc[i]
            # Coordinates in 1x1 space
            slots = np.where(row.locations)[0]
            object_xy_coords = self.centroid_array[slots]            
            object_pixel_coords = [self.map_scale_pixel[tuple(xy)] for xy in object_xy_coords]
            # Shape indices for the objects
            object_shapes = [row['shape_map'][obj] for obj in slots]

            # Insert the specified shapes into the image at the specified locations
            image = np.zeros(image_size_nobord, dtype=np.float32)
            for shape_idx, (x,y) in zip(object_shapes, object_pixel_coords):
                    image[y:y+self.char_height:, x:x+self.char_width] = chars[shape_idx]
            # Add border
            image_wbord = np.zeros(imsize_wbord, dtype=np.float32)
            image_wbord[half_glim:-half_glim,half_glim:-half_glim] = image
            # data.at[i, 'bw image'] = image_wbord

            # Convert toy glimpse coordinates to pixel coordinates
            # glimpse_coords_1x1 = row.xy_1x1.copy()
            glimpse_coords_1x1 = row.glimpse_coords_1x1.copy()
            glimpse_coords_image = np.multiply(glimpse_coords_1x1, image_size_nobord[::-1]) + half_glim
            glimpse_coords_image = np.round(glimpse_coords_image).astype(int)
            data.at[i, 'glimpse_coords_image'] = glimpse_coords_image
            data.at[i, 'glimpse_coords_scaled'] = np.divide(glimpse_coords_image, imsize_wbord[::-1])      

            # Get target and distractor coordinates in the same scaled frame of 
            # reference as the glimpse coords. Important for analysis of 
            # activations and eye tracking.
            target_slots = np.where(row['locations_count'])[0]
            target_coords_1x1 = self.centroid_array[target_slots]
            # target_coords_image = (target_coords_1x1 * image_size_nobord[::-1]) + half_glim
            # Get the upper left corner and then add to get centroid
            target_coords_image = np.array([self.map_scale_pixel[tuple(xy)]for xy in target_coords_1x1]) + half_glim 
            target_coords_image = target_coords_image + [(self.char_width/2) - 0.5, (self.char_height/2) - 0.5]
            target_coords_scaled = target_coords_image/imsize_wbord[::-1]
            data.at[i, 'target_coords_scaled'] = target_coords_scaled
            
            # Only add distractor locations when distractors exist
            distract_slots = np.where(row['locations_distract'])[0]
            if len(distract_slots) > 0:
                distract_coords_1x1 = self.centroid_array[distract_slots]
                # distract_coords_image = (distract_coords_1x1 * image_size_nobord[::-1]) + half_glim
                distract_coords_image = np.array([self.map_scale_pixel[tuple(xy)]for xy in distract_coords_1x1]) + half_glim 
                distract_coords_image = distract_coords_image + [(self.char_width/2) - 0.5, (self.char_height/2) - 0.5]
                distract_coords_scaled = distract_coords_image/imsize_wbord[::-1]
                data.at[i, 'distract_coords_scaled'] = distract_coords_scaled
            else:
                # If columns are left containing None, savemat will throw an error
                # Put an empty list instead
                data.at[i, 'distract_coords_scaled'] = []
            
            # Solarize and add Gaussian noise
            fg, bg = np.random.choice(lums, size=2, replace=False)
            # ensure all images have a contrast of 0.3
            # apparently 0.9 - 0.6 =  0.30000000000000004
            while not isclose(abs(fg - bg), 0.3):
                fg, bg = np.random.choice(lums, size=2, replace=False)
            data.at[i, 'luminances'] = [fg, bg]
            noised = self.get_solarized_noise(image_wbord, fg, bg)
            # noised = self.get_solarized_noise(image, fg, bg)
            data.at[i, 'noised_image'] = noised

            # Warp noised image to generaye log-polar glimpses
            # Fixed gaze at the centre
            centre = [size//2 for size in imsize_wbord]
            # centre = [size//2 for size in image_size]
            fixation = warp_polar(noised, scaling='log', output_shape=imsize_wbord, center=centre, mode='edge')  # rows, cols
            # fixation = warp_polar(noised, scaling='log', output_shape=image_size, center=centre, mode='edge')  # rows, cols

            data.at[i, 'centre_fixation'] = fixation
            # Glimpses
            lp_glimpses = [warp_polar(noised, scaling='log', output_shape=imsize_wbord, center=(y, x), mode='edge') for x, y in glimpse_coords_image]
            # lp_glimpses = [warp_polar(noised, scaling='log', output_shape=image_size, center=(y, x), mode='edge') for x, y in glimpse_coords]
            data.at[i, 'logpolar_pixels'] = lp_glimpses
            
            # Plot
            plt.matshow(noised, cmap='Greys', vmin=0, vmax=1)
            plt.axis('off')
            plt.savefig(f'{dirname}/{filename}', bbox_inches='tight', dpi=300,
                        transparent=True, pad_inches=0)
            # plt.imsave(dirname + filename, dpi=300)
            plt.close()
        return data
    
    def generate_dataset(self, config):
        """Fill data frame with toy examples."""
        # numbers = np.arange(num_range[0], num_range[1] + 1)
        numbers = np.arange(config.min_num, config.max_num + 1)
        n_examples = config.size
        n_repeat = np.ceil(n_examples/len(numbers)).astype(int)
        nums = np.tile(numbers[::-1], n_repeat)
        if 'distract' in config.challenge:
            n_distractor_set = [3, 2, 1]
            n_repeat_d = np.ceil(n_examples/len(n_distractor_set)).astype(int)
            n_distract = np.repeat(n_distractor_set, n_repeat_d)
            n_unique = np.empty_like(nums) * np.nan
        elif 'unique' in config.challenge:
            n_distract = np.zeros_like(nums)
            n_unique_set = [1, 2, 3]
            assert config.min_num >= max(n_unique_set)
            n_repeat_u = np.ceil(n_examples/len(n_unique_set)).astype(int)
            n_unique = np.repeat(n_unique_set, n_repeat_u)
        else:
            n_distract = np.zeros_like(nums)
            n_unique = np.empty_like(nums) * np.nan
        data = [self.generate_one_example(nums[i], n_distract[i], n_unique[i], config) for i in range(n_examples)]
        # data = [generate_one_example(nums[i], noise_level, pass_count_range, num_range, shapes_set, n_shapes, same) for i in range(n_examples)]
        df = pd.DataFrame(data)
        return df
    
    
def process_args(conf):
    """Set global variables and convert strings to ints."""
    if conf.shapes[0].isnumeric():
        conf.shapes = [int(i) for i in conf.shapes]
    elif conf.shapes[0].isalpha():
        letter_map = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'I':8,
                      'J':9, 'K':10, 'N':11, 'O':12, 'P':13, 'R':14, 'S':15,
                      'U':16, 'Z':17}
        conf.shapes = [letter_map[i] for i in conf.shapes]
    return conf


def main():
    """Create 8 stimuli sets for 4 conditions + practice trials.
    
    For each task, generate two stimsets for the two viewing conditions. Repeat
    for the practice trials but generate only half the number of imaages.
    
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='PyTorch network settings')
    parser.add_argument('--min_pass', type=int, default=0)
    parser.add_argument('--max_pass', type=int, default=6)
    parser.add_argument('--min_num', type=int, default=2)
    parser.add_argument('--max_num', type=int, default=7)
    parser.add_argument('--shapes', type=list, default=[0, 1, 2, 3, 5, 6, 7, 8])
    parser.add_argument('--noise_level', type=float, default=0.9)
    parser.add_argument('--size', type=int, default=100)
    parser.add_argument('--n_shapes', type=int, default=10, help='How many shapes to the relevant training and test sets span?')
    parser.add_argument('--same', action='store_true', default=False)
    parser.add_argument('--grid', type=int, default=6)
    # parser.add_argument('--distract', action='store_true', default=False)
    # parser.add_argument('--distract_corner', action='store_true', default=False)
    # parser.add_argument('--random', action='store_true', default=False)
    parser.add_argument('--challenge', type=str, default='')
    parser.add_argument('--luminances', nargs='*', type=float, default=[0, 0.5, 1], help='at least two values between 0 and 1')
    parser.add_argument('--solarize', action='store_true', default=False)
    parser.add_argument('--no_glimpse', action='store_true', default=False)
    parser.add_argument('--logpolar', action='store_true', default=False)
    # parser.add_argument('--distract', action='store_true', default=False)
    # parser.add_argument('--distract_corner', action='store_true', default=False)
    # parser.add_argument('--random', action='store_true', default=False)
    parser.add_argument('--n_glimpses', type=int, default=12, help='how many glimpses to generate per image')
    parser.add_argument('--policy', type=str, default='cheat+jitter')
    parser.add_argument('--subject', type=str, default='00')
    conf = parser.parse_args()

    same = 'same' if conf.same else ''
    challenge = f'_{conf.challenge}' if conf.challenge is not None else ''
    # solar = 'solarized_' if conf.solarize else ''
    transform = 'logpolar_' if conf.logpolar else f'gw{conf.glimpse_wid}_'
    shapes = ''.join(conf.shapes)
    if conf.no_glimpse:
        n_glimpses = 'nogl'
    elif conf.n_glimpses is not None:
        n_glimpses = f'{conf.n_glimpses}_'
    else:
        n_glimpses = ''
    policy = conf.policy
    conf = process_args(conf) # make sure this is called after the above shapes =
    
    fldr = f'datasets/stimuli/julian/participant_{conf.subject}'    
    if not os.path.isdir(fldr):
        os.makedirs(fldr)
    
    # Loop through 2(task)x2(view)=4 conditions for practice runs and main
    # experiment, saving one stimset for each of the 8
    set_no = 0
    for challenge, task in zip(['', 'distract'], ['count-all', 'ignore-distractors']):
        for set_, size in product(['A', 'B'], [conf.size, conf.size//2]):
            
            practice = 'main' if size == conf.size else 'practice'
            base = f'num{conf.min_num}-{conf.max_num}_{shapes}{same}_{task}_{set_}_{practice}'
            # Save config details
            with open(f'{fldr}/{base}_config.txt', 'w') as f:
                json.dump(conf.__dict__, f, indent=2)

            print(f'Generating new dataset {base}')
            fname = f'{fldr}/{base}'
            if not os.path.isdir(fname):
                os.makedirs(fname)

            # define_globals(conf)
            conf.challenge = challenge
            conf.seed = int(conf.subject)*10 + set_no  # There are only 8 stim sets per participant so 10 is big enough to never repeat a seed
            generator = DatasetGeneratorJulian(conf)
            
            # Generate toy version, apply symbolic model
            toydata = generator.generate_dataset(conf)  
            toydata['task'] = task
            toydata['set'] = set_
            toydata['practice'] = practice

            # Use symbolic description as template from which to synthesize images and take glimpses
            data = generator.add_logpolar_glimpses_pandas(toydata, fname, conf)
            
            # Save pickle version for easy loading in python
            print(f'Saving {fname}.pkl')
            data.to_pickle(fname + '.pkl')
            
            # Save MATLAB version
            data['initial_filled_locations'] = data['initial_filled_locations'].apply(list)  # Cant save set easily so convert to list
            cols2save = data.columns
            dict_for_mat = {name: col.values for name, col in data.items() if name in cols2save}
            # dict_for_mat = {name: data[name].values for name in data.data_vars}
            savemat(f'{fldr}/image_metadata_{base}.mat', dict_for_mat)
            
            # data = data.drop('shape_map')  # drop before saving because netcdf can't handle dicts
            # print(f'Saving {fname_gw}.nc')
            # data.to_netcdf(fname_gw + '.nc')
            
            # Save to cvs for long term storage
            data.to_csv(fname + '.csv')
            set_no += 1

if __name__ == '__main__':
    main()