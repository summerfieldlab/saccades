"""Generates image datasets for training neural networks on numerical reasoning.

Images

author: Jessica Thompson
date: 2023-06-13
"""
import os
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import torch
from torch import nn
from matplotlib import pyplot as plt
from scipy.io import savemat
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial import distance
from skimage.transform import warp_polar
from itertools import product
import random

import symbolic_model as solver
from letters import get_alphabet
import utils


class NoConvNet(nn.Module):
    """For generating humanlike fixation heatmaps from which to sample fixations."""
    def __init__(self):
        super().__init__()
        self.relu = nn.LeakyReLU()
        # self.fc1 = nn.Linear(72, 75)
        self.fc1 = nn.Linear(36, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 1024)
        self.dropout = nn.Dropout(p=0.5)
        self.fc5 = nn.Linear(1024, 48*48)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.relu(self.fc5(x))
        nex = x.shape[0]
        heat_map = x.view((nex, 1, 48, 48))
        return heat_map, heat_map

class DatasetGenerator():
    def __init__(self, conf):
        random.seed(conf.seed)
        np.random.seed(conf.seed)
        self.n_shapes = conf.n_shapes  # total number of shape classes, the length of any shape vectors, e.g. 25, 
        self.n_glimpses = conf.n_glimpses
        self.conf = conf
        self.eps = 1e-7
        self.char_width = 4
        self.char_height = 5
        self.shape_holder = np.zeros((self.char_height, self.char_width))
        # global GRID, GRID_SIZE, CENTROID_ARRAY, POSSIBLE_CENTROIDS
        # global PIXEL_HEIGHT, PIXEL_WIDTH, MAP_SCALE_PIXEL

        if conf.grid == 3:
            self.grid = [0.2, 0.5, 0.8]
        elif conf.grid == 9:
            self.grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        else:
            # print('Grid size not implemented')
            self.grid = np.linspace(0.1, 0.9, conf.grid, dtype=np.float32)

        self.ncols = len(self.grid)
        self.pixel_height = (self.char_height + 2) * self.ncols
        self.pixel_width = (self.char_width + 2) * self.ncols
        self.pixel_X = [col*(self.char_width + 2) + 1 for col in range(self.ncols)]   #[1, 6, 11] # col *5 + 1
        self.pixel_Y = [row*(self.char_height + 2) + 1 for row in range(self.ncols)]
        self.pixel_topleft = [(x, y) for (x, y) in product(self.pixel_X, self.pixel_Y)]
        self.possible_centroids = [(x, y) for (x, y) in product(self.grid, self.grid)]
        self.centroid_array = np.array(self.possible_centroids)
        # POSSIBLE_CENTROIDS = POSSIBLE_CENTROIDS.copy()
        # if conf.distract_corner:
        #     del POSSIBLE_CENTROIDS[0]
        # Map from the 1x1 reference frame to the x,y coordinate of the top left pixel for the corresponding slot
        self.map_scale_pixel = {scaled:pixel for scaled,pixel in zip(self.possible_centroids, self.pixel_topleft)}
        self.grid_size = len(self.possible_centroids)
        
         
    def get_xy_coords(self, numerosity, n_distract, noise_level, challenge, policy):
        """Randomly select n spatial locations and take noisy observations of them.

        Objects are placed on a grid within a 1x1 square. In the case of a 3x3
        grid, there are nine possible true locations corresponding to
        product([0.2, 0.5, 0.8], [0.2, 0.5, 0.8])

        These objects generate a sequence of glimpse coordinates within the same
        1x1 space, according to a saliency map.
        
        This generates an abstract set of coordinates which may or maynot be
        used to then synthesize glimpes of an image. Some of the variable naming
        is confusing as a consequence of having originally written this for
        toy simulations without images.

        TODO: change how I parameterize noise level to be more intuitive. Make it
        as fraction of distance between objects in the scaled 1x1 space. 0.57
        creates good distribution over integration scores
        
        Arguments:
            numerosity (int): how many obejcts to place
            noise level (float): how noisy are the observations. nl*0.1/2 = scale. nl*0.1 = cutoff
            items in the array (not including any random glimpses)
            distract (bool): Whether to include distractor objects

        Returns:
            array: x,y coordinates of the glimpses (noisy observations)
            array: symbolic rep (index into grid) of the glimpsed objects, which true locations
            array: x,y coordinates of the above
        """
        # numerosity = 4
        # noise_level = 1.6
        # nl = 1
        nl = 0.1 * noise_level
        # min_dist = 0.3
        # n_glimpses = 4

        if '+random' in challenge:
            # Number of random glimpses to be included in the set of glimpes
            n_rand_gl = random.choice([0, 1, 2])
            # Total number of glimpses should still be constant, so if adding less
            # than two random glimpses, add extra glimpses generated by items
            n_glimpses = self.n_glimpses + 2 - n_rand_gl
        else:
            n_glimpses = self.n_glimpses

        n_symbols = len(self.possible_centroids)
        # Randomly select where to place objects
        locations = range(1, n_symbols) if 'corner' in challenge else range(0, n_symbols)
        objects2count = random.sample(locations, numerosity)
        object_coords = [self.possible_centroids[i] for i in objects2count]
        distractors = []
        if 'distract' in challenge:
            # randomly choose whether to place 0, 1, or 2 distractor objects
            # n_distract = random.choice([0, 1, 2])
            # select where to place distractor(s)
            empty_locs = [i for i in range(n_symbols) if i not in objects2count]
            if n_distract > 0:
                if 'corner' in challenge:
                    distractors = [0 for _ in range(n_distract)]
                else:
                    distractors = random.sample(empty_locs, n_distract)
            if 'boost_target' in challenge:
                all_objects = objects2count + distractors + objects2count
                assert n_glimpses >= len(all_objects)
            elif 'only_target' in challenge:
                all_objects = objects2count
            else:
                all_objects = objects2count + distractors
        else:
            all_objects = objects2count

        if policy == 'random':
            glimpse_coords = np.array([(np.random.uniform(0.05, 0.95), np.random.uniform(0.05, 0.95)) for _ in range(n_glimpses)])
            glimpsed_objects = np.array([-1 for _ in range(n_glimpses)])
            coords_glimpsed_objects = [self.possible_centroids[0] for _ in range(n_glimpses)]
        else:
            if policy == 'all_slots':
                assert self.n_glimpses == len(self.possible_centroids)
                glimpsed_objects = [i for i in range(len(self.possible_centroids))]
                random.shuffle(glimpsed_objects)
                coords_glimpsed_objects = [self.possible_centroids[object] for object in glimpsed_objects]
            else:
                # Each glimpse is associated with an object idx
                # All objects must be glimpsed, but the glimpse order should be random
                glimpse_candidates = all_objects.copy()
                while len(glimpse_candidates) < n_glimpses:
                    to_append = glimpse_candidates.copy()
                    random.shuffle(to_append)
                    glimpse_candidates += to_append

                glimpsed_objects = glimpse_candidates[:n_glimpses]
                random.shuffle(glimpsed_objects)
                coords_glimpsed_objects = [self.possible_centroids[object] for object in glimpsed_objects]

            # Take noisy observations of glimpsed objects
            noise_x = np.random.normal(loc=0, scale=nl/2, size=n_glimpses)
            noise_y = np.random.normal(loc=0, scale=nl/2, size=n_glimpses)
            # Enfore a hard bound on how far the glimpse coordinate can be from the item that generated it.
            while any(abs(noise_x) > nl):
                idx = np.where(abs(noise_x) > nl)[0]
                noise_x[idx] = np.random.normal(loc=0, scale=nl, size=len(idx))
            while any(abs(noise_y) > nl):
                idx = np.where(abs(noise_y) > nl)[0]
                noise_y[idx] = np.random.normal(loc=0, scale=nl, size=len(idx))
            # Add noise to generate sequence of saccadic targets
            noise_xy = zip(noise_x, noise_y)
            glimpse_coords = [(x+del_x, y+del_y) for ((x,y), (del_x, del_y)) in zip(coords_glimpsed_objects, noise_xy)]
            # Add 0,1 or 2 random glimpses
            if 'random' in challenge:
                for r_gl in range(n_rand_gl):
                    glimpse_coords.append((np.random.uniform(0.05, 0.95), np.random.uniform(0.05, 0.95)))
                    # glimpsed_objects.append(None)
                # shuffle order of glimpses
                random.shuffle(glimpse_coords)
            glimpse_coords = np.array(glimpse_coords, dtype=np.float32)
            glimpsed_objects = np.array(glimpsed_objects)
            coords_glimpsed_objects = np.array(coords_glimpsed_objects)
        return glimpse_coords, glimpsed_objects, coords_glimpsed_objects, objects2count, distractors
    
    def calculate_proximity(self, glimpse_coords, item_coords, item_slots, shape_map, max_dist):
        """Calculate proximity of fixation point to nearby item shape categories. 
        
        e.g. proximity to the letter A will be the first element of the vector, 
        which may be greater than 1 if there are multiple As in the vicinity.
        """
        shape_coords = np.zeros((self.n_glimpses, self.n_shapes), dtype=np.float32)
        # Calculate a weighted 'glimpse label' that indicates the proximity of each
        # glimpse to neighboring shapes
        eu_dist = euclidean_distances(glimpse_coords, item_coords)
        # max_dist = np.sqrt((noise_level*0.1)**2 + (noise_level*0.1)**2)
        glimpse_idx, object_idx = np.where(eu_dist <= max_dist + self.eps)
        shape_idx = [shape_map[obj] for obj in item_slots[object_idx]]
        # proximity = {idx:np.zeros(9,) for idx in glimpse_idx}
        # The same cell of the shape_coords matrix may by incremented several
        # times if there are multiple occurances of a shape "within" a single
        # glimpse.
        for gl_idx, obj_idx, sh_idx in zip(glimpse_idx, object_idx, shape_idx):
            if max_dist != 0:
                prox = 1 - (eu_dist[gl_idx, obj_idx]/max_dist)
            else:  # avoid division by zero
                prox  = 1
            # print(f'gl:{gl_idx} obj:{obj_idx} sh:{sh_idx}, prox:{prox}')
            shape_coords[gl_idx, sh_idx] += prox
        return shape_coords
        
    def get_shape_coords(self, glimpse_coords, max_dist, shapes_set, same,
                         objects2count, distractors):
        """Generate glimpse shape feature vectors.

        Each object is randomly assigned one shape. The shape feature
        vector indicates the proximity of each glimpse to objects of each shape.
        If there are no objects of shape m within a radius of max_dist from the
        glimpse, the mth element of the shape vector will be 0. If the glimpse
        coordinates are equal to the coordinates of an object of shape m, the mth
        element of the shape feature vector will be 1.

        Args:
            glimpse_coords (array): xy coordinates of the glimpses
            objects (array): Symbolic representation (0-8) of the object locations
            noiseless_coords (array of tuples): xy coordinates of the above
            max_dist (float): maximum distance to use when bounding proximity measure
            shapes_set (list): The set of shape indexes from which shapes should be sampled
            n_shapes (int): how many shapes exist in this world (including other
                            datasets). Must be greater than the maximum entry in
                            shapes_set.
            same (bool): whether all shapes within one image should be identical
            objects2count (list): location indices of objects to count
            distractors (list): location indices of distractor objects

        Returns:
            array: nxn_shapes array where n is number of glimpses.
            dict: maps numbered locations to shape identities
            list: histogram representation of how many of each shape are in the image

        """
        #
        
        distractor_shape = 0
        assert distractor_shape not in shapes_set

        # these two need to be in the same order. unique just sorts them which works but is not necessary
        # item_slots = np.unique(glimpsed_objects) # 
        # unique_objects = np.array(objects2count + distractors)
        # item_coords = np.unique(noiseless_coords, axis=0)
        
        item_slots = np.array([i for i in range(len(self.possible_centroids)) if (i in objects2count or i in distractors)])
        item_coords = np.array([self.possible_centroids[i] for i in item_slots])
        num = len(objects2count)
        
        # Assign a random shape to each object
        if same:  # All shapes in the image will be the same
            shape = np.random.choice(shapes_set)
            shape_assign = np.repeat(shape, num)
        else:
            shapes_copy = shapes_set.copy()
            random.shuffle(shapes_copy)
            shape_assign = np.tile(shapes_copy, int(np.ceil(num/len(shapes_set))))[:num]
            # shape_assign = np.random.choice(shapes_set, size=num, replace=True)
        dist_map = {dist_loc: distractor_shape for dist_loc in distractors}
        shape_map = {object: shape for (object, shape) in zip(objects2count, shape_assign)}
        shape_map.update(dist_map)
        # shape_map = {object: shape for (object, shape) in zip(unique_objects, shape_assign)}
        shape_map_vector = np.ones((self.grid_size,)) * -1 # 9 for the 9 locations?
        for object in shape_map.keys():
            shape_map_vector[object] = shape_map[object]
        shape_hist = [sum(shape_map_vector == shape) for shape in range(self.n_shapes)]

        shape_coords = self.calculate_proximity(glimpse_coords, item_coords, item_slots, shape_map, max_dist)
        # shape_coords[glimpse_idx, shape_idx] = 1 - eu_dist[glimpse_idx, obj_idx]/min_dist
        # make sure each glimpse has at least some shape info (not if random glimpses)
        # assert np.all(shape_coords.sum(axis=1) > 0)
        return shape_coords, shape_map, shape_hist

    def generate_one_example(self, num, n_disract, n_unique, config):
        """Synthesize a single toy glimpse sequence and apply symbolic solver."""
        same = config.same
        # distract = config.distract
        # rand_g = config.random
        noise_level = config.noise_level
        min_pass_count = config.min_pass
        max_pass_count = config.max_pass
        num_low = config.min_num
        num_high = config.max_num
        num_range = [num_low, num_high]
        if np.isnan(n_unique):
            shapes_set = config.shapes
        else:
            shapes_set = random.sample(config.shapes, n_unique)
        n_shapes = config.n_shapes
        challenge = config.challenge
        policy = config.policy

        # if config.distract:
        #     challenge = 'distract'
        # elif config.distract_corner:
        #     challenge = 'distract_corner'
        # elif config.random:
        #     challenge = 'random'
        if config.n_glimpses is not None:
            n_glimpses = config.n_glimpses
        else:
            n_glimpses = num_high 
            if ('distract' in challenge):
                n_glimpses += 2
                num_range[1] += 2
        
        # The number of glimpses will also be increased in the random glimpse
        # condition but they will be added at the end. This parameter therefore is
        # how many glimpses will be generated by objects, not including those that
        # are generated uniform randomly.

        # Synthesize glimpses - paired observations of xy and shape coordinates
        max_dist = np.sqrt((noise_level*0.1)**2 + (noise_level*0.1)**2)
        # 0.12727922061357858
        # min_pass_count, max_pass_count = pass_count_range
        # num_low, num_high = num_range
        # num = random.randrange(num_low, num_high + 1)
        final_pass_count = -1
        # This loop will continue synthesizing examples until it finds one within
        # the desired pass count range. The numerosity is determined before hand so
        # that limiting the pass count doesn't bias the distribution of numerosities
        # TODO: Make this such that pass count range is only considered if included as cli argument. 
        # Otherwise, range set to be inclusive. That way don't need to bother with setting those cli arguments when not relevant
        while final_pass_count < min_pass_count or final_pass_count > max_pass_count:
            xy_coords, objects, noiseless_coords, to_count, distractors = self.get_xy_coords(num, n_disract, noise_level, challenge, policy)
            shape_coords, shape_map, shape_hist = self.get_shape_coords(xy_coords, max_dist, shapes_set, same, to_count, distractors)
            # Initialize records
            example = solver.GlimpsedImage(xy_coords, shape_coords, shape_map, shape_hist, objects, max_dist, num_range)
            pass_count = 0
            done = example.check_if_done(pass_count)

            # First pass
            example.process_xy()
            if not done:
                pass_count += 1
                done = example.check_if_done(pass_count)
            initial_candidates = example.candidates.copy()
            initial_filled_locations = example.filled_locations.copy()

            while not done and pass_count < max_pass_count:
                pass_count += 1

                tbr = example.to_be_resolved

                keys = list(tbr.keys())
                idx = keys[0]
                cand_list, loc_list = tbr[idx]
                new_object_idxs = example.use_shape_to_resolve(idx, cand_list)

                for loc in new_object_idxs:
                    example.toggle(idx, loc)

                # Check if done
                done = example.check_if_done(pass_count)

            if not done:
                # print('DID NOT SOLVE \r')
                example.pred_num = example.lower_bound
                # example.plot_example(pass_count)

            final_pass_count = pass_count
        # unique_objects = set(example.objects)
        all_objects = to_count + distractors
        filled_locations = [1 if i in all_objects else 0 for i in range(self.grid_size)]
        locations_2count = [1 if i in to_count else 0 for i in range(self.grid_size)]
        locations_dist = [1 if i in distractors else 0 for i in range(self.grid_size)]
        # these won't be exactly correct because of the small outerborder.
        target_coords_1x1 = [self.possible_centroids[target] for target in to_count]
        distract_coords_1x1 = [self.possible_centroids[distract] for distract in distractors]
        example_dict = {'glimpse_coords_1x1': xy_coords, 'symbolic_shape': shape_coords,
                        'numerosity_target': num,
                        'numerosity_dist': len(distractors),
                        'num_unique': n_unique,
                        'num_min': example.min_num,
                        'predicted_num': example.pred_num, 'count': example.count,
                        'locations': filled_locations,
                        'locations_count': locations_2count,
                        'locations_distract': locations_dist,
                        'object_coords': noiseless_coords,
                        'shape_map': shape_map,
                        'pass_count': pass_count, 'unresolved_ambiguity': not done,
                        'special_xy': example.special_case_xy,
                        'special_shape': example.special_case_shape,
                        'lower_bound': example.lower_bound,
                        'upper_bound': example.upper_bound,
                        'min_shape': example.min_shape, 
                        'initial_candidates': initial_candidates,
                        'initial_filled_locations': initial_filled_locations,
                        'shape_hist': shape_hist,
                        'target_coords_1x1': target_coords_1x1,
                        'distract_coords_1x1': distract_coords_1x1
                        }
        return example_dict

    def generate_dataset(self, config):
        """Fill data frame with toy examples."""
        # numbers = np.arange(num_range[0], num_range[1] + 1)
        numbers = np.arange(config.min_num, config.max_num + 1)
        n_examples = config.size
        n_repeat = np.ceil(n_examples/len(numbers)).astype(int)
        nums = np.tile(numbers[::-1], n_repeat)
        if 'distract012' in config.challenge:
            n_distractor_set = [2, 1, 0]
            # n_distractor_set = [3, 2, 1]
            n_repeat_d = np.ceil(n_examples/len(n_distractor_set)).astype(int)
            n_distract = np.repeat(n_distractor_set, n_repeat_d)
            n_unique = np.empty_like(nums) * np.nan
        elif 'distract123' in config.challenge:
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
        elif config.challenge != '':
            print(f'Challenge {config.challenge} not implemented. Exiting.')
            exit()
        else:
            n_distract = np.zeros_like(nums)
            n_unique = np.empty_like(nums) * np.nan

        data = [self.generate_one_example(nums[i], n_distract[i], n_unique[i], config) for i in range(n_examples)]
        # data = [generate_one_example(nums[i], noise_level, pass_count_range, num_range, shapes_set, n_shapes, same) for i in range(n_examples)]
        df = pd.DataFrame(data)
        return df
    
    def pandas_to_xr(self, df):
        """Convert pandas DataFrame to Xarray Dataset."""
        def get_stackable_shape_map(shape_map):
            stackable = np.zeros((36,))
            for key in shape_map.keys():
                stackable[key] = shape_map[key]
            return stackable
        
        ds = xr.Dataset(
            {
                "glimpse_coords_1x1": (["image", "glimpse", "coordinate"], np.stack(df['glimpse_coords_1x1'].to_numpy())),
                "symbolic_shape": (["image", "glimpse", "character"], np.stack(df['symbolic_shape'].to_numpy())),
                'numerosity_target': (["image"], np.stack(df['numerosity_target'].to_numpy())),
                'numerosity_dist': (["image"], np.stack(df['numerosity_dist'].to_numpy())),
                'num_unique': (["image"], np.stack(df['num_unique'].to_numpy())),
                'num_min': (["image"], np.stack(df['num_min'].to_numpy())),
                'predicted_num': (["image"], np.stack(df['predicted_num'].to_numpy())),
                'locations': (["image", "slot"], np.stack(df['locations'].to_numpy())),
                'locations_count': (["image", "slot"], np.stack(df['locations_count'].to_numpy())),
                'locations_distract': (["image", "slot"], np.stack(df['locations_distract'].to_numpy())),
                'object_coords': (["image", "glimpse", "coordinates"], np.stack(df['object_coords'].to_numpy())),
                'shape_map': (["image", "slot"], np.stack(df['shape_map'].apply(get_stackable_shape_map))),
                'pass_count': (["image"], np.stack(df['pass_count'].to_numpy())),
                'unresolved_ambiguity': (["image"], np.stack(df['unresolved_ambiguity'].to_numpy())),
                'special_xy': (["image"], np.stack(df['special_xy'].to_numpy())),
                'special_shape': (["image"], np.stack(df['special_shape'].to_numpy())),
                'lower_bound': (["image"], np.stack(df['lower_bound'].to_numpy())),
                'upper_bound': (["image"], np.stack(df['upper_bound'].to_numpy())),
                'min_shape': (["image"], np.stack(df['min_shape'].to_numpy())),
                # 'initial_candidates': (["image"], df['initial_candidates']),
                # 'initial_filled_locations': (["image"], df['initial_filled_locations']),
                'shape_hist': (["image", "character"], np.stack(df['shape_hist'].to_numpy())),
                # "pixel_topleft": self.pixel_topleft,
                # "possible_centroids": self.possible_centroids,
                # "map_scale_pixel": self.map_scale_pixel,
                # "centroid_array": self.centroid_array
            },
            coords={
                "image": range(len(df)),
                "glimpse": range(self.n_glimpses),
                "coordinate": ["x", "y"],
                "character": range(25),
                "slot": range(36)
            },
            attrs={
                "char_width": self.char_width,
                "char_height": self.char_height,
                "grid": self.grid,
                "pixel_height": self.pixel_height,
                "pixel_width": self.pixel_width,
            }
        )
        return ds

    def get_solarized_noise(self, image, fg_lum, bg_lum):
        """Sample bg and fg luminances from distributions.

        Args:
            image (array): image template of just zeros and ones
            fg_lum (float): foregound luminance value
            bg_lum (float): backgound luminance value

        Returns:
            array: solarized image. zeros mapped to bg lum, ones mapped to fg lum
        """

        dict = {0: bg_lum, 1: fg_lum}
        # dict = {0: np.random.normal(loc=bg_lum, scale=0.1), 1: np.random.normal(loc=fg_lum, scale=0.1)}
        solarized = np.vectorize(dict.get)(image)
        noised = np.vectorize(np.random.normal)(loc=solarized, scale=0.05)
        noised = np.float32(noised)
        return noised

    # def add_logpolar_glimpses_pandas(self, data, conf):
    #     """Synthesize and glimpse images from symbolic description."""
    #     glim_wid = conf.glimpse_wid
    #     lums = conf.luminances
    #     half_glim = glim_wid//2
    #     chars = get_alphabet()
    #     data['glimpse_coords'] = None
    #     data['luminances'] = None
    #     data['noised_image'] = None
    #     data['centre_fixation'] = None
    #     data['logpolar_pixels'] = None

    #     for i in range(len(data)):
    #         filename = f'image_{i}.png'
    #         if not i % 10:
    #             print(f'Synthesizing image {i}', end='\r')
    #         row = data.iloc[i]
    #         # Coordinates in 1x1 space
    #         object_xy_coords = self.centroid_array[np.where(row.locations)[0]]
    #         # Calculate saliency map
    #         # Saliency map should be the size of the image WITHOUT the added border
    #         image_size_nobord = (self.pixel_height, self.pixel_width)
    #         object_pixel_coords = [self.map_scale_pixel[tuple(xy)] for xy in object_xy_coords]
    #         # Shape indices for the objects
    #         object_shapes = [row['shape_map'][obj] for obj in np.where(row.locations)[0]]

    #         # Insert the specified shapes into the image at the specified locations
    #         image = np.zeros(image_size_nobord, dtype=np.float32)
    #         for shape_idx, (x,y) in zip(object_shapes, object_pixel_coords):
    #                 image[y:y+self.char_height:, x:x+self.char_width] = chars[shape_idx]

    #         # Convert scaled glimpse coordinates to pixel coordinates
    #         scaled_glimpse_coords = row.xy.copy()
    #         scaled_glimpse_coords[:, 0] = scaled_glimpse_coords[:, 0]*self.pixel_width
    #         scaled_glimpse_coords[:, 1] = scaled_glimpse_coords[:, 1]*self.pixel_height
    #         glimpse_coords = np.round(scaled_glimpse_coords).astype(int)
    #         data.at[i, 'glimpse_coords'] = glimpse_coords
    #         imsize = [self.pixel_height+glim_wid, self.pixel_width+glim_wid]
    #         image_wbord = np.zeros(imsize, dtype=np.float32)
    #         image_wbord[half_glim:-half_glim,half_glim:-half_glim] = image
    #         # data.at[i, 'bw image'] = image_wbord
    #         glimpse_coords += half_glim
            
    #         # Solarize and add Gaussian noise
    #         fg, bg = np.random.choice(lums, size=2, replace=False)
    #         # ensure that the difference between the foreground and background
    #         # is at least 0.2, which is the smallest difference in the test sets
    #         while abs(fg - bg) < 0.2:
    #             fg, bg = np.random.choice(lums, size=2, replace=False)
    #         data.at[i, 'luminances'] = [fg, bg]
    #         noised = self.get_solarized_noise(image_wbord, fg, bg)
    #         data.at[i, 'noised_image'] = noised

    #         # Warp noised image to generaye log-polar glimpses
    #         # Fixed gaze at the centre
    #         centre = [size//2 for size in imsize]
    #         fixation = warp_polar(noised, scaling='log', output_shape=imsize, center=centre, mode='edge')  # rows, cols
    #         data.at[i, 'centre_fixation'] = fixation
    #         # Glimpses
    #         lp_glimpses = [warp_polar(noised, scaling='log', output_shape=imsize, center=(y, x), mode='edge') for x, y in glimpse_coords]
    #         data.at[i, 'logpolar_pixels'] = lp_glimpses
    #     return data
    
    def sample_from_smooth_map(self, smooth_map, inhibition=False):
        dist = smooth_map
        dist = nn.functional.relu(dist).squeeze().detach().numpy()
        dist = dist/dist.sum()

        # Create a flat copy of the array
        flat = dist.flatten()

        # Then, sample an index from the 1D array with the
        # probability distribution from the original array
        n_samples = self.n_glimpses
        if inhibition:
            human_like_glimpse_coords = []
            
            for sample in range(n_samples):
                sample_index = np.random.choice(a=flat.size, p=flat, size=1)
                adjusted_index = np.unravel_index(sample_index, dist.shape)
                human_like_glimpse_coords.extend(adjusted_index)
                # Inhibit location of selected fixation
                inhibitory = np.zeros_like(dist)
                inhibitory[adjusted_index[0], adjusted_index[1]] = -1

        else:
            # If not implementing inhibition of return, then can take all samples at once
            sample_index = np.random.choice(a=flat.size, p=flat, size=n_samples)

            # Take this index and adjust it so it matches the original array
            adjusted_index = [np.unravel_index(index, dist.shape) for index in sample_index]
            human_like_glimpse_coords = [[np.float32(x/48.0), np.float32(y/48.0)] for y, x in adjusted_index]
            
        

        return human_like_glimpse_coords

    def add_logpolar_glimpses_xr(self, data_pd, conf):
        print('Adding logpolar glimpses...')
        if conf.policy == 'humanlike':
            # Load generative model of human fixations
            fix_dir = '/home/jessica/Dropbox/saccades/eye_tracking/'
            fixator  = torch.load(fix_dir + 'noconv_onlypretrain_drop50_mse_20eps20230915-17.56.42.pt')  #'sparse0.01_fixator_20230913-14.24.33.pt')
            fixator = fixator.to('cpu')

        # Convert to xarray
        data = self.pandas_to_xr(data_pd)
        chars = get_alphabet()
        glim_wid = 6
        lums = conf.luminances
        half_glim = glim_wid//2
        n_images = len(data.image)
        image_size_nobord = [self.pixel_height, self.pixel_width]
        imsize_wbord = [self.pixel_height+glim_wid, self.pixel_width+glim_wid]
        data['glimpse_coords_image'] = (("image", "glimpse", "coordinate"), np.empty((n_images, self.n_glimpses, 2), dtype=np.float32))
        data['glimpse_coords_scaled'] = (("image", "glimpse", "coordinate"), np.empty((n_images, self.n_glimpses, 2), dtype=np.float32))
        data['glimpse_coords_humanlike'] = (("image", "glimpse", "coordinate"), np.empty((n_images, self.n_glimpses, 2), dtype=np.float32))
        data['symbolic_shape_humanlike'] = (("image", "glimpse", "character"), np.empty((n_images, self.n_glimpses, self.n_shapes), dtype=np.float32))

        data['luminances'] = (("image", "ground"), np.empty((n_images, 2)))
        data['noised_image'] = (("image", "row", "column"), np.empty((n_images, 48, 42), dtype=np.float32))
        data['centre_fixation'] =(("image", "row", "column"), np.empty((n_images, 48, 42), dtype=np.float32))
        data['logpolar_pixels'] = (("image", "glimpse", "row", "column"), np.empty((n_images, self.n_glimpses, 48, 42), dtype=np.float32))

        # data['logpolar_pixels_humanlike'] = (("image", "glimpse", "row", "column"), np.empty((n_images, self.n_glimpses, 48, 42), dtype=np.float32))
            
        data_pd['target_coords_scaled'] = None
        data_pd['distract_coords_scaled'] = None
        for i in range(n_images):
            if not i % 10:
                print(f'Synthesizing image {i}', end='\r')
            row_pd = data_pd.iloc[i]
            row = data.isel(image=i)
            
            # PREPARE IMAGE
            # Get coordinates of items
            slots =  np.where(row.locations.values)[0]
            object_xy_coords = self.centroid_array[slots]

            object_pixel_coords = [self.map_scale_pixel[tuple(xy)] for xy in object_xy_coords] # Where to insert letters
            # Shape indices for the objects
            object_shapes = [row_pd['shape_map'][obj] for obj in slots]

            # Insert the specified shapes into the image at the specified locations
            image = np.zeros(image_size_nobord, dtype=np.float32)
            for shape_idx, (x,y) in zip(object_shapes, object_pixel_coords):
                image[y:y+self.char_height:, x:x+self.char_width] = chars[shape_idx]

            # Add small border to image (3 pixels on all sides)
            image_wbord = np.zeros(imsize_wbord, dtype=np.float32)
            image_wbord[half_glim:-half_glim,half_glim:-half_glim] = image
            # Solarize and add Gaussian noise to image
            fg, bg = np.random.choice(lums, size=2, replace=False)
            # ensure that the difference between the foreground and background
            # is at least 0.2, which is the smallest difference in the test sets
            while abs(fg - bg) < 0.2:
                fg, bg = np.random.choice(lums, size=2, replace=False)
            # data.at[i, 'luminances'] = [fg, bg]
            data['luminances'].loc[dict(image=i)] = [fg, bg]
            noised = self.get_solarized_noise(image_wbord, fg, bg)
            # data.at[i, 'noised_image'] = noised
            data['noised_image'].loc[dict(image=i)] = noised
            
            # PREPARE COORDINATES
            # Get target and distractor coordinates in the same scaled frame of 
            # reference as the glimpse coords. Important for analysis of 
            # activations and eye tracking.
            target_slots = np.where(row_pd['locations_count'])[0]
            target_coords_1x1 = self.centroid_array[target_slots]
            # target_coords_image = (target_coords_1x1 * image_size_nobord[::-1]) + half_glim
            target_coords_image = np.array([self.map_scale_pixel[tuple(xy)]for xy in target_coords_1x1]) + half_glim 
            target_coords_image = target_coords_image + [(self.char_width/2) - 0.5, (self.char_height/2) - 0.5]
            target_coords_scaled = target_coords_image/imsize_wbord[::-1]
            data_pd.at[i, 'target_coords_scaled'] = target_coords_scaled
            
            distract_slots = np.where(row_pd['locations_distract'])[0]
            if len(distract_slots) > 0:
                distract_coords_1x1 = self.centroid_array[distract_slots]
                # distract_coords_image = (distract_coords_1x1 * image_size_nobord[::-1]) + half_glim
                distract_coords_image = np.array([self.map_scale_pixel[tuple(xy)]for xy in distract_coords_1x1]) + half_glim 
                distract_coords_image = distract_coords_image + [(self.char_width/2) - 0.5, (self.char_height/2) - 0.5]
                distract_coords_scaled = distract_coords_image/imsize_wbord[::-1]
                data_pd.at[i, 'distract_coords_scaled'] = distract_coords_scaled
                
            # Convert 1x1 glimpse coordinates of glimpse and locations to pixel 
            # coordinates and scale between 0 and 1 (because of image border)
            glimpse_coords_1x1 = row.glimpse_coords_1x1.values.copy()
            glimpse_coords_image = np.multiply(glimpse_coords_1x1, image_size_nobord[::-1]) + half_glim
            glimpse_coords_image = np.round(glimpse_coords_image).astype(int)
            # data.at[i, 'glimpse_coords'] = glimpse_coords
            data['glimpse_coords_image'].loc[dict(image=i)] = glimpse_coords_image
            data['glimpse_coords_scaled'].loc[dict(image=i)] = np.divide(glimpse_coords_image, imsize_wbord[::-1])
            
            # PREPARE GLIMPSE CONTENTS
            if conf.policy == 'humanlike':
                # **Get humanlike glimpse coordinates**
                # Prepare the input features for the generative model
                locations_count_T = utils.transpose(row.locations_count.values)
                locations_distract_T = utils.transpose(row.locations_distract.values)
                fixator_features = np.array(locations_count_T) + 0.5*np.array(locations_distract_T)
                features = torch.from_numpy(fixator_features).float()
        
                # Use pretained generative model to produce humanlike fixation heatmaps
                _, smooth_map = fixator(features.unsqueeze(0))
                humanlike_coords = self.sample_from_smooth_map(smooth_map)
                data['glimpse_coords_humanlike'].loc[dict(image=i)] = humanlike_coords
                            
                # Calculate shape proximity vector for humanlike glimpses
                max_dist = 0.25
                shape_coords = self.calculate_proximity(humanlike_coords, object_xy_coords, slots, row_pd.shape_map, max_dist)
                data["symbolic_shape_humanlike"].loc[dict(image=i)] = shape_coords
                
                # Warp noised image to generate log-polar glimpses
                lp_glimpses = [warp_polar(noised, scaling='log', output_shape=imsize_wbord, center=(y*48, x*48), mode='edge') for x, y in humanlike_coords]
            else:
                # Warp noised image to generate log-polar glimpses
                lp_glimpses = [warp_polar(noised, scaling='log', output_shape=imsize_wbord, center=(y, x), mode='edge') for x, y in glimpse_coords_image]
            # data.at[i, 'logpolar_pixels'] = lp_glimpses
            data['logpolar_pixels'].loc[dict(image=i)] = lp_glimpses
            
            # Fixed gaze at the centre
            centre = [size//2 for size in imsize_wbord]
            fixation = warp_polar(noised, scaling='log', output_shape=imsize_wbord, center=centre, mode='edge')  # rows, cols
            # data.at[i, 'centre_fixation'] = fixation
            data['centre_fixation'].loc[dict(image=i)] = fixation

        return data, data_pd
    
    
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
    parser.add_argument('--policy', type=str, default='cheat+jitter')
    parser.add_argument('--luminances', nargs='*', type=float, default=[0, 0.5, 1], help='at least two values between 0 and 1')
    parser.add_argument('--solarize', action='store_true', default=False)
    parser.add_argument('--no_glimpse', action='store_true', default=False)
    parser.add_argument('--logpolar', action='store_true', default=False)
    # parser.add_argument('--distract', action='store_true', default=False)
    # parser.add_argument('--distract_corner', action='store_true', default=False)
    # parser.add_argument('--random', action='store_true', default=False)
    parser.add_argument('--n_glimpses', type=int, default=12, help='how many glimpses to generate per image')
    parser.add_argument('--seed', type=int, default=0)
    conf = parser.parse_args()

    same = 'same' if conf.same else ''
    challenge = f'_{conf.challenge}' if conf.challenge != '' else ''
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
    conf = process_args(conf)  # make sure this line is after 'shapes =' above

    # define_globals(conf)
    generator = DatasetGenerator(conf)
    toydata = generator.generate_dataset(conf)  # Generate toy version, apply symbolic model
    # data = generator.add_logpolar_glimpses_pandas(toydata, conf)
    data, data_pd = generator.add_logpolar_glimpses_xr(toydata, conf)
    
    dirname = 'datasets/image_sets'
    fname_gw = f'{dirname}/num{conf.min_num}-{conf.max_num}_nl-{conf.noise_level}_{shapes}{same}{challenge}_grid{conf.grid}_policy-{policy}_lum{conf.luminances}_{transform}{n_glimpses}{conf.size}'
    if not os.path.isdir(fname_gw):
            os.makedirs(fname_gw)
            
    # Save example images for viewing
    sample = np.random.choice(np.arange(len(data.image)), 10)
    for idx in sample:
        # plt.matshow(data.iloc[idx]['noised_image'], vmin=0, vmax=1, cmap='Greys')
        plt.matshow(data['noised_image'].loc[dict(image=idx)], vmin=0, vmax=1, cmap='Greys')
        target_loc = data_pd.iloc[idx].target_coords_scaled * [42.0, 48.0]
        plt.scatter(target_loc[:, 0], target_loc[:, 1], label='targets')
        distract_loc = data_pd.iloc[idx].distract_coords_scaled 
        if distract_loc is not None:
            distract_loc = distract_loc * [42.0, 48.0]
            plt.scatter(distract_loc[:, 0], distract_loc[:, 1], label='distractor')
        plt.legend()
        plt.axis('off')
        
        plt.savefig(f'{fname_gw}/example_{idx}.png', bbox_inches='tight', dpi=300, transparent=True, pad_inches=0)
        plt.close() 
    
    # print(f'Saving {fname_gw}.pkl')
    # data_pd.to_pickle(fname_gw + '.pkl')
    # data = data.drop('shape_map')# drop before saving because netcdf can't handle dicts
    # data['noised_image'] -= data['noised_image'].min()
    # data['noised_image'] /= data['noised_image'].max()
    # data['logpolar_pixels'] -= data['logpolar_pixels'].min()
    # data['logpolar_pixels'] /= data['logpolar_pixels'].max()
    # WE WANT TO PRESERVE DIFFERENCES IN LUMINANCE BETWEEN DIFFERENT DATASETS
    print(f'Saving {fname_gw}.nc')
    data.to_netcdf(fname_gw + '.nc')

if __name__ == '__main__':
    main()