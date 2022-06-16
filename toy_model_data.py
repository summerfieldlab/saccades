import sys
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial import distance
from itertools import product, combinations
import random

# EPS = sys.float_info.epsilon
EPS = 1e-7

GRID = [0.2, 0.5, 0.8]
POSSIBLE_CENTROIDS = [(x, y) for (x, y) in product(GRID, GRID)]
CENTROID_ARRAY = np.array(POSSIBLE_CENTROIDS)


def get_xy_coords(numerosity, noise_level):
    """Randomly select n spatial locations and take noisy observations of them.

    Nine possible true locations corresponding to a 3x3 grid within a 1x1 space
    locations are : product([0.2, 0.5, 0.8], [0.2, 0.5, 0.8])

    Arguments:
    numerosity - how many obejcts to place
    noise level - how noisy are the observations. 1 means add upto +/- 0.1. Any
        noise level > 1.5 creates ambiguity since the objects are 0.3 apart

    Returns:
    x,y coordinates of the glimpses (noisy observations)
    symbolic rep (index 0-8) of the glimpsed objects, which true locations
    x,y coordinates of the above
    """
    # numerosity = 4
    # noise_level = 1.6
    # nl = 1
    nl = 0.1 * noise_level
    # min_dist = 0.3
    n_glimpses = 4

    n_symbols = len(POSSIBLE_CENTROIDS)
    # Randomly select where to place objects
    objects = random.sample(range(n_symbols), numerosity)
    object_coords = [POSSIBLE_CENTROIDS[i] for i in objects]

    # Each glimpse is associated with an object
    # All objects must be glimpsed, but the glimpse order should be random
    glimpse_candidates = objects.copy()
    if len(glimpse_candidates) < n_glimpses:
        to_append = glimpse_candidates.copy()
        random.shuffle(to_append)
        to_append
        glimpse_candidates += to_append

    glimpsed_objects = glimpse_candidates[:n_glimpses]
    random.shuffle(glimpsed_objects)
    coords_glimpsed_objects = [POSSIBLE_CENTROIDS[object] for object in glimpsed_objects]

    # Take noisy observations of glimpsed objects
    # vals = [-0.05, .0, 0.05]
    # possible_noise = [(x*nl,y*nl) for (x,y) in product(vals, vals)] + [(x*nl,y*nl) for (x,y) in product([-.1, .1], [.0])] + [(x*nl,y*nl) for (x,y) in product([.0], [-.1, .1])]
    # noises = random.choices(possible_noise, k=n_glimpses)
    noise_x = np.random.normal(loc=0, scale=nl/2, size=n_glimpses)
    noise_y = np.random.normal(loc=0, scale=nl/2, size=n_glimpses)

    while any(abs(noise_x) > nl):
        idx = np.where(abs(noise_x) > nl)[0]
        noise_x[idx] = np.random.normal(loc=0, scale=nl, size=len(idx))
    while any(abs(noise_y) > nl):
        idx = np.where(abs(noise_y) > nl)[0]
        noise_y[idx] = np.random.normal(loc=0, scale=nl, size=len(idx))

    glimpse_coords = [(x+del_x, y+del_y) for ((x,y), (del_x,del_y)) in zip(coords_glimpsed_objects, zip(noise_x, noise_y))]

    return np.array(glimpse_coords), np.array(glimpsed_objects), np.array(coords_glimpsed_objects)


def get_shape_coords(glimpse_coords, objects, noiseless_coords, max_dist):
    """Just as there are 9 possible location in space there are 9 possible
    'shapes'. Each object is randomly assigned one shape.

    returns nx9 array where n is number of glimpses
    """
    seq_len = glimpse_coords.shape[0]
    unique_objects = np.unique(objects)
    unique_object_coords = np.unique(noiseless_coords, axis=0)
    num = len(unique_objects)
    n_shapes = 9
    shape_coords = np.zeros((seq_len, n_shapes))
    # Assign a random shape to each object
    shape_assign = np.random.choice(range(n_shapes), size=num, replace=True)
    shape_map = {object: shape for (object, shape) in zip(unique_objects, shape_assign)}
    # Calculate a weighted 'glimpse label' that indicates the shapes in a glimpse
    eu_dist = euclidean_distances(glimpse_coords, unique_object_coords)
    # max_dist = np.sqrt((noise_level*0.1)**2 + (noise_level*0.1)**2)
    glimpse_idx, object_idx = np.where(eu_dist <= max_dist + EPS)
    shape_idx = [shape_map[obj] for obj in unique_objects[object_idx]]
    # proximity = {idx:np.zeros(9,) for idx in glimpse_idx}
    for gl_idx, obj_idx, sh_idx in zip(glimpse_idx, object_idx, shape_idx):
        if max_dist != 0:
            prox = 1 - (eu_dist[gl_idx, obj_idx]/max_dist)
        else:
            prox  = 1
        # print(f'gl:{gl_idx} obj:{obj_idx} sh:{sh_idx}, prox:{prox}')
        shape_coords[gl_idx, sh_idx] += prox
    # shape_coords[glimpse_idx, shape_idx] = 1 - eu_dist[glimpse_idx, obj_idx]/min_dist
    # make sure each glimpse has at least some shape info
    assert np.all(shape_coords.sum(axis=1) > 0)
    return shape_coords, shape_map


class GlimpsedImage():
    def __init__(self, xy_coords, shape_coords, shape_map, objects, max_dist):
        self.empty_locations = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
        self.filled_locations = set()
        self.lower_bound = 2
        self.upper_bound = 4
        self.count = 0

        self.xy_coords = xy_coords
        self.shape_coords = shape_coords
        self.shape_map = shape_map

        self.n_glimpses = xy_coords.shape[0]
        self.objects = objects
        self.max_dist = max_dist
        self.numerosity = len(np.unique(objects))

        # Special cases where the numerosity can be established from the number
        # of unique candidate locations or distinct shapes
        self.special_case_xy = False
        self.shape_count = len(np.unique(shape_coords.nonzero()[1]))
        if self.shape_count == self.upper_bound:
            self.special_case_shape = True
        else:
            self.special_case_shape = False
        self.lower_bound = max(self.lower_bound, self.shape_count)

    def plot_example(self, pass_count, id='000'):
        # assigned_list = [ass for ass in assignment if ass is not None]
        # pred_num = len(assigned_list)
        uni_objs = np.unique(self.objects)
        # unassigned_list = [centroid for centroid in np.arange(9) if centroid not in objects]
        marker_list = ['o', '^', 's', 'p', 'P', '*', 'D', 'X', 'h']
        winter = plt.get_cmap('winter')
        glimpse_colors = winter([0, 0.25, 0.5, 1])
        fig, ax = plt.subplots(figsize=(5, 5))
        # Grid
        for idx in range(9):
            x, y = CENTROID_ARRAY[idx,0], CENTROID_ARRAY[idx, 1]
            plt.scatter(x, y, color='gray', marker=f'${idx}$')
        lim = (-0.0789015869776648, 1.0789015869776648)
        plt.xlim(lim)
        plt.ylim(lim)
        plt.ylabel('Y')
        plt.xlabel('X')
        # plt.savefig('figures/toy_example1.png')
        # plt.axis('equal')
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        # ax.spines['left'].set_visible(False)

        # Objects
        for obj in uni_objs:
            plt.scatter(CENTROID_ARRAY[obj, 0], CENTROID_ARRAY[obj,1], color='green', marker=marker_list[self.shape_map[obj]], facecolors='none', s=100.0)
        # plt.savefig('figures/toy_example2.png')

        # Glimpses
        xy = self.xy_coords
        for glimpse_idx in range(xy.shape[0]):
            plt.scatter(xy[glimpse_idx,0], xy[glimpse_idx,1], color=glimpse_colors[glimpse_idx], marker=f'${glimpse_idx}$')
        # plt.savefig('figures/toy_example3.png')

        # Noise
        patches = []
        for idx in range(9):
            x, y = CENTROID_ARRAY[idx,0], CENTROID_ARRAY[idx,1]
            patches.append(Circle((x, y), radius=self.max_dist, facecolor='none'))
        p = PatchCollection(patches, alpha=0.1)
        ax.add_collection(p)

        # Assignments
        if pass_count > 0:
            for glimpse_idx, cand_list in enumerate(self.candidates):
                for cand in cand_list:
                    x = [xy[glimpse_idx,0], CENTROID_ARRAY[cand,0]]
                    y = [xy[glimpse_idx,1], CENTROID_ARRAY[cand,1]]
                    if len(cand_list)==1:
                        plt.plot(x, y, color=glimpse_colors[glimpse_idx])
                    else:
                        plt.plot(x, y, '--', color=glimpse_colors[glimpse_idx])
            filled = list(self.filled_locations)
            plt.scatter(CENTROID_ARRAY[filled, 0], CENTROID_ARRAY[filled, 1], marker='o', s=300, facecolors='none', edgecolors='red')
        plt.title(f'count={self.count}')
        # plt.savefig(f'figures/toy_examples/revised_example_{id}.png')

    def process_xy(self):
        """Determine ambiguous glimpses, if any."""
        dist = euclidean_distances(POSSIBLE_CENTROIDS, self.xy_coords)
        dist_th = np.where(dist <= self.max_dist + EPS, dist, -1)
        self.candidates = [np.where(col >= 0)[0] for col in dist_th.T]
        # Count number of unique candidate locations
        unique_cands = set()
        _ = [unique_cands.add(cand) for candlist in self.candidates for cand in candlist]
        if len(unique_cands) == self.lower_bound:
            self.pred_num = self.lower_bound
            self.upper_bound = self.lower_bound
            self.special_case_xy = True
        self.unambiguous_idxs = [idx for idx in range(self.n_glimpses) if len(self.candidates[idx])==1]
        self.ambiguous_idxs = [idx for idx in range(self.n_glimpses) if len(self.candidates[idx])>1]

        for idx in self.unambiguous_idxs:
            loc = self.candidates[idx][0]
            if loc in self.empty_locations:
                self.empty_locations.remove(loc)
                self.filled_locations.add(loc)
        self.count = len(self.filled_locations)
        self.lower_bound = max(self.lower_bound, self.count)

    def check_if_done(self, pass_count):
        """."""
        # self = example
        if self.lower_bound == self.upper_bound:
            self.pred_num = self.lower_bound
            # print('Done because lower_bound reached upper_bound')
            return True
        if pass_count > 0:
            if not self.ambiguous_idxs:  # no ambiguous glimpses
                self.pred_num = self.count
                # print('Done because no ambiguous glimpses')
                return True
            done = True
            to_be_resolved = dict()
            for j in self.ambiguous_idxs:
                for cand in self.candidates[j]:
                    if cand in self.empty_locations:
                        done = False
                        if j not in to_be_resolved.keys():
                            to_be_resolved[j] = (self.candidates[j], [cand])
                        else:
                            to_be_resolved[j][1].extend([cand])
            self.to_be_resolved = to_be_resolved
            if done:
                # print('Done because no abiguity about numerosity')
                self.pred_num = self.count
        else:
            return False
        return done

    # def use_shape_to_resolve_oner(self, idx, cand_list, new_loc):
    #     # What other assigned glimpses have overlapping candidates?
    #     similar = set()
    #     for ua_idx in self.unambigous_glimpses:
    #         if self.candidates(ua_idx) in cand_list:
    #             similar.add(ua_idx)
    #
    #     ambig_shape_idxs = self.shape_coords[idx, :].nonzero()[1]
    #     similar_shape_idxs = self.shape_coords[list(similar), :].nonzero()[1]
    #     # Does this ambiguous glimpse provide evidence of a shape not associated
    #     # with previously toggled locations?
    #     new_shape = False
    #     for shape_idx in ambig_shape_idxs:
    #         if shape_idx not in similar_shape_idxs:
    #             new_shape = True
    #
    #     # Compare hypotheses
    #     # cand_list is at least two and only one of them is a new location
    #     # options are
    #     # 1) no additional object, there are only the object(s) at a previously
    #     # toggled location
    #     # 2) there is an additional object of the same shape at the new location
    #
    #     if not new_shape:
    #         toggled_cands = [cand for cand in cand_list if cand != new_loc[0]]
    #         #
    #
    #         dist = euclidean_distances(self.xy_coords[idx,:].reshape(1, -1), CENTROID_ARRAY[toggled_cands])
    #         prox = 1 - (dist/self.max_dist)
    #
    #     return new_shape

    def use_shape_to_resolve(self, idx, cand_list):
        """
        """
        # idx
        # cand_list
        # self = example
        self.xy_coords.shape
        self.xy_coords[idx,:]
        CENTROID_ARRAY[cand_list]
        dist = euclidean_distances(CENTROID_ARRAY[cand_list], self.xy_coords[idx, :].reshape(1, -1))
        # dist = np.where(dist < self.max_dist, dist, 0)
        prox = (1 - (dist/self.max_dist)).flatten()
        object_locations = []
        for i in range(len(cand_list)):
            if any(np.isclose(self.shape_coords[idx, :], prox[i])):
                object_locations.append(cand_list[i])
        # if any(np.isclose(self.shape_coords[idx, :], prox[1])):
        #     object_locations.append(cand_list[1])
        for comb in combinations(range(len(cand_list)), 2):
            if any(np.isclose(self.shape_coords[idx, :], sum(prox[list(comb)]))):
                object_locations = [cand_list[i] for i in comb]

        return object_locations

    def toggle(self, idx, loc):
        self.candidates[idx] = [loc]
        if loc in self.empty_locations:
            # self.count += 1
            self.empty_locations.remove(loc)
            self.filled_locations.add(loc)
            self.lower_bound = max(self.lower_bound, self.count)
        self.count = len(self.filled_locations)


def generate_one_example(noise_level):
    """Synthesize a single sequence and determine numerosity."""
    # Synthesize glimpses - paired observations of xy and shape coordinates
    max_dist = np.sqrt((noise_level*0.1)**2 + (noise_level*0.1)**2)
    num = random.randrange(2, 5)
    xy_coords, objects, noiseless_coords = get_xy_coords(num, noise_level)
    shape_coords, shape_map = get_shape_coords(xy_coords, objects, noiseless_coords, max_dist)
    # print(shape_coords)

    # Initialize records
    example = GlimpsedImage(xy_coords, shape_coords, shape_map, objects, max_dist)
    pass_count = 0
    done = example.check_if_done(pass_count)

    # First pass
    if not done:
        example.process_xy()
        pass_count += 1
        done = example.check_if_done(pass_count)

    while not done and pass_count < 6:
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
        print('DID NOT SOLVE')
        example.pred_num = example.lower_bound
        example.plot_example(pass_count)
    unique_objects = set(example.objects)
    filled_locations = [1 if i in unique_objects else 0 for i in range(9)]
    example_dict = {'xy': xy_coords, 'shape': shape_coords, 'numerosity': num,
                    'predicted num': example.pred_num, 'count': example.count,
                    'locations': filled_locations,
                    'pass count': pass_count, 'unresolved ambiguity': not done,
                    'special xy': example.special_case_xy,
                    'special shape': example.special_case_shape,
                    'lower bound': example.lower_bound,
                    'upper bound':example.upper_bound}
    return example_dict


def generate_dataset(noise_level, n_examples):
    """Fill data frame with toy examples."""

    data = [generate_one_example(noise_level) for _ in range(n_examples)]
    print('Putting data into DataFrame')
    df = pd.DataFrame(data)
    # df['pass count'].hist()
    # df[df['unresolved ambiguity'] == True]
    # df[df['numerosity'] != df['predicted num']]
    # df['pass count'].max()
    return df
