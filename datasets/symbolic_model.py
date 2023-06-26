import numpy as np
from itertools import product, combinations
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
from sklearn.metrics.pairwise import euclidean_distances

class GlimpsedImage():
    def __init__(self, xy_coords, shape_coords, shape_map, shape_hist, objects, max_dist, num_range):
        self.eps = 1e-7
        self.grid_size = 6
        self.grid = np.linspace(0.1, 0.9, self.grid_size, dtype=np.float32)
        self.possible_centroids = [(x, y) for (x, y) in product(self.grid, self.grid)]
        self.grid_size = len(self.possible_centroids)
        self.centroid_array = np.array(self.possible_centroids)
        # self.empty_locations = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
        self.empty_locations = set(range(self.grid_size))
        self.filled_locations = set()
        self.lower_bound, self.upper_bound = num_range
        self.count = 0

        # Remove glimpses that have empty shape feature vectors
        n_glimpses = xy_coords.shape[0]
        # self.xy_coords = np.array([xy_coords[i] for i in range(n_glimpses) if sum(shape_coords[i]>0)])
        # self.shape_coords = np.array([coords for coords in shape_coords if sum(coords)>0])

        self.xy_coords = xy_coords
        self.shape_coords = shape_coords

        # self.xy_coords = xy_coords
        # self.shape_coords = shape_coords
        self.shape_map = shape_map
        self.min_shape = np.argmin(shape_hist)
        self.min_num = shape_hist[self.min_shape]

        self.n_glimpses = self.xy_coords.shape[0]
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
            x, y = self.centroid_array[idx,0], self.centroid_array[idx, 1]
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
            plt.scatter(self.centroid_array[obj, 0], self.centroid_array[obj,1], color='green', marker=marker_list[self.shape_map[obj]], facecolors='none', s=100.0)
        # plt.savefig('figures/toy_example2.png')

        # Glimpses
        xy = self.xy_coords
        for glimpse_idx in range(xy.shape[0]):
            plt.scatter(xy[glimpse_idx,0], xy[glimpse_idx,1], color=glimpse_colors[glimpse_idx], marker=f'${glimpse_idx}$')
        # plt.savefig('figures/toy_example3.png')

        # Noise
        patches = []
        for idx in range(9):
            x, y = self.centroid_array[idx,0], self.centroid_array[idx,1]
            patches.append(Circle((x, y), radius=self.max_dist, facecolor='none'))
        p = PatchCollection(patches, alpha=0.1)
        ax.add_collection(p)

        # Assignments
        if pass_count > 0:
            for glimpse_idx, cand_list in enumerate(self.candidates):
                for cand in cand_list:
                    x = [xy[glimpse_idx,0], self.centroid_array[cand,0]]
                    y = [xy[glimpse_idx,1], self.centroid_array[cand,1]]
                    if len(cand_list)==1:
                        plt.plot(x, y, color=glimpse_colors[glimpse_idx])
                    else:
                        plt.plot(x, y, '--', color=glimpse_colors[glimpse_idx])
            filled = list(self.filled_locations)
            plt.scatter(self.centroid_array[filled, 0], self.centroid_array[filled, 1], marker='o', s=300, facecolors='none', edgecolors='red')
        plt.title(f'count={self.count}')
        # plt.savefig(f'figures/toy_examples/revised_example_{id}.png')

    def process_xy(self):
        """Determine ambiguous glimpses, if any."""
        dist = euclidean_distances(self.possible_centroids, self.xy_coords)
        dist_th = np.where(dist <= self.max_dist + self.eps, dist, -1)
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
        """Return True if any of a number of termination conditions are met."""
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
        """Use shape vector to determine which candidate locations hold objects.
        """
        # idx
        # cand_list
        # self = example
        # self.xy_coords.shape
        # self.xy_coords[idx,:]
        # self.centroid_array[cand_list]
        dist = euclidean_distances(self.centroid_array[cand_list], self.xy_coords[idx, :].reshape(1, -1))
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