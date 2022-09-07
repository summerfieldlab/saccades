"""Add pseudo images containing numeric characters to toy dataset."""
import os
import numpy as np
import pandas as pd
from itertools import product
from matplotlib import pyplot as plt
import argparse
import toy_model_data as toy

GRID = [0.2, 0.5, 0.8]
PIXEL_X = [col*5 +1 for col in [0, 1, 2]]   #[1, 6, 11] # col *5 + 1
PIXEL_Y = [row*7 +1 for row in [0, 1, 2]]
POSSIBLE_CENTROIDS = [(x, y) for (x, y) in product(GRID, GRID)]
PIXEL_TOPLEFT = [(x, y) for (x, y) in product(PIXEL_X, PIXEL_Y)]
MAP_SCALE_PIXEL = {scaled:pixel for scaled,pixel in zip(POSSIBLE_CENTROIDS, PIXEL_TOPLEFT)}
CENTROID_ARRAY = np.array(POSSIBLE_CENTROIDS)
zero = np.array([[1, 1, 1],
                 [1, 0, 1],
                 [1, 0, 1],
                 [1, 0, 1],
                 [1, 1, 1]])
one = np.array([[0, 1, 0],
                [0, 1, 0],
                [0, 1, 0],
                [0, 1, 0],
                [0, 1, 0]])
two = np.array([[1, 1, 1],
                [0, 0, 1],
                [1, 1, 1],
                [1, 0, 0],
                [1, 1, 1]])
three = np.array([[1, 1, 1],
                  [0, 0, 1],
                  [1, 1, 1],
                  [0, 0, 1],
                  [1, 1, 1]])
four = np.array([[1, 0, 1],
                 [1, 0, 1],
                 [1, 1, 1],
                 [0, 0, 1],
                 [0, 0, 1]])
five = np.array([[1, 1, 1],
                 [1, 0, 0],
                 [1, 1, 1],
                 [0, 0, 1],
                 [1, 1, 1]])
six = np.array([[1, 1, 1],
                [1, 0, 0],
                [1, 1, 1],
                [1, 0, 1],
                [1, 1, 1]])
seven = np.array([[1, 1, 1],
                  [0, 0, 1],
                  [0, 0, 1],
                  [0, 0, 1],
                  [0, 0, 1]])
eight = np.array([[1, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
nine = np.array([[1, 1, 1],
                 [1, 0, 1],
                 [1, 1, 1],
                 [0, 0, 1],
                 [0, 0, 1]])

shape_holder = np.zeros((4, 4))
chars = [zero, one, two, three, four, five, six, seven, eight, nine]
pixel_counts = [np.sum(char) for char in chars]
np.mean(pixel_counts[0:5])
np.mean(pixel_counts[5:10])
# for char in chars:
#     # plt.matshow(char)
#     print(np.sum(char))


def get_overlap(char_set):
    overlaps = []
    for i, char1 in enumerate(char_set):
        for j, char2 in enumerate(char_set[1:]):
            if j < i:
                continue
            overlaps.append(np.sum(char1 == char2))
    avg = np.mean(overlaps)
    return avg


def add_char_glimpses(data, glim_wid=6):
    """Synthesize and glimpse small image of pixel corners.

    The image is a 12 by 12 grid (+ a 2 pixel border). The 9 possible object
    locations from the toy dataset generator thus correspond to 4x4 regions
    of the image. Here my shapes are the 4 possible 3-pixel corners:
    1 0     0 1     1 1     1 1
    1 1     1 1     1 0     0 1

    The number and location of these shapes as well as the location of the
    the glimpses of these shapes are determined by the toy_model_data
    generator used for previous experiments. Thus we retain the knowledge of
    the difficulty of each 'image', or how much it requires the integration of
    both sources of glimpse information (glimpse content and glimpse location,
    what and where) to determine the numerosity.

    The synthesized image as well as the glimpse sequences are appended as new
    columns to the existing data frame. These new columns can then be used as
    input instead of the existing xy and shape column, used in previous
    experiments.
    """
    char_width = 3
    char_height = 5
    pixel_height = 7*3
    pixel_width = 5*3

    i=0
    data['char glimpse coords'] = None
    data['char glimpse pixels'] = None
    data['char image'] = None
    data['char overlap'] = None
    for i in range(len(data)):
        if not i % 10:
            print(f'Synthesizing image {i}', end='\r')
        row = data.iloc[i]
        # Coordinates in 1x1 space
        object_xy_coords = CENTROID_ARRAY[np.where(row.locations)[0]]
        object_pixel_coords = [MAP_SCALE_PIXEL[tuple(xy)] for xy in object_xy_coords]
        # Shape indices for the objects
        object_shapes = [row['shape_map'][obj] for obj in np.where(row.locations)[0]]
        char_set = [chars[idx] for idx in object_shapes]

        char_similarity = get_overlap(char_set)
        data.at[i, 'char overlap'] = char_similarity
        # Insert the specified shapes into the image at the specified locations
        image = np.zeros((pixel_height, pixel_width))
        # plt.matshow(image, origin='lower')
        # plt.plot([3.5, 3.5], [0, 11], color='cyan')
        # plt.plot([7.5, 7.5], [0, 11], color='cyan')
        # plt.plot([0, 11], [3.5, 3.5], color='cyan')
        # plt.plot([0, 11], [7.5, 7.5], color='cyan')
        for shape_idx, (x,y) in zip(object_shapes, object_pixel_coords):
                # print(f'{shape_idx} {x} {y}')
                # yy = pixel_height - y - 5
                # xx = pixel_width - x - 3
                # image[yy:yy+char_height:, xx:xx+char_width] = chars[shape_idx]
                image[y:y+char_height:, x:x+char_width] = chars[shape_idx]

        # Convert glimpse coordinates to pixel coordinates
        scaled_glimpse_coords = row.xy
        scaled_glimpse_coords[:,0] = scaled_glimpse_coords[:,0]*pixel_width
        scaled_glimpse_coords[:,1] = scaled_glimpse_coords[:,1]*pixel_height
        glimpse_coords = np.round(row.xy).astype(int)

        # plt.matshow(image, origin='upper')
        # plt.scatter(glimpse_coords[:,0], glimpse_coords[:,1], color='red')
        # plt.scatter(scaled_glimpse_coords[:,0], scaled_glimpse_coords[:,1], color='green')

        # Add border of 2 pixels so all gimpses are the same size
        # glim_wid = 6
        half_glim = glim_wid//2
        border = half_glim
        image_wbord = np.zeros((pixel_height+glim_wid, pixel_width+glim_wid))
        image_wbord[half_glim:-half_glim,half_glim:-half_glim] = image
        # plt.matshow(image_wbord, origin='upper')
        glimpse_coords += half_glim
        # Extract glimpse pixels
        glimpse_pixels = [image_wbord[y-half_glim:y+half_glim, x-half_glim:x+half_glim].flatten() for x,y in glimpse_coords]
        glimpse_pixels[0].shape
        # Store glimpse data and image in the original dataframe
        data.at[i,'char glimpse pixels'] = glimpse_pixels
        data.at[i,'char glimpse coords'] = glimpse_coords
        data.at[i,'char image'] = image_wbord

        glimpse_pixels_to_plot = [image_wbord[y-half_glim:y+half_glim, x-half_glim:x+half_glim]for x,y in glimpse_coords]

        # bounding = [plt.Rectangle((x-half_glim-0.5, y-half_glim-0.5), glim_wid, glim_wid, fc='none',ec="red") for x,y in glimpse_coords]
        # plt.matshow(glimpse_pixels_to_plot[0], origin='upper')
        # plt.matshow(glimpse_pixels_to_plot[1], origin='upper')
        # plt.matshow(glimpse_pixels_to_plot[2], origin='upper')
        # plt.matshow(glimpse_pixels_to_plot[3], origin='upper')
        # plt.matshow(image_wbord, origin='upper')
        # plt.scatter(glimpse_coords[:,0]-0.5, glimpse_coords[:,1]-0.5, color='red')
        # for box in bounding:
        #     plt.gca().add_patch(box)
    return data


def add_chars(fname):
    if os.path.exists(fname):
        print(f'Loading saved dataset {fname}')
        data = pd.read_pickle(fname)
    else:
        print('Dataset does not exist')
        exit()
    data = add_char_glimpses(data)
    print(f'Saving {fname} with numeric character images')
    data.to_pickle(fname)
    return data


def main():
    parser = argparse.ArgumentParser(description='PyTorch network settings')
    parser.add_argument('--min_pass', type=int, default=0)
    parser.add_argument('--max_pass', type=int, default=6)
    parser.add_argument('--min_num', type=int, default=2)
    parser.add_argument('--max_num', type=int, default=7)
    parser.add_argument('--shapes', type=list, default=[0, 1, 2, 3, 5, 6, 7, 8])
    parser.add_argument('--noise_level', type=float, default=1.7)
    parser.add_argument('--size', type=int, default=100)
    parser.add_argument('--n_shapes', type=int, default=10, help='How many shapes to the relevant training and test sets span?')
    parser.add_argument('--glimpse_wid', type=int, default=6, help='How many pixels wide and tall should glimpses be?')
    parser.add_argument('--same', action='store_true', default=False)
    conf = parser.parse_args()
    conf.shapes = [int(i) for i in conf.shapes]
    same = 'same' if conf.same else ''
    fname_gw = f'toysets/toy_dataset_num{conf.min_num}-{conf.max_num}_nl-{conf.noise_level}_diff{conf.min_pass}-{conf.max_pass}_{conf.shapes}{same}_gw{conf.glimpse_wid}_{conf.size}.pkl'
    fname = f'toysets/toy_dataset_num{conf.min_num}-{conf.max_num}_nl-{conf.noise_level}_diff{conf.min_pass}-{conf.max_pass}_{conf.shapes}{same}_{conf.size}.pkl'
    if os.path.exists(fname):
        print(f'Loading saved dataset {fname}')
        data = pd.read_pickle(fname)
    else:
        print('Generating new dataset')
        data = toy.generate_dataset(conf.noise_level, conf.size, (conf.min_pass, conf.max_pass), (conf.min_num, conf.max_num), conf.shapes, conf.n_shapes, conf.same)
        data.to_pickle(fname)

    data = add_char_glimpses(data, conf.glimpse_wid)
    # fname = f'toysets/toy_dataset_nl-{noise_level}_diff-{min_pass_count}-{max_pass_count}_{shapes_set}_{size}_tetris.pkl'
    print(f'Saving {fname_gw}')
    data.to_pickle(fname_gw)

if __name__ == '__main__':
    main()
