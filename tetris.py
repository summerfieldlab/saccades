"""Synthesize and glimpse small 'images' of tetris-like shapes based on toy dataset.

It continues to be the bane of my existance that you index into matrices with
the origin in the upper left, but my x,y coordinates assume origin in bottom
left. How to reconcile? When plotting the image, always plot img.T, origin='lower'

"""
import os
import numpy as np
import pandas as pd
from itertools import product
from matplotlib import pyplot as plt
import toy_model_data as toy

GRID = [0.2, 0.5, 0.8]
POSSIBLE_CENTROIDS = [(x, y) for (x, y) in product(GRID, GRID)]
CENTROID_ARRAY = np.array(POSSIBLE_CENTROIDS)
shape1 = np.array([[1, 1],
                   [0, 1]])
shape2 = np.array([[1, 1],
                   [1, 0]])
shape3 = np.array([[1, 0],
                   [1, 1]])
shape4 = np.array([[0, 1],
                   [1, 1]])
shapes = [shape1, shape2, shape3, shape4]

def add_tetris_glimpses(data):
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
    pixel_width = 4*3
    data['tetris glimpse coords'] = None
    data['tetris glimpse pixels'] = None
    data['tetris image'] = None
    for i in range(len(data)):
        if not i % 10:
            print(f'Synthesizing image {i}', end='\r')
        row = data.iloc[i]
        # Coordinates in 1x1 space
        object_xy_coords = CENTROID_ARRAY[np.where(row.locations)[0]]
        # Shape indices for the objects
        object_shapes = [row['shape_map'][obj] for obj in np.where(row.locations)[0]]
        # We want to put the corners in the centre 4 pixels of each 4x4 section
        # of the image. For this we use a fixed mapping that returns the upper
        # left pixel of that 2x2 centre where we want to insert the shape.
        coord_map12 = {0.2: 1, 0.5:5, 0.8:9}
        def map_coords(x):
            return coord_map12[x]
        map_coords_vec =  np.vectorize(map_coords)
        object_pixel_coords = map_coords_vec(object_xy_coords)

        # Insert the specified shapes into the image at the specified locations
        image = np.zeros((pixel_width, pixel_width))
        for shape_idx, (x,y) in zip(object_shapes, object_pixel_coords):
                # print(f'{shape_idx} {x} {y}')
                image[x:x+2:, y:y+2] = shapes[shape_idx]

        # Convert glimpse coordinates to pixel coordinates
        scaled_glimpse_coords = row.xy*pixel_width
        glimpse_coords = np.round(row.xy*pixel_width).astype(int)

        # plt.matshow(image.T, origin='lower')
        # plt.scatter(glimpse_coords[:,0], glimpse_coords[:,1], color='red')
        # plt.scatter(scaled_glimpse_coords[:,0], scaled_glimpse_coords[:,1], color='green')

        # Add border of 2 pixels so all gimpses are the same size
        image_wbord = np.zeros((pixel_width+4, pixel_width+4))
        image_wbord[2:-2,2:-2] = image
        glimpse_coords += 2
        # Extract glimpse pixels
        glimpse_pixels = [image_wbord[x-2:x+2, y-2:y+2].flatten() for x,y in glimpse_coords]
        # Store glimpse data and image in the original dataframe
        data.at[i,'tetris glimpse pixels'] = glimpse_pixels
        data.at[i,'tetris glimpse coords'] = glimpse_coords
        data.at[i,'tetris image'] = image_wbord
        # bounding = [plt.Rectangle((x-2,y-2), 4, 4, fc='none',ec="red") for x,y in glimpse_coords]
        # plt.matshow(glimpse_pixels[0].T, origin='lower')
        # plt.matshow(glimpse_pixels[1].T, origin='lower')
        # plt.matshow(glimpse_pixels[2].T, origin='lower')
        # plt.matshow(glimpse_pixels[3].T, origin='lower')
        # plt.matshow(image_wbord.T, origin='lower')
        # plt.scatter(glimpse_coords[:,0], glimpse_coords[:,1], color='red')
        # for box in bounding:
        #     plt.gca().add_patch(box)
    return data

def main():

    noise_level = 1.7
    size = 5000
    min_pass_count = 0
    max_pass_count = 6
    shapes_set = [3]
    fname = f'toysets/toy_dataset_nl-{noise_level}_diff-{min_pass_count}-{max_pass_count}_{shapes_set}_{size}.pkl'
    if os.path.exists(fname):
        print(f'Loading saved dataset {fname}')
        data = pd.read_pickle(fname)
    else:
        print('Generating new dataset')
        data = toy.generate_dataset(noise_level, size, min_pass_count, max_pass_count, shapes_set)
        data.to_pickle(fname)

    data = add_tetris_glimpses(data)
    fname = f'toysets/toy_dataset_nl-{noise_level}_diff-{min_pass_count}-{max_pass_count}_{shapes_set}_{size}_tetris.pkl'
    print(f'Saving {fname}')
    data.to_pickle(fname)

if __name__ == '__main__':
    main()

# fname = f'toysets/toy_dataset_nl-{noise_level}_diff-{min_pass_count}-{max_pass_count}_{shapes_set}_{size}_tetris.pkl'
# test = pd.read_pickle(fname)




#
