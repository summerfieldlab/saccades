import os
from datetime import datetime
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

def colorbar(mappable):
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar

class Timer():
    """A class for timing code execution.
    Copied from HRS.
    """
    def __init__(self):
        self.start = datetime.now()
        self.end = None
        self.elapsed_time = None

    def stop_timer(self):
        self.end = datetime.now()
        self.elapsed_time = self.end - self.start
        print('Execution time: {}'.format(self.elapsed_time))
        
        
def convert_to_float_array(string):
    """Convert string to numeric list.
    
    For use when variable lengthed arrays were saved to python readable format
    from MATLAB.
    
    example:
    df.fixation_posX = df.fixation_posX.apply(convert_to_float_array)
    """
    string_list = string.split()
    float_list = [float(x) for x in string_list]
    return float_list

def convert_to_int_array(string):
    """Convert string to numeric list.
    
    For use when variable lengthed arrays were saved to python readable format
    from MATLAB.
    
    example:
    df.fixation_posX = df.fixation_posX.apply(convert_to_float_array)
    """
    string_list = string.split()
    float_list = [int(x) for x in string_list]
    return float_list

def pixel_to_scaled_x(coords):
    screen_width, screen_height = 1280, 1024
    centerX = screen_width//2
    image_width = screen_height-280 # 744x744 square
    upper_left_x = centerX - image_width//2

    scaled = [(x-upper_left_x)/image_width for x in coords]
    # If fixation is off image, map to the nearest edge of the image
    scaled = [0 if coord < 0 else coord for coord in scaled]
    scaled = [1 if coord > 1 else coord for coord in scaled]
    return scaled

def pixel_to_scaled_y(coords):
    screen_width, screen_height = 1280, 1024
    centerY = screen_height//2
    image_height = screen_height-280 # 744x744 # this is correct because height and width are equal
    upper_left_y = centerY - image_height//2
    
    scaled = [(y-upper_left_y)/image_height for y in coords]
    # If fixation is off image, map to the nearest edge of the image
    scaled = [0 if coord < 0 else coord for coord in scaled]
    scaled = [1 if coord > 1 else coord for coord in scaled]
    return scaled

def gkern(l=5, sig=1.):
        """\
        creates gaussian kernel with side length `l` and a sigma of `sig`
        """
        ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
        gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
        kernel = np.outer(gauss, gauss)
        return kernel / np.sum(kernel)
    
def map_jess_to_tim_blocks(fname):
# %MAP_JESS_TO_TIM_BLOCKS convert from jess' to tim's naming convention.
# %
# %Confusingly, 'block' is used to refer to at least two pieces of the
# %experimental design. Here, it refers to the stimuli sets from which the
# %stimuli for each participant were selected. Elsewhere, the same word is
# %used to indicate what might also be called a run of the experiment.
# %
# %Jess made 11 blocks of image(0-10), Tim used block 0 as 
# % practice, and blocks 1,2,3,4,5,7,8,9 as the 8 blocks for the actual 
# % experiment, and the folder names 6,7,8 correspond to 7,8,9 of the 
# % metadata files. Tim left out 6 and 10 because there was an error loading 
# % these two image folders on the lab computer, and Tim thought 8 blocks 
# % would be sufficient. (What happened here was: Tim didn't use two of the 
# % sets of stimuli I generated. He then renamed the names of the directories 
# % in which the images live AND the image metadata files so that there were 
# % no missing numbers. The folder_name column of the behavioural data
# % corresponds to the renamed folders. 
# % The 'image_filename' and 'block' fields of the image metadata correspond 
# % to Jess' original block labeling. Additionally, Tim renamed the image 
# % files to prepend all single digit image numbers (except 0) with a zero 
# % (image_0.png, image_01.png, etc.).  )
# %
# % This script takes a list of image file names in my original naming scheme
# % and converst it to a list of image file names in Tim's renamed scheme.
# %        
    p = fname.split('_')
    b = p[1].split('/')
    block = b[0]
    im_no = p[2].split('.')
    im_no = im_no[0]
    
    if block == '7':
        block = '6'
    elif block == '8':
        block = '7'
    elif block == '9':
        block = '8'

    new_middle = os.path.join(block, b[1])
    if int(im_no) > 0 and int(im_no) < 10:
        # Prepend 0 to the image number only for 1-9
        fname = f'{p[0]}_{new_middle}_0{p[2]}'
    else:
        fname = f'{p[0]}_{new_middle}_{p[2]}'
    return fname


# Augmentation functions
def transpose(locations_list):
    """When grid=6, list of 36 corresponds to 6x6 grid to be transposed.
    Returns a list of length 36 after having transposed the grid."""
    grid_sz = int(np.sqrt(len(locations_list)))
    grid = np.reshape(locations_list, (grid_sz, grid_sz))
    transposed = list(grid.T.flatten())
    return transposed

def mirror_gridx(locations_list):
    # if isinstance(locations_list, list):
    grid_sz = int(np.sqrt(len(locations_list)))
    grid = np.reshape(locations_list, (grid_sz, grid_sz))
    # mirrorx = list(grid[:, ::-1].flatten())
    mirrorx = list(grid[::-1, :].flatten())
    # else:
    #     grid = locations_list
    #     mirrorx = grid[:, ::-1]
    return mirrorx

def mirror_gridy(locations_list):
    """Mirror on x axis.
    Used during augmentation to transform the binary lists indicating the 
    locations of targets and distractors. The same function is also used to
    apply the same transformation to the image itself so that (augmented) 
    fixations can be overlayed on a corresponding image. 
    """
    # if isinstance(locations_list, list):
    grid_sz = int(np.sqrt(len(locations_list)))
    grid = np.reshape(locations_list, (grid_sz, grid_sz))
    # mirrory = list(grid[::-1, :].flatten())
    mirrory = list(grid[:, ::-1].flatten())
    # else:
    #     grid = locations_list
    #     mirrory = grid[::-1, :]
    return mirrory

def mirror(coordinates, length=1, pixels=41):
    """For scaled coordinates, length=1."""

    # mirrored1 = [mid - (coord - mid) for coord in coordinates]
    mirrored2 = [abs(coord - (length - (length/pixels))) for coord in coordinates]
    return mirrored2


def apply_image_transform(image, transform):
    if transform==1: # transpose
        image = np.transpose(image)
    if transform==2: # mirror x
        image = image[:, ::-1]
    if transform==3: # mirror y
        image = image[::-1, :]
    if transform==4: # mirror y
        image = image[::-1, ::-1]
    if transform==5: # transpose + mirror x
        image = np.transpose(image)
        image = image[:, ::-1]
    if transform==6: # transpose + mirror y
        image = np.transpose(image)
        image = image[::-1, :]
    if transform==7: # transpose + mirror y
        image = np.transpose(image)
        image = image[::-1, ::-1]
    return image

def apply_coords_transform(coords, transform, length=[1,1], pixels=[41, 41]):
    transformed = coords.copy()
    if transform==1: # transpose
        transformed[:, 0] = coords[:, 1].copy()
        transformed[:, 1] = coords[:, 0].copy()
    if transform==2: # mirror x
        # x, y?
        transformed[:, 0] = mirror(transformed[:, 0], length=length[0], pixels=pixels[0])
    if transform==3: # mirror y
        transformed[:, 1] = mirror(transformed[:, 1], length=length[1], pixels=pixels[1])
    if transform==4: # mirror x and y
        transformed[:, 0] = mirror(transformed[:, 0], length=length[0], pixels=pixels[0])
        transformed[:, 1] = mirror(transformed[:, 1], length=length[1], pixels=pixels[1])
    if transform==5: # transpose + mirror x
        transformed[:, 0] = coords[:, 1].copy()
        transformed[:, 1] = coords[:, 0].copy()
        transformed[:, 0] = mirror(transformed[:, 0], length=length[0], pixels=pixels[0])
    if transform==6: # transpose + mirror y
        transformed[:, 0] = coords[:, 1].copy()
        transformed[:, 1] = coords[:, 0].copy()
        transformed[:, 1] = mirror(transformed[:, 1], length=length[1], pixels=pixels[1])
    if transform==7: # transpose + mirror y
        transformed[:, 0] = coords[:, 1].copy()
        transformed[:, 1] = coords[:, 0].copy()
        transformed[:, 0] = mirror(transformed[:, 0], length=length[0], pixels=pixels[0])
        transformed[:, 1] = mirror(transformed[:, 1], length=length[1], pixels=pixels[1])
    return transformed

def apply_slot_transform(slot_list, transform):
    if transform==1: # transpose
        slot_list = transpose(slot_list)
    if transform==2: # mirror x
        slot_list = mirror_gridx(slot_list)
    if transform==3: # mirror y
        slot_list = mirror_gridy(slot_list)
    if transform==4: # mirror y
        slot_list = mirror_gridx(slot_list)
        slot_list = mirror_gridy(slot_list)
    if transform==5: # transpose + mirror x
        slot_list = transpose(slot_list)
        slot_list = mirror_gridx(slot_list)
    if transform==6: # transpose + mirror y
        slot_list = transpose(slot_list)
        slot_list = mirror_gridy(slot_list)
    if transform==7: # transpose + mirror y
        slot_list = transpose(slot_list)
        slot_list = mirror_gridx(slot_list)
        slot_list = mirror_gridy(slot_list)
    return slot_list

