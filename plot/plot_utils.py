"""Utility functions for loading and processing data to prepare for plotting."""
import os
from datetime import datetime
from itertools import product
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

HOME = '/home/jessica/Dropbox/saccades/rnn_tests/'
EYE_HOME1 = '/home/jessica/Dropbox/saccades/eye_tracking/tim/'
EYE_HOME2 = '/home/jessica/Dropbox/saccades/eye_tracking/julian/'
RES_DIR = HOME + 'results/logpolar'
LETTERS_DIR = HOME + 'results/toy/letters'

def get_results(nreps, model_desc, data_desc, train_desc):
    """Load saved model performance."""
    results_list = []
    for rep in range(nreps):
        # print(f'Rep {rep}')
        train_desc_rep = train_desc + f'_rep{rep}'
        tr_file = f'{RES_DIR}/results_{model_desc}_{data_desc}_{train_desc_rep}.pkl'
        if not os.path.exists(tr_file):
            tr_file = f'{LETTERS_DIR}/toy_results_{model_desc}_{data_desc}_{train_desc_rep}.pkl'
        try:
            data = pd.read_pickle(tr_file)
#             print(f'Loading {tr_file}')
            # data = data[data['dataset'] == 'train']
            try:
                data = data.drop(columns=['shape loss', 'rnn iterations'])
            except:
                data = data.drop(columns=['shape loss'])
            data['repetition'] = rep
            results_list.append(data)
        except FileNotFoundError:
            pass
            # print(f'Rep {rep} not found {tr_file}') # , end='\r   ') 
        
    if len(results_list) > 1:
        results = pd.concat(results_list, ignore_index=True)
    elif len(results_list) == 0:
        # print('No matching results found')
        results = []
    else:
        results = results_list[0]
    print(f'{len(results_list)} hits for {model_desc}')
    return results

def get_test_results(nreps, model_desc, data_desc, train_desc, pass_count=False):
    """Load detailed per-trial performance of all model test sets."""
    shapes_map = {'[0, 1, 2, 3, 4]': 'same as training', '[5, 6, 7, 8, 9]': 'new'}
    lums_map = {'[0.0, 0.5, 1.0]': 'same as training', '[0.1, 0.3, 0.7, 0.9]': 'new'}
    results = []
    for rep in range(0, nreps):
        # print(f'Rep {rep}')
        train_desc_rep = train_desc + f'_rep{rep}'
        # Look in scaled dir first
        te_file = f'{RES_DIR}/test_results_{model_desc}_{data_desc}_{train_desc_rep}.pkl'
        if not os.path.exists(te_file):
            te_file = f'{LETTERS_DIR}/test_results_{model_desc}_{data_desc}_{train_desc_rep}.pkl'
#         print(f'Loading {te_file}')
        try:
            data = pd.read_pickle(te_file)
        except FileNotFoundError:
            # print(f'Missing {te_file}')
            continue
        if 'test_shapes' in data.columns:
            data = data.drop(columns=['test_shapes', 'train_shapes'])
#         data = data.rename(columns={"test shapes": "Shapes", "test lums": "Luminance"})
        data['Test Shapes'] = data['test shapes'].apply(lambda x: shapes_map[x])
        data['Test Luminance'] = data['test lums'].apply(lambda x: lums_map[x])
        
        if pass_count:
            data = data[data['Test Shapes'] == 'new']           
            data = data[data['Test Luminance'] == 'new']
            data = data.rename(columns={'pass count': 'Integration Score'})
            mean_data = data.groupby(['epoch', 'Integration Score', 'repetition']).mean()
        else:
            mean_data = data.groupby(['epoch', 'Test Shapes', 'Test Luminance', 'repetition']).mean()
        results.append(mean_data)
    if len(results > 1):
        te_data = pd.concat(results)
    else:
        te_data = results[0]
    te_data = te_data.reset_index()
    return te_data

def get_confusion(nreps, model_desc, data_desc, train_desc, checkpoint=None):
    """Load training perofrmance confusion matrices saved at checkpoints."""
    start_rep = 50
    confs = []
    for rep in range(start_rep, start_rep + nreps):
        # print(f'Rep {rep}')
        train_desc_rep = train_desc + f'_rep{rep}'

        if checkpoint is not None:
            conf_file = f'{RES_DIR}/confusion_at_{checkpoint}_{model_desc}_{data_desc}_{train_desc_rep}.npy'
        else:
            conf_file = f'{RES_DIR}/confusion_{model_desc}_{data_desc}_{train_desc_rep}.npy'
        try:
            conf = np.load(conf_file, allow_pickle=True)
            confs.append(conf)
        except FileNotFoundError:
            continue
            # print(f'Misssing {conf_file}')
    if not confs:
        print('No matching files found')
        return
    else:
        confs = np.array(confs)
    return confs

def relabel_dataset(dataset, shapes, lums):
    """Update dataset names to reflect condition rather than the specfic items."""
    train_shapes = str(['B', 'C', 'D', 'E'])
    train_lums = str([0.1, 0.4, 0.7])
    if dataset == 'ood':
        if shapes != train_shapes and lums != train_lums:
            dataset = 'OOD Both'
        elif shapes != train_shapes:
            dataset = 'OOD Shape'
        elif lums != train_lums:
            dataset = 'OOD Luminance'
    elif dataset == 'train': dataset = 'Train'
    elif dataset == 'validation': dataset = 'Validation'
    
    return dataset

def load_human_beh_exp2():
    """Load the human behaviour from the second eye tracking experiment."""
    participants = np.arange(1, 27)
    missing = [1, 14]
    par_to_load = np.setdiff1d(participants, missing)
    beh_data_list = []
    for sub in par_to_load:
        csv = f'{EYE_HOME2}Eyetracking_participant_data/P{sub}_data/subject_{sub:02}.csv'
        beh_data_list.append(pd.read_csv(csv))
    beh_data = pd.concat(beh_data_list)
    beh_data = beh_data.assign(View=beh_data.View.map({0: "Free", 1: "Fixed"}))
    beh_data = beh_data.assign(Task=beh_data.Task.map({0: "Ignore \nDistractors", 1: "Simple \nCounting"}))
    beh_data['accuracy count'] = beh_data['Accuracy']*100
    # Drop aborted trials
    beh_data = beh_data.drop(beh_data.query('Aborted==1').index)
    return beh_data

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

# def pixel_to_scaled_x(coords):
#     """Transform reference frames.

#     Transform from pixel coordinates in the reference from of the monitor to 
#     scaled coordinates between 0 and 1 in the reference from of the stimulus 
#     image.
#     """
#     screen_width, screen_height = 1280, 1024
#     centerX = screen_width//2
#     image_width = screen_height-280 # 744x744 square
#     upper_left_x = centerX - image_width//2

#     scaled = [(x-upper_left_x)/image_width for x in coords]
#     return scaled

# def pixel_to_scaled_y(coords):
#     """Transform reference frames.

#     Transform from pixel coordinates in the reference from of the monitor to 
#     scaled coordinates between 0 and 1 in the reference from of the stimulus 
#     image.
#     """
#     screen_width, screen_height = 1280, 1024
#     centerY = screen_height//2
#     image_height = screen_height-280 # 744x744
#     upper_left_y = centerY - image_height//2
    
#     scaled = [(y-upper_left_y)/image_height for y in coords]
    
#     return scaled

def calculate_coords(locA, locB, locC=None):
    """Get coordinates of all items."""
    if locC is not None:
        locs = np.concatenate([locA, locB, locC])
    else:
        locs = np.concatenate([locA, locB])
    coords = index_to_coord(locs)
    return coords

def calculate_fix_to_item_dist(fix_coords, item_coords):
    """Number of fixations and number of items are variable. Create a nan-padded distance matrix to average."""
    max_n_fix = 34
    max_n_items = 15
    dist_mat = np.empty((max_n_fix, max_n_items))
    dist_mat[:] = np.nan
    distances = euclidean_distances(fix_coords, item_coords)
    # r, c = distances.shape
    # dist_mat[0:r, 0:c] = distances
    # return dist_mat
    return distances

def calculate_relative_fixation(dist_mat, fixation_xy, all_item_coords):
    """Transform fixation coordinates into the reference from of the nearest item."""
    # Assign each fixation to its nearest item
    # Calculate the relative position of each fixation from its assigned item
    # Calculate standard deviation of relative fixation position
    assignment = np.argsort(dist_mat, axis=1)[:, 0]
    relative_fixation = fixation_xy - all_item_coords[assignment]
    return relative_fixation

def load_eye1():
    """Load eye tracking data from the first human experiment."""
    all_participants = []
    participants = range(3, 22)
    columns = ['SubjectNumber', 'Object_Type', 'loc_consonants', 'loc_vowels', 
            'loc_symbols', 'fixation_posX', 'fixation_posY', 
            'Image_Name_to_Merge', 'n_consonants', 'n_vowels', 'n_symbols', 
            'total_numerosity', 'Response']
    for par in participants:
        datafile = f'{EYE_HOME1}preprocessed_human_data/preprocessed_data_pandasdf/participant_{par}.pickle'
        # datafile = f'preprocessed_human_data/preprocessed_data_pandasdf/participant_{par}.pickle'

        data = pd.read_pickle(datafile)
        all_participants.append(data[columns])
    tim_data = pd.concat(all_participants)
    all_participants = []
    # Load stimuli  metadata
    stim_dir = EYE_HOME1 + 'stimuli_as_generated/'
    # stim_dir = 'stimuli_as_generated/'
    stim_file = 'stimset_num6-15_nl-0.9_diff0-16_BCNPZAEIOU_distract_grid6_lum[0.2, 0.4, 0.6, 0.8]_gw6_solarized_64_block_{0}.pkl'
    blocks = [1, 2, 3, 4, 5, 7, 8, 9]
    meta = [pd.read_pickle(stim_dir + stim_file.format(block)) for block in blocks]
    all_meta = pd.concat(meta)
    all_meta['image_filename_tim'] = all_meta['image filename'].apply(map_jess_to_tim_blocks)

    tim_data = tim_data.merge(all_meta, left_on='Image_Name_to_Merge', right_on='image_filename_tim')
    tim_data = tim_data.rename(columns={'noised image': 'noised_image'})

    # human fixations in pixel space
    for col in ['fixation_posX', 'fixation_posY']:
        tim_data[col] = tim_data[col].apply(convert_to_float_array)
    tim_data['fixation_posX'] = tim_data['fixation_posX'].apply(pixel_to_scaled_x)
    tim_data['fixation_posY'] = tim_data['fixation_posY'].apply(pixel_to_scaled_y)
    # drop extreme fixation values, this would only work on exploded version
    # to_drop = tim_data.query('fixation_posX > 1.25 | fixation_posY > 1.25 | fixation_posX < -0.25 | fixation_posY < -0.25')

    # exploded_on_human = tim_data.explode(column=['fixation_posX', 'fixation_posY'])

    tim_data['fixation_xy'] = tim_data.apply(lambda row: [[x,y] for x,y in zip(row.fixation_posX, row.fixation_posY)], axis=1 )

    tim_data['all_item_coords'] = tim_data.apply(lambda x: calculate_coords(x.locations_A, x.locations_B, x.locations_C), axis=1)
    tim_data['dist_mat'] = tim_data.apply(lambda x: calculate_fix_to_item_dist(x.fixation_xy, x.all_item_coords), axis=1)
    tim_data['assignment'] = tim_data['dist_mat'].apply(lambda x: np.argsort(x, axis=1)[:, 0])
    tim_data['unglimpsed_items'] = tim_data.apply(lambda x: [i for i in range(x.numerosity) if i not in x.assignment], axis=1)
    tim_data['fraction_unglimpsed'] = tim_data.apply(lambda x: len(x.unglimpsed_items)/x.numerosity, axis=1)

    target_lam = lambda x: x.n_consonants if x.Object_Type == 'consonants' else x.n_vowels
    tim_data['target_numerosity'] = tim_data.apply(target_lam, axis=1)
    tim_data['correct'] = tim_data['target_numerosity'] == tim_data['Response']
    return tim_data

def load_eye2():
    """Load eyetracking data from the second human experiment."""
    cond_dirs = ['num3-6_ESUZFCKJsame_count-all_A_main', 'num3-6_ESUZFCKJsame_count-all_B_main', 
                'num3-6_ESUZFCKJsame_ignore-distractors_A_main', 'num3-6_ESUZFCKJsame_ignore-distractors_B_main']
    datadir = EYE_HOME2 + 'preprocessed_human_data/'
    participants = np.arange(1, 27)
    missing = [1, 14]
    par_to_load = np.setdiff1d(participants, missing)
    df = [pd.read_pickle(datadir + f'participant_{p}.pickle') for p in par_to_load]
    for p, participant in enumerate(par_to_load):
        # Load eye position and behaviour
        df[p] = df[p][['fixation_posX_stim', 'fixation_posY_stim', 'fixation_posX_blank', 
                    'fixation_posY_blank', 'View', 'Task', 'Image_Name_to_Merge', 'Response']]
        df[p].fixation_posX_stim = df[p].fixation_posX_stim.apply(convert_to_float_array)
        df[p].fixation_posY_stim = df[p].fixation_posY_stim.apply(convert_to_float_array)
        df[p].fixation_posX_blank = df[p].fixation_posX_blank.apply(convert_to_float_array)
        df[p].fixation_posY_blank = df[p].fixation_posY_blank.apply(convert_to_float_array)
        df[p]['fixation_posX_stim'] = df[p]['fixation_posX_stim'].apply(pixel_to_scaled_x)
        df[p]['fixation_posY_stim'] = df[p]['fixation_posY_stim'].apply(pixel_to_scaled_y)
        df[p]['n_fix_blank'] = df[p].fixation_posX_blank.apply(len)
        df[p] = df[p].assign(View=df[p].View.map({0: "Free", 1: "Fixed"}))
        df[p] = df[p].assign(Task=df[p].Task.map({0: "Ignore Distractors", 1: "Simple Counting"}))
        # Load image metadata
        stim_dir = f'{EYE_HOME2}stimuli/participant_{participant:02d}'
        meta = [pd.read_pickle(f'{stim_dir}/{cond}.pkl') for cond in cond_dirs]
        all_meta = pd.concat(meta)
        df[p] = df[p].merge(all_meta, left_on='Image_Name_to_Merge', right_on='image_filename', validate="1:1")
        df[p]['Subject_Number'] = p
        wrong = sum(df[p][df[p].Task=='Simple Counting'].numerosity_dist != 0)
        print(f'Participant {participant} has {wrong} simple counting trials with distractors.')
    

    df = pd.concat(df, ignore_index=True)
    # df['fixation_xy'] = df.apply(lambda row: [[x,y] for x,y in zip(row.fixation_posX_stim, row.fixation_posY_stim)], axis=1 )
    # df['all_item_coords'] = df.apply(lambda x: calculate_coords(x.locations_count, x.locations_distract), axis=1)
    # df['dist_mat'] = df.apply(lambda x: calculate_fix_to_item_dist(x.fixation_xy, x.all_item_coords), axis=1)
    # df['assignment'] = df['dist_mat'].apply(lambda x: np.argsort(x, axis=1)[:, 0])
    # df['unglimpsed_items'] = df.apply(lambda x: [i for i in range(x.numerosity) if i not in x.assignment], axis=1)
    # df['fraction_unglimpsed'] = df.apply(lambda x: len(x.unglimpsed_items)/x.numerosity, axis=1)
    return df

def bound(x, y):
    """Map out-of-bounds pixel coordinates in-to-bounds."""
    if y==48: y=47 # Translations may lead to out of bounds
    if x==48: x=47
    if y==49: y=47 # Setting off image fixations to 1 could lead to index fo 49 with translation
    if x==49: x=47
    return x, y

def make_map(posX, posY, width, height):
    """Collapse over time to get 2d heatmap of where people looked."""
    map_of_gaze = np.zeros((height, width))
    if len(posX) > 1:
        for fix_no, (x,y) in enumerate(zip(posX, posY)):
            # Keep the first fixation only if not centered
            if fix_no > 0 or ((y<23 and y>25) or (x<23 and x>25)):
                x,y = bound(x, y)
                map_of_gaze[y, x] += 1
    from skimage import measure as sm
    reduced_arr = sm.block_reduce(map_of_gaze, block_size=8, func=np.max)
    return reduced_arr


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
    """Transform reference frames.

    Transform from pixel coordinates in the reference from of the monitor to 
    scaled coordinates between 0 and 1 in the reference from of the stimulus 
    image.
    """
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
    """Transform reference frames.

    Transform from pixel coordinates in the reference from of the monitor to 
    scaled coordinates between 0 and 1 in the reference from of the stimulus 
    image.
    """
    screen_width, screen_height = 1280, 1024
    centerY = screen_height//2
    image_height = screen_height-280 # 744x744 # this is correct because height and width are equal
    upper_left_y = centerY - image_height//2
    
    scaled = [(y-upper_left_y)/image_height for y in coords]
    # If fixation is off image, map to the nearest edge of the image
    scaled = [0 if coord < 0 else coord for coord in scaled]
    scaled = [1 if coord > 1 else coord for coord in scaled]
    return scaled

def index_to_coord(index_list):
    """Convert from index [0-35] to xy image coordinate scaled to [0-1]."""
    index_list = [int(idx) for idx in index_list]
    GRID = np.linspace(0.1, 0.9, 6)
    POSSIBLE_CENTROIDS = [(x, y) for (x, y) in product(GRID, GRID)]
    CENTROID_ARRAY = np.array(POSSIBLE_CENTROIDS)
    # This is to account for the fact that a three pixel border was added to the images
    CENTROID_ARRAY = ((CENTROID_ARRAY*42) + 3) / 48.0
    coords = np.array([CENTROID_ARRAY[idx] for idx in index_list])
    return coords



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
def mirror(coordinates):
    mirrored = [0.5 - (coord-0.5) for coord in coordinates]
    return mirrored
    
def transpose(locations_list):
    """Assumes list of 36 corresponds to 6x6 grid to be transposed.
    Returns a list of length 36 after having transposed the grid."""
    grid = np.reshape(locations_list, (6, 6))
    transposed = list(grid.T.flatten())
    return transposed

def mirror_gridx(locations_list):
    if isinstance(locations_list, list):
        grid = np.reshape(locations_list, (6, 6))
        mirrorx = list(grid[:, ::-1].flatten())
    else:
        grid = locations_list
        mirrorx = grid[:, ::-1]
    return mirrorx

def mirror_gridy(locations_list):
    """Mirror on x axis.
    Used during augmentation to transform the binary lists indicating the 
    locations of targets and distractors. The same function is also used to
    apply the same transformation to the image itself so that (augmented) 
    fixations can be overlayed on a corresponding image. 
    """
    if isinstance(locations_list, list):
        grid = np.reshape(locations_list, (6, 6))
        mirrory = list(grid[::-1, :].flatten())
    else:
        grid = locations_list
        mirrory = grid[::-1, :]
    return mirrory