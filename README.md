# Zero-shot visual numerical reasoning with dual-stream glimpsing recurrent neural networks

This repository is associated with the following publication: 

Thompson, J.A., Sheahan, H., Dumbalska, T., Sandbrink, J.D., Piazza, M., and Summerfield, C., Zero-shot counting with a dual-stream neural network model. _Neuron_. 2024.

Relevant human and neural network data can be accessed on OSF: https://osf.io/h6evt/

This code implements theories about the computational ingredients that underly 
structure learning in the primate brain. We investigate the setting where a
learner needs to apply previously learned knowledge about the structure of 
visual scenes (here, numerosity) to new scenes containing unfamiliar objects in
new contexts. The command line interface allows one to test the influence of
various architectural motifs, learning objectives, training curricula, and 
representations on the ability to generalize "zero-shot" in numerical reasoning 
tasks.

All neural networks are implemented in pytorch.

`environment.yml` lists all packages in the conda environment used for these 
experiments. To create a conda environment from this file: 
```conda env create -f environment.yml```

## Dataset Generation

To generate all image datasets used in the published work run `datasets/make_all_datasets.sh`

Image sets are synthesized with `datasets/dataset_generator.py`
The dataset generator synthesizes images and preglimpses them according to a 
specified saccadic policy. Configuration parameters control the number of items,
the shape of the items, the average background and foreground luminances, the
size of the image, and the number of glimpses. Glimpse contents consist of 
logpolar transformed images centred on a fixation point. Datasets are 
manipulated with xarray and stored in netcdf files. Only a handful of literal 
image files are saved for visual inspection. The rest of the image data are
saved along with their corresponding metadata in the netcdf files for ease of
loading and manipulating.

```python3 datasets/dataset_generator.py --policy=random --logpolar  --n_glimpses=20 --luminances 0.1 0.4 0.7 --seed=0 --noise_level=0.9 --size=100000 --shapes ESUZ --min_num=1 --max_num=5 --solarize --n_shapes=25 --grid=6 --same```


## Model Training

To train a model, call the `main.py` script. The command line arguments specify
which datasets should be used for training and testing, which input features
should be used, which model class should be trained, which objectives should be
minimized, and any hyperparameters. 

```python3 main.py --policy='random' --model_type=rnn_classifier2stream --sort --train_on=both --use_loss=both --shape_input=logpolar  --opt=Adam --same  --min_num=1 --max_num=5 --n_glimpses=20 --h_size=1024 --train_shapes=ESUZ --test_shapes ESUZ FCKJ --noise_level=0.9 --train_size=100000 --test_size=1000 --n_epochs=300 --act=lrelu --dropout=0.5 --rep=0 --grid=6```

Some models configurations include pretaining a ventral stream module. This is
done with `ventral.py`. This trains a feedforward network (with or without 
convolutions) to categorize individual glimpses. This saves a pretrained ventral 
stream module can then be loaded into a recurrent dual-stream model.

All code written by Jessica Thompson unless otherwise indicated. An earlier 
version of this project was started by Hannah Sheahan. This code does not build 
directly on hers, but I certainly took inspiration on how to structure the code.

The configurations that were run for the paper are documented in several shell
scripts prefixed with 'submit_'. These are not intended to necessarily be run as
written. One would likely want to distribute over several GPUs in practice, but
these scripts document the command line arguments needed to replicate the
models described in the article.

## Neural Coding
Characterization of the patterns of unit selectivity in the recurrent layer of
the dual-stream RNN was performed in MATLAB. See `main.m` in the 'neural coding'
folder. The unit activations that were analysed can be found in our OSF
repository where you will find both .mat and .npz versions.

## Human Eyetracking Experiment
The PsychToolbox code to run the human eye tracking experiment can be found in
the 'eye_tracking' folder.

## Figures
Code to prepare the figure panels can be found in the 'plot' folder. 
`barplots.py` makes most of the main model comparison figures. 
`plot_development.py` visualizes the qualitative comparisons to human behaviour

