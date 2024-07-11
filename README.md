# Zero-shot visual numerical reasoning with dual-stream glimpsing recurrent neural networks

This code tests theories about the computational ingredients that underly 
general visual relational reasoning, numerical reasoning in particular, in the
primate brain. The command line interface allows one to test the influence of
various architectural motifs, learning objectives, training curricula, and 
representations on the ability to generalize numerical reasoning tasks to new
objects in new contexts. 

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
convolutions) to categorize individual glimpses. This saves a A pretrained ventral stream
module can then be loaded into a recurrent dual-stream model using the 

All code written by Jessica Thompson unless otherwise indicated. An earlier 
version of this project was started by Hannah Sheahan. This code does not build 
directly on hers, but I certainly took inspiration on how to structure the code.


