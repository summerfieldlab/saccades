# Synthesize all the image datasets used in the paper


# FOR MODELING
# Simple counting, no distractors
# Test sets
python3 datasets/dataset_generator.py --polar --scaling=log --luminances 0.1 0.4 0.7 --seed=0 --noise_level=0.74 --size=5000 --shapes BCDE --min_num=1 --max_num=5 --solarize --n_shapes=25 --grid=6  --same --n_glimpses=12
python3 datasets/dataset_generator.py --polar --scaling=log --luminances 0.3 0.6 0.9 --seed=1 --noise_level=0.74 --size=5000 --shapes BCDE --min_num=1 --max_num=5 --solarize --n_shapes=25 --grid=6  --same --n_glimpses=12
python3 datasets/dataset_generator.py --polar --scaling=log --luminances 0.1 0.4 0.7 --seed=2 --noise_level=0.74 --size=5000 --shapes FGHJ --min_num=1 --max_num=5 --solarize --n_shapes=25 --grid=6  --same --n_glimpses=12
python3 datasets/dataset_generator.py --polar --scaling=log --luminances 0.3 0.6 0.9 --seed=3 --noise_level=0.74 --size=5000 --shapes FGHJ --min_num=1 --max_num=5 --solarize --n_shapes=25 --grid=6  --same --n_glimpses=12
# Training set
python3 datasets/dataset_generator.py --polar --scaling=log --luminances 0.1 0.4 0.7 --seed=4 --noise_level=0.74 --size=100000 --shapes BCDE --min_num=1 --max_num=5 --solarize --n_shapes=25 --grid=6  --same --n_glimpses=12


# Ignore 0-2 distractors
python3 datasets/dataset_generator.py --challenge=distract012 --polar --scaling=log --luminances 0.1 0.4 0.7 --seed=0 --noise_level=0.74 --size=5000 --shapes BCDE --min_num=1 --max_num=5 --solarize --n_shapes=25 --grid=6  --same --n_glimpses=12
python3 datasets/dataset_generator.py --challenge=distract012 --polar --scaling=log --luminances 0.3 0.6 0.9 --seed=1 --noise_level=0.74 --size=5000 --shapes BCDE --min_num=1 --max_num=5 --solarize --n_shapes=25 --grid=6  --same --n_glimpses=12
python3 datasets/dataset_generator.py --challenge=distract012 --polar --scaling=log --luminances 0.1 0.4 0.7 --seed=2 --noise_level=0.74 --size=5000 --shapes FGHJ --min_num=1 --max_num=5 --solarize --n_shapes=25 --grid=6  --same --n_glimpses=12
python3 datasets/dataset_generator.py --challenge=distract012 --polar --scaling=log --luminances 0.3 0.6 0.9 --seed=3 --noise_level=0.74 --size=5000 --shapes FGHJ --min_num=1 --max_num=5 --solarize --n_shapes=25 --grid=6  --same --n_glimpses=12

python3 datasets/dataset_generator.py --challenge=distract012 --polar --scaling=log --luminances 0.1 0.4 0.7 --seed=4 --noise_level=0.74 --size=100000 --shapes BCDE --min_num=1 --max_num=5 --solarize --n_shapes=25 --grid=6 --same --n_glimpses=12

# To pretrain the ventral stream
# Train and test sets are sampled from same distribution (no OOD generalization) which span all of the test sets above
python3 datasets/dataset_generator.py --policy=cheat+jitter --polar --scaling=log --challenge=distract012 --luminances 0.1 0.4 0.7 --seed=5 --noise_level=0.74 --size=1000 --shapes BCDE --min_num=1 --max_num=5 --solarize --n_shapes=25 --grid=6 --same --n_glimpses=12
python3 datasets/dataset_generator.py --policy=cheat+jitter --polar --scaling=log --challenge=distract012 --luminances 0.3 0.6 0.9 --seed=6 --noise_level=0.74 --size=1000 --shapes BCDE --min_num=1 --max_num=5 --solarize --n_shapes=25 --grid=6 --same --n_glimpses=12

python3 datasets/dataset_generator.py --policy=cheat+jitter --polar --scaling=log --challenge=distract012 --luminances 0.1 0.4 0.7 --seed=7 --noise_level=0.74 --size=1000 --shapes FGHJ --min_num=1 --max_num=5 --solarize --n_shapes=25 --grid=6 --same --n_glimpses=12
python3 datasets/dataset_generator.py --policy=cheat+jitter --polar --scaling=log --challenge=distract012 --luminances 0.3 0.6 0.9 --seed=8 --noise_level=0.74 --size=1000 --shapes FGHJ --min_num=1 --max_num=5 --solarize --n_shapes=25 --grid=6 --same --n_glimpses=12

python3 datasets/dataset_generator.py --policy=cheat+jitter --polar --scaling=log --challenge=distract012 --luminances 0.1 0.4 0.7 --seed=9 --noise_level=0.74 --size=10000 --shapes BCDE --min_num=1 --max_num=5 --solarize --n_shapes=25 --grid=6 --same --n_glimpses=12
python3 datasets/dataset_generator.py --policy=cheat+jitter --polar --scaling=log --challenge=distract012 --luminances 0.3 0.6 0.9 --seed=10 --noise_level=0.74 --size=10000 --shapes BCDE --min_num=1 --max_num=5 --solarize --n_shapes=25 --grid=6 --same --n_glimpses=12

python3 datasets/dataset_generator.py --policy=cheat+jitter --polar --scaling=log --challenge=distract012 --luminances 0.1 0.4 0.7 --seed=11 --noise_level=0.74 --size=10000 --shapes FGHJ --min_num=1 --max_num=5 --solarize --n_shapes=25 --grid=6 --same --n_glimpses=12
python3 datasets/dataset_generator.py --policy=cheat+jitter --polar --scaling=log --challenge=distract012 --luminances 0.3 0.6 0.9 --seed=12 --noise_level=0.74 --size=10000 --shapes FGHJ --min_num=1 --max_num=5 --solarize --n_shapes=25 --grid=6 --same --n_glimpses=12

python3 datasets/merge_datasets.py # merge subsets



# FOR SIMULATING HUMAN EXPERIMENT
# Human experiment uses numerosities 3-6, letters ESUZFCKJA and images have constant (average) contrast of 0.3 from the set of luminances 0.3, 0.6, 0.9
# Simple Counting
python3 datasets/dataset_generator.py --polar --scaling=log  --luminances 0.3 0.6 0.9 --seed=7 --noise_level=0.74 --constant_contrast --size=100000 --shapes ESUZFCKJ --min_num=3 --max_num=6 --solarize --n_shapes=25 --grid=6 --same --n_glimpses=12
python3 datasets/dataset_generator.py --polar --scaling=log  --luminances 0.3 0.6 0.9 --seed=8 --noise_level=0.74 --constant_contrast --size=5000 --shapes ESUZFCKJ --min_num=3 --max_num=6 --solarize --n_shapes=25 --grid=6 --same --n_glimpses=12
python3 datasets/dataset_generator.py --polar --scaling=log  --luminances 0.1 0.4 0.7 --seed=9 --noise_level=0.74 --constant_contrast --size=5000 --shapes ESUZFCKJ --min_num=3 --max_num=6 --solarize --n_shapes=25 --grid=6 --same --n_glimpses=12

# Ignore 1-3 distractors
python3 datasets/dataset_generator.py --polar --scaling=log  --luminances 0.3 0.6 0.9 --seed=7 --noise_level=0.74 --constant_contrast --size=100000 --shapes ESUZFCKJ --min_num=3 --max_num=6 --challenge=distract123 --solarize --n_shapes=25 --grid=6 --same --n_glimpses=12
python3 datasets/dataset_generator.py --polar --scaling=log  --luminances 0.3 0.6 0.9 --seed=8 --noise_level=0.74 --constant_contrast --size=5000 --shapes ESUZFCKJ --min_num=3 --max_num=6 --challenge=distract123 --solarize --n_shapes=25 --grid=6 --same --n_glimpses=12
python3 datasets/dataset_generator.py --polar --scaling=log  --luminances 0.1 0.4 0.7 --seed=9 --noise_level=0.74 --constant_contrast --size=5000 --shapes ESUZFCKJ --min_num=3 --max_num=6 --challenge=distract123 --solarize --n_shapes=25 --grid=6 --same --n_glimpses=12

# for ventral for human sim
python3 datasets/dataset_generator.py --polar --scaling=log  --luminances 0.3 0.6 0.9 --seed=5 --noise_level=0.74 --constant_contrast --size=20000 --shapes ESUZFCKJ --min_num=3 --max_num=6 --challenge=distract123 --solarize --n_shapes=25 --grid=6 --same --n_glimpses=12
python3 datasets/dataset_generator.py --polar --scaling=log  --luminances 0.3 0.6 0.9 --seed=6 --noise_level=0.74 --constant_contrast --size=2000 --shapes ESUZFCKJ --min_num=3 --max_num=6 --challenge=distract123 --solarize --n_shapes=25 --grid=6 --same --n_glimpses=12
