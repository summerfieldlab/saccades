#!/bin/bash

# Conv Net Models

# * Simple Counting *
echo Simple Counting w/o map
for i in $(seq 0 19);
do
    echo Simple Counting $i
    python3 main.py  --model_type=bigcnn --whole_image  --sort --train_on=shape --use_loss=num --opt=Adam --wd=0.00001 --same  --shape_input=noise --min_num=1 --max_num=5 --n_glimpses=12 --h_size=1024 --train_shapes=BCDE --test_shapes BCDE FGHJ --noise_level=0.74 --train_size=100000 --test_size=5000 --n_epochs=300 --act=lrelu --dropout=0.5 --rep=$i --grid=6 
    # Get process ID
    pid=$!
    # Wait until process is completed
    wait $pid
done


# Simple Counting w/ map
for i in $(seq 0 19);
do
    echo Simple Counting w/ map $i
    python3 main.py  --model_type=bigcnn --whole_image  --sort --train_on=shape --use_loss=both --opt=Adam --wd=0.00001 --same  --shape_input=noise --min_num=1 --max_num=5 --n_glimpses=12 --h_size=1024 --train_shapes=BCDE --test_shapes BCDE FGHJ --noise_level=0.74 --train_size=100000 --test_size=5000 --n_epochs=300 --act=lrelu --dropout=0.5 --rep=$i --grid=6 
    # Get process ID
    pid=$!
    # Wait until process is completed
    wait $pid
done

# # * Ignore Distractors *
for i in $(seq 0 19);
do
    echo Ignore Distractors w/o map $i
    python3 main.py  --model_type=bigcnn --whole_image --sort --pass_penult --train_on=shape --use_loss=num --opt=Adam --wd=0.00001 --same --challenge=distract012 --shape_input=noise --min_num=1 --max_num=5 --n_glimpses=12 --h_size=1024 --train_shapes=BCDE --test_shapes BCDE FGHJ --noise_level=0.74 --train_size=100000 --test_size=5000 --n_epochs=300 --act=lrelu --dropout=0.5 --rep=$i --grid=6
    pid=$!
    wait $pid
done

# Ignore Distractors with map loss
for i in $(seq 0 19);
do
    echo Ignore Distractors w/ map $i
    python3 main.py  --model_type=bigcnn --whole_image --sort --pass_penult --train_on=shape --use_loss=both --opt=Adam --wd=0.00001 --same --challenge=distract012 --shape_input=noise --min_num=1 --max_num=5 --n_glimpses=12 --h_size=1024 --train_shapes=BCDE --test_shapes BCDE FGHJ --noise_level=0.74 --train_size=100000 --test_size=5000 --n_epochs=300 --act=lrelu --dropout=0.5 --rep=$i --grid=6
    pid=$!
    wait $pid
done

# Save CNN activations
python3 main.py  --model_type=bigcnn --save_act --whole_image --sort --pass_penult --train_on=shape --use_loss=both --opt=Adam --wd=0.00001 --same --challenge=distract012 --shape_input=noise --min_num=1 --max_num=5 --n_glimpses=12 --h_size=1024 --train_shapes=BCDE --test_shapes BCDE FGHJ --noise_level=0.74 --train_size=100000 --test_size=5000 --n_epochs=300 --act=lrelu --dropout=0.5 --rep=20 --grid=6 
