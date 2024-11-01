#!/bin/bash

# ./submit_controls.sh 2>&1 | tee -a logs/controls.log 

# Repetitions of control models to test the influence of various computational ingredients

# * Ignore Distractors without pretraining*
for i in $(seq 0 19);
do
    echo Ignore Distractors no pretraining $i
    python3 main.py --model_type=pretrained_ventral-cnn-mse-finetune --no_pretrain --sort --pass_penult --train_on=both --use_loss=both --opt=Adam --wd=0.00001 --same --challenge=distract012 --shape_input=logpolar --min_num=1 --max_num=5 --n_glimpses=12 --h_size=1024 --train_shapes=BCDE --test_shapes BCDE FGHJ --noise_level=0.74 --train_size=100000 --test_size=5000 --n_epochs=300 --act=lrelu --dropout=0.5 --rep=$i --grid=6 --ventral='ventral_cnn-lrelu_hsize-25_logpolar_num1-5_nl-0.74_diff-0-6_grid6_policy-cheat+jitter_lum-[0.1, 0.4, 0.7, 0.3, 0.6, 0.9]_trainshapes-BCDEFGHJsame_distract012_logpolar_40000_loss-mse_opt-Adam_drop0.4_sort_200eps_rep0_ep-200.pt'
    pid=$!
    wait $pid
done

# Recurrent control (Simple Counting)
for i in $(seq 0 19);
do
    echo Recurrent control $i
    python3 main.py --if_exists=force --model_type=recurrent_control --train_on=shape --sort --use_loss=both --opt=Adam --wd=0.00001 --same --shape_input=noise --whole_image --min_num=1 --max_num=5 --n_glimpses=12 --h_size=1024 --train_shapes=BCDE --test_shapes BCDE FGHJ  --noise_level=0.74 --train_size=100000 --test_size=5000 --n_epochs=300 --act=lrelu --dropout=0.5 --rep=$i
    pid=$!
    wait $pid
done

# MLP with map
for i in $(seq 0 19);
do
    echo mlp $i
    python3 main.py --if_exists=force --model_type=mlp --train_on=shape --sort --use_loss=both --opt=Adam --wd=0.00001 --same --shape_input=noise --whole_image --min_num=1 --max_num=5 --n_glimpses=12 --h_size=1024 --train_shapes=BCDE --test_shapes BCDE FGHJ  --noise_level=0.74 --train_size=100000 --test_size=5000 --n_epochs=300 --act=lrelu --dropout=0.5 --rep=$i
    pid=$!
    wait $pid
done

# MLP without map
for i in $(seq 0 19);
do
    echo mlp $i
    python3 main.py --if_exists=force --model_type=mlp --train_on=shape --sort --use_loss=num --opt=Adam --wd=0.00001 --same --shape_input=noise --whole_image --min_num=1 --max_num=5 --n_glimpses=12 --h_size=1024 --train_shapes=BCDE --test_shapes BCDE FGHJ  --noise_level=0.74 --train_size=100000 --test_size=5000 --n_epochs=300 --act=lrelu --dropout=0.5 --rep=$i
    pid=$!
    wait $pid
done

# Coherence Illusion Test
for i in $(seq 0 19);
do
    echo coh $i
    python3 main.py --model_type=rnn_classifier2stream --sort --train_on=both --use_loss=both --opt=Adam --wd=0.00001 --mixed  --shape_input=logpolar --min_num=1 --max_num=5 --n_glimpses=12 --h_size=1024 --train_shapes=BCDE --test_shapes BCDE BCDE --noise_level=0.74 --train_size=100000 --test_size=5000 --n_epochs=300 --act=lrelu --dropout=0.5 --rep=$i --grid=6 
    pid=$!
    wait $pid
done

# CNN coherence illusion
for i in $(seq 0 19);
do
    echo coh $i
    python3 main.py --model_type=bigcnn --whole_image --sort --train_on=both --use_loss=both --opt=Adam --wd=0.00001 --mixed  --shape_input=logpolar --min_num=1 --max_num=5 --n_glimpses=12 --h_size=1024 --train_shapes=BCDE --test_shapes BCDE BCDE --noise_level=0.74 --train_size=100000 --test_size=5000 --n_epochs=300 --act=lrelu --dropout=0.5 --rep=$i --grid=6 
    pid=$!
    wait $pid
done

## Misaligned streams control
for i in $(seq 0 19);
do
    echo misalign_sim $i
    python3 main.py --model_type=rnn_classifier2stream --misalign --sort --train_on=both --use_loss=both --opt=Adam --wd=0.00001 --same --shape_input=logpolar --min_num=1 --max_num=5 --n_glimpses=12 --h_size=1024 --train_shapes=BCDE --test_shapes BCDE FGHJ --noise_level=0.74 --train_size=100000 --test_size=5000 --n_epochs=300 --act=lrelu --dropout=0.5 --rep=$i --grid=6
    pid=$!
    wait $pid
done
