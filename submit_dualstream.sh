# #!/bin/bash

# # * Simple Counting *
#  save activations
python3 main.py --model_type=rnn_classifier2stream --save_act --sort --train_on=both --use_loss=both --opt=Adam --wd=0.00001 --same  --shape_input=logpolar --min_num=1 --max_num=5 --n_glimpses=12 --h_size=1024 --train_shapes=BCDE --test_shapes BCDE FGHJ --noise_level=0.74 --train_size=100000 --test_size=5000 --n_epochs=300 --act=lrelu --dropout=0.5 --rep=0 --grid=6 

echo Simple Counting
for i in $(seq 1 19);
do
    echo Simple Counting $i
    python3 main.py --model_type=rnn_classifier2stream --sort --train_on=both --use_loss=both --opt=Adam --wd=0.00001 --same  --shape_input=logpolar --min_num=1 --max_num=5 --n_glimpses=12 --h_size=1024 --train_shapes=BCDE --test_shapes BCDE FGHJ --noise_level=0.74 --train_size=100000 --test_size=5000 --n_epochs=300 --act=lrelu --dropout=0.5 --rep=$i --grid=6 
    # Get process ID
    pid=$!
    # Wait until process is completed
    wait $pid
done

# Train ventral stream for ignore distractors task
python3 ventral.py --model_type=cnn --policy=cheat+jitter --logpolar --sort --loss=mse --solarize --same --challenge=distract012 --shape_input=logpolar --min_num=1 --max_num=5 --train_shapes=BCDEFGHJ --test_shapes BCDEFGHJ --lums 0.1 0.4 0.7 0.3 0.6 0.9 --noise_level=0.74 --train_size=40000 --test_size=4000 --act=lrelu --dropout=0.4 --rep=0 --grid=6 --n_epochs=200 --opt=Adam


# # * Ignore Distractors *
for i in $(seq 0 19);
do
    echo Ignore Distractors $i
    python3 main.py --model_type=pretrained_ventral-cnn-mse --sort --pass_penult --train_on=both --use_loss=both --opt=Adam --wd=0.00001 --same --challenge=distract012 --shape_input=logpolar --min_num=1 --max_num=5 --n_glimpses=12 --h_size=1024 --train_shapes=BCDE --test_shapes BCDE FGHJ --noise_level=0.74 --train_size=100000 --test_size=5000 --n_epochs=300 --act=lrelu --dropout=0.5 --rep=$i --grid=6 --ventral='ventral_cnn-lrelu_hsize-25_logpolar_num1-5_nl-0.74_diff-0-6_grid6_policy-cheat+jitter_lum-[0.1, 0.4, 0.7, 0.3, 0.6, 0.9]_trainshapes-BCDEFGHJsame_distract012_logpolar_40000_loss-mse_opt-Adam_drop0.4_sort_200eps_rep0_ep-200.pt'
    pid=$!
    wait $pid
done
