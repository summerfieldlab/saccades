# Simulations of human experiment with fixed viewing 

# logpolar_mixed means that during training, model is exposed to a 50% mixture of fixed and free glimpses

# train dual stream Simple Counting With Covert Saccades

for i in $(seq 0 19);
do
    python3 main.py --model_type=rnn_classifier2stream --wd=0.0001 --human_sim --sort --train_on=both --use_loss=both --opt=Adam --same  --shape_input=logpolar_mixed  --min_num=3 --max_num=6 --n_glimpses=12 --h_size=1024 --train_shapes=ESUZFCKJ --test_shapes ESUZFCKJ ESUZFCKJ --noise_level=0.74 --train_size=100000 --test_size=5000 --n_epochs=300 --act=lrelu --dropout=0.5 --rep=$i --grid=6 
    pid=$!
    wait $pid
done

# Train ventral stream for human task
python3 ventral.py --model_type=cnn --policy=cheat+jitter --logpolar --sort --loss=mse --solarize --same --challenge=distract123 --shape_input=logpolar --min_num=3 --max_num=6 --train_shapes=ESUZFCKJ --test_shapes ESUZFCKJ --lums 0.3 0.6 0.9 --noise_level=0.74 --train_size=20000 --test_size=2000 --act=lrelu --dropout=0.4 --rep=10 --grid=6 --n_epochs=200 --opt=Adam --n_glimpses=15


# pretrained ventral Ignore Distractors

for i in $(seq 0 19);
do
    python3 main.py --model_type=pretrained_ventral-cnn-mse  --human_sim --sort --pass_penult --train_on=both --use_loss=both --opt=Adam --wd=0.0001 --same --challenge=distract123 --shape_input=logpolar_mixed --min_num=3 --max_num=6 --n_glimpses=12 --h_size=1024 --train_shapes=ESUZFCKJ --test_shapes ESUZFCKJ ESUZFCKJ --noise_level=0.74 --train_size=100000 --test_size=5000 --n_epochs=300 --act=lrelu --dropout=0.5 --rep=$i --grid=6 --ventral='ventral_cnn-lrelu_hsize-25_logpolar_num3-6_nl-0.74_diff-0-6_grid6_policy-cheat+jitter_lum-[0.3, 0.6, 0.9]_trainshapes-ESUZFCKJsame_distract123_logpolar_20000_loss-mse_opt-Adam_drop0.4_sort_200eps_rep10_ep-200.pt'
    pid=$!
    wait $pid
done

# WITHOUT COVERT SACCADES

# Simple
for i in $(seq 0 19);
do
    python3 main.py --model_type=rnn_classifier2stream --wd=0.0001 --human_sim --sort --train_on=both --use_loss=both --opt=Adam --same  --shape_input=logpolar_mixed_no_covert  --min_num=3 --max_num=6 --n_glimpses=12 --h_size=1024 --train_shapes=ESUZFCKJ --test_shapes ESUZFCKJ ESUZFCKJ --noise_level=0.74 --train_size=100000 --test_size=5000 --n_epochs=300 --act=lrelu --dropout=0.5 --rep=$i --grid=6 
    pid=$!
    wait $pid
done

# # Ignore Distractors
for i in $(seq 0 19);
do
    python3 main.py --model_type=pretrained_ventral-cnn-mse --human_sim --sort --pass_penult --train_on=both --use_loss=both --opt=Adam --wd=0.0001 --same --challenge=distract123 --shape_input=logpolar_mixed_no_covert --min_num=3 --max_num=6 --n_glimpses=12 --h_size=1024 --train_shapes=ESUZFCKJ --test_shapes ESUZFCKJ ESUZFCKJ --noise_level=0.74 --train_size=100000 --test_size=5000 --n_epochs=300 --act=lrelu --dropout=0.5 --rep=$i --grid=6 --ventral='ventral_cnn-lrelu_hsize-25_logpolar_num3-6_nl-0.74_diff-0-6_grid6_policy-cheat+jitter_lum-[0.3, 0.6, 0.9]_trainshapes-ESUZFCKJsame_distract123_logpolar_20000_loss-mse_opt-Adam_drop0.4_sort_200eps_rep10_ep-200.pt'
    pid=$!
    wait $pid
done


# # CONV NET

# # Simple
for i in $(seq 0 19);
do
    python3 main.py --model_type=bigcnn --human_sim --sort --train_on=shape --use_loss=num --opt=Adam --wd=0.00001 --same  --shape_input=noise --whole_image  --min_num=3 --max_num=6 --n_glimpses=12 --h_size=1024 --train_shapes=ESUZFCKJ --test_shapes ESUZFCKJ ESUZFCKJ --noise_level=0.74 --train_size=100000 --test_size=5000 --n_epochs=300 --act=lrelu --dropout=0.5 --rep=$i --grid=6 
    pid=$!
    wait $pid
done



# # Ignore Distractors
for i in $(seq 0 19);
do
    python3 main.py  --model_type=bigcnn --human_sim  --sort --train_on=shape --use_loss=num --opt=Adam --wd=0.00001 --same --challenge=distract123 --shape_input=noise --whole_image --min_num=3 --max_num=6 --n_glimpses=12 --h_size=1024 --train_shapes=ESUZFCKJ --test_shapes ESUZFCKJ ESUZFCKJ --noise_level=0.74 --train_size=100000 --test_size=5000 --n_epochs=300 --act=lrelu --dropout=0.5 --rep=$i --grid=6 
    pid=$!
    wait $pid
done
