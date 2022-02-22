#!/bin/bash

# MAPSIZES='10 15 25 50 100 200 400'
MAPSIZES='25 50 100 200 400'
MODELVERSION=two_step_ws
USELOSS=map

for i in $MAPSIZES; do
  echo $i
  python3 -u count_unique.py --map_size=$i --model_version=$MODELVERSION \
  --use_loss=$USELOSS --rnn_act=tanh --no_cuda 2>&1 > logs/${MODELVERSION}_${i}_${USELOSS}.log&
done
# put the big one on the gpu
# python3 -u count_unique.py --map_size=676 --model_version=$MODELVERSION \
# --use_loss=$USELOSS --rnn_act=tanh 2>&1 > logs/${MODELVERSION}_676_${USELOSS}.log&
