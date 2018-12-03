#!/bin/bash
# Optimal thresholds along precision vs. recall curve (max Dice coefficent):
#         DRIVE   CHASE_DB1
# M2UNet  0.5215  0.4627
# DRIU    0.1529    -
# ERFNet  0.1490  0.1176
# UNet    0.1333    -

# devicename: 'cpu' or gpu (e.g. 'cuda:0')

python3 benchmark_pytorch.py --model M2UNet \
                             --state_dict M2UNetDRIVE.pth \
                             --dataset DRIVE \
                             --threshold 0.5215 \
                             --devicename cpu \
                             --batch_size 1 \
                             --save_prob False \
                             --save_binary_mask False \
                             
