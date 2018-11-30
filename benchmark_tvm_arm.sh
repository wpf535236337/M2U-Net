#!/bin/bash
# test_images: 
# CHASE_DB1: Image_01L.jpg
# DRIVE: 01_test.tif
# HRF : 08_h.jpg
# repeat: 3
# number: 30

python3 benchmark_tvm_arm.py --name M2UNetDRIVE.onnx \
                             --dataset DRIVE \
                             --host 192.168.1.135 \
                             --port 9091 \
                             --target arm_cpu \
                             --devicetype rk3399 \
                             --repeat 3 \
                             --number 30 \
                             --test_image 01_test.tif \