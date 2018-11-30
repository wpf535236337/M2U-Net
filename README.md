# M2U-Net
Code for __"[M2U-Net: Efficient Retinal Vessel Segmentation for Resource Constraint Environments](https://arxiv.org/abs/1811.07738)"__ 

This repository contains the M2U-Net model definition in PyTorch and code for benchmarking inference speeds on GPU, CPU and ARM ([ROCKPro64 SoC](https://www.pine64.org/?product=rockpro64-4gb-single-board-computer)).

## Environment
Tested on Python 3.6 and PyTorch 1.0.0.dev2018092, for more details see requirements.txt.

## Datasets
Download [DRIVE](https://www.isi.uu.nl/Research/Databases/DRIVE/), [CHASE_DB1](https://blogs.kingston.ac.uk/retinal/chasedb1/) and [HRF](https://www5.cs.fau.de/research/data/fundus-images/) and place them in the following directory tree structure:
```
├── data
│   ├── CHASE_DB1
│   │    ├── predictions
│   │    └── test
│   │        └── images
│   ├── DRIVE
│   |    ├── predictions
│   |    └── test
│   |        └── images
|   └── HRF
│        ├── predictions
│        └── test
│            └── images
```
The predictions folder are used for the binary and probabily map outputs of our model.  
`dataset/test/images` folders should contain the test images (see paper for details on train/test split for each dataset). 

## Pretrained models
Trained model weights of M2U-Net, ERFNet and DRIU are included in the model subdirectory. The U-Net is available to download at http://bit.ly/2IeZZUL (119MB). 


## ROCKPro64 Benchmark
### Benchmark PyTorch (NNPACK backend) 
Example to install and run benchmark on a ARM powererd SBC. Tested on a ROCKPro64 with Ubuntu 18.04.
### Setup
Install PyTorch on the device:
```bash
sudo apt-get install python3-dev python3-setuptools python3-numpy

# Pillow 
sudo apt-get update
sudo apt-get install libjpeg-turbo8-dev zlib1g-dev libtiff5-dev
sudo pip3 install Pillow

# build PyTorch Master from source 
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
export NO_CUDA=1
sudo python3 setup.py install

# Torchvision 
git clone https://github.com/pytorch/vision.git
cd vision
sudo python3 setup.py install

```

### Benchmark
Run `bash benchmark_pytorch.sh` on the device. Executes benchmark_pytorch.py with the following arguments:
```
usage: benchmark_pytorch.py [-h] [--model {M2UNet,DRIU,ERFNet,UNet}]
               [--state_dict STATE_DICT] [--dataset {DRIVE,CHASE_DB1,HRF}]
               [--threshold THRESHOLD] [--devicename DEVICENAME]
               [--batch_size BATCH_SIZE] [--save_prob SAVE_PROB]
               [--save_binary_mask SAVE_BINARY_MASK]
               [--save_overlayed SAVE_OVERLAYED]

optional arguments:
  -h, --help            show this help message and exit
  --model {M2UNet,DRIU,ERFNet,UNet}
  --state_dict STATE_DICT
                        name of the pretrained model weights file, stored in "models"
  --dataset {DRIVE,CHASE_DB1,HRF}
                        determines the dataset directory and the amount of cropping that
                        is performed to ensure that the loaded images are
                        multiples of 32.
  --threshold THRESHOLD
                        threshold to convert probability vessel map to binary
                        map
  --devicename DEVICENAME
                        device type, default: "cpu"
  --batch_size BATCH_SIZE
                        inference batch size, default:  1
  --save_prob SAVE_PROB
                        save probability vessel maps to disk
  --save_binary_mask SAVE_BINARY_MASK
                        save binary mask to disk
```
Reports batch inference times and mean inference time per image. 

### Benchmark TVM
#### Setup
Requires TVM on the target device (e.g. ROCKPro64) and the local host device (e.g. MacBook Pro).  
For installation on the host, follow the official [TVM instructions](https://docs.tvm.ai/install/from_source.html).  
On the __target device__, follow the below instructions:
  
Install OpenCL driver (only required if model should run on the Mali GPU):
```bash
sudo apt-get install libmali-rk-midgard-t86x-r14p0
sudo apt-get install opencl-headers
```  
Install TVM:
```bash
git clone --recursive https://github.com/dmlc/tvm
cd tvm
cp cmake/config.cmake .
sed -i "s/USE_OPENCL OFF/USE_OPENCL ON/" config.cmake
make runtime -j4
```
add `export PYTHONPATH=$PYTHONPATH:~/tvm/python` to `~/.bashr` and execute `source ~/.bashrc`.  

Run RPC server:  

`python3 -m tvm.exec.rpc_server --host 0.0.0.0 --port=9090`

#### Bechmark
Run `bash benchmark_tvm_arm.sh` on the __host__ machine.  Executes `benchmark_tvm_arm.py` with the following arguments:

```bash
usage: benchmark_tvm_arm.py [-h] [--name NAME]
                            [--dataset {DRIVE,CHASE_DB1,HRF}] [--host HOST]
                            [--port PORT] [--target {arm_cpu,mali_gpu}]
                            [--devicetype DEVICETYPE] [--repeat REPEAT]
                            [--number NUMBER] [--test_image TEST_IMAGE]

optional arguments:
  -h, --help            show this help message and exit
  --name NAME           name of the onnx model file
  --dataset {DRIVE,CHASE_DB1,HRF}
                        determines the dataset directory and the amount of
                        cropping that is performed to ensure that the loaded
                        images are multiples of 32.
  --host HOST           local ip address of ROCKPro64
  --port PORT           RPC port of remote arm device
  --target {arm_cpu,mali_gpu}
  --devicetype          default: rk3399
  --repeat REPEAT       number of times to run the timer measurement if repeat
  --number NUMBER       the number of steps used in measuring each time
                        interval
  --test_image TEST_IMAGE
                        image to test

```
:warning: make sure to set the local ip address and port of your remote target device.