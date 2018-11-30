import tvm
import nnvm.compiler
import nnvm.testing
import nnvm
import onnx
from tvm import rpc
from tvm.contrib import util, graph_runtime as runtime
from pathlib import Path
import torchvision.transforms.functional as VF
import torch
from PIL import Image
import numpy as np
from argparse import ArgumentParser


def load_onnx_model(name,model_path,n,c,h,w):
    '''
    Assumes model does not have a final sigmoid layer
    Args:
        name: onnx model name (stored in models Path)
        n: batch size 
        c: channel
        h: height
        w: width 
    '''
    input_shape = (n,c,h,w)
    onnx_model_path = model_path.joinpath(name)
    # Load ONNX model
    onnx_model = onnx.load(str(onnx_model_path))
    # we can load the graph as NNVM compatible model
    sym, params = nnvm.frontend.from_onnx(onnx_model)
    # add sigmoid layer
    sym = nnvm.sym.sigmoid(sym)
    return sym,params,input_shape


def main():
    # Paths 
    parser = ArgumentParser()
    arg = parser.add_argument
    arg('--name',default='M2UNetDRIVE.onnx',type=str,help='name of the onnx model file')
    arg('--dataset',default='DRIVE',choices=['DRIVE','CHASE_DB1','HRF'],type=str,help='determines the dataset directory  and the amount of cropping that is performed to ensure that the loaded images are multiples of 32.')
    arg('--host',default='192.168.1.135',type=str,help='local ip address of remote arm device')
    arg('--port',default=9090,type=int,help='RPC port of remote arm device')
    arg('--target',default='arm_cpu',choices=['arm_cpu','mali_gpu'])
    arg('--devicetype',default='rk3399',help='default: rk3399')
    arg('--repeat',default=3,type=int,help='number of times to run the timer measurement if repeat ')
    arg('--number',default=30,type=int,help='the number of steps used in measuring each time interval')
    arg('--test_image',default='01_test.tif',help='image to test')
    args = parser.parse_args()

    # Paths 
    model_path = Path('models')
    data_path = Path('data')
    dataset_path = data_path.joinpath(args.dataset)
    prediction_output_path = dataset_path.joinpath('predictions')
    image_file_path = dataset_path.joinpath('test/images').joinpath(args.test_image)

    if args.dataset == 'DRIVE':
        w = 544
        h = 544
        c = 3
    if args.dataset == 'CHASE_DB1':
        w = 960
        h = 960
        c = 3
    if args.dataset == 'HRF':
        w = 3504
        h = 2336
        c = 3

    sym, params, input_shape = load_onnx_model(args.name,model_path,1,c,h,w)
    input_name = sym.list_input_names()[0]
    shape_dict = {input_name: input_shape}

    # target host  (rk339 board)
    target_host = 'llvm -target=aarch64-linux-gnu'

    if args.target == 'arm_cpu':
        target = tvm.target.arm_cpu(args.devicetype)
    if args.target == 'mali_gpu':
        target = tvm.target.mali(args.devicetype)

    # compile
    print('Compiling the model')
    with nnvm.compiler.build_config(opt_level=3):
        graph, lib, params = nnvm.compiler.build(sym, target=target
                ,shape=shape_dict, dtype = {'dtype':('float32','int64')},params=params,target_host=target_host)

    # Save the library at local temporary directory.
    tmp = util.tempdir()
    lib_fname = tmp.relpath('net.tar')
    lib.export_library(lib_fname)

    # load test image and transform 
    img = Image.open(image_file_path)

    if args.dataset == 'DRIVE':
        img = VF.center_crop(img,(544,544))
    if args.dataset =='CHASE_DB1':
        img = np.array(img)
        img = img[:,18:978]

    tensor = VF.to_tensor(img)
    tensor = tensor.unsqueeze(0)
    x = tensor.numpy()

    # connect to device
    remote = rpc.connect(args.host, args.port)
    # upload the library to remote device and load it
    remote.upload(lib_fname)
    rlib = remote.load_module('net.tar')

    # create the remote runtime 
    ctx = remote.context(str(target), 0)
    module = runtime.create(graph, rlib, ctx)
    # set parameter (upload params to the remote device)
    print()
    print('uploading parameters to remote device')
    module.set_input(**params)
    # set input data
    module.set_input(input_name, tvm.nd.array(x.astype('float32')))
    print('creating output image')
    module.run()
    out = module.get_output(0)
    out = out.asnumpy()
    out = torch.from_numpy(out)
    preds_img = VF.to_pil_image(out.cpu().data[0])
    preds_img.save(str(prediction_output_path.joinpath(args.test_image[:-3]+'gif')))
    print('benchmarking inference times')
    ftimer = module.module.time_evaluator("run", ctx, args.number, args.repeat)
    prof_res = np.array(ftimer().results)
    print('mean inference time (std dev): {:.5f}s ({:.5f}s)'.format(np.mean(prof_res),np.std(prof_res)))


if __name__ == '__main__':
    main()