from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.functional as VF
import torch.backends.cudnn as cudnn
import time
import numpy as np
from PIL import Image
from argparse import ArgumentParser

from dataset import get_file_lists, RetinaDataset
# Networks
from m2unet import m2unet
from driu import DRIU
from erfnet import Net as ERFNet
from unet import UNet



def run(model,dataloader,batch_size,threshold,device,save_prob=False,save_binary_mask=False
            ,pred_path=None,file_names=None):
    """
    
    """
    model.eval().to(device)
    sigm = torch.nn.Sigmoid()
    
    with torch.no_grad():
        batch_times = []
        prob_images_so_far = 0
        bin_images_so_far = 0
        # inference loop
        for i,inputs in enumerate(dataloader):
            # start timer
            start_time = time.perf_counter()
            # to device
            inputs = inputs.to(device) 
            # forward pass
            outputs = model(inputs)
            # prob and threshold 
            pred_prob = sigm(outputs)
            preds = torch.gt(pred_prob,threshold).float()
            # end timer
            end_time = time.perf_counter()
            inf_time = (end_time - start_time)/batch_size
            print('Batch {}/{} inference time per image: {:.5f}s'.format(i+1,len(dataloader),inf_time))
            batch_times.append(inf_time)
            
            # save binary mask
            if save_binary_mask:
                for j in range(inputs.size()[0]):
                    preds_img = VF.to_pil_image(preds.cpu().data[j])
                    pred_name = '{}_vessel_binary.gif'.format(file_names[bin_images_so_far])
                    preds_img.save(pred_path.joinpath(pred_name))
                    bin_images_so_far += 1
                
            # save vessel probabilities 
            if save_prob:
                for j in range(inputs.size()[0]):
                    preds_img = VF.to_pil_image(pred_prob.cpu().data[j])
                    pred_name = '{}_vessel_prob.gif'.format(file_names[prob_images_so_far])
                    preds_img.save(pred_path.joinpath(pred_name))
                    prob_images_so_far += 1

    # ignore first batch for warm up     
    batch_avg = np.mean(batch_times[1:])
    print()
    print('Mean inference time per image: {:.5f}s'.format(batch_avg))
    
def main():
    parser = ArgumentParser()
    arg = parser.add_argument
    arg('--model',default='M2UNet',type=str,choices=['M2UNet','DRIU','ERFNet','UNet'])
    arg('--state_dict',default='M2UNetDRIVE.pth',type=str,help='pretrained model weights file, stored in models')
    arg('--dataset',default='DRIVE',choices=['DRIVE','CHASE_DB1','HRF'],type=str,help='determines the dataset directory and the amount of cropping that is performed to ensure that the loaded images are multiples of 32.')
    arg('--threshold',default=0.5215,type=float,help='threshold to convert probability vessel map to binary map')
    arg('--devicename',default='cpu',type=str,help='device type, default: "cpu"')
    arg('--batch_size',default=1,type=int,help='inference batch size, default: 1')
    arg('--save_prob',default=False,help='save probability vessel maps to disk')
    arg('--save_binary_mask',default=False,help='save binary mask to disk')
        
    # Paths 
    model_path = Path('models')
    data_path = Path('data')
    log_path = Path('logs')
    
    # parse arguments 
    args = parser.parse_args()
    state_dict_path = model_path.joinpath(args.state_dict)
    dataset_path = data_path.joinpath(args.dataset)
    image_file_path = dataset_path.joinpath('test/images')
    prediction_output_path = dataset_path.joinpath('predictions')
    threshold = args.threshold
    devicename = args.devicename
    batch_size = args.batch_size
    dataset = args.dataset
    save_prob = args.save_prob
    save_binary_mask = args.save_binary_mask
    
    # default device type is 'cuda:0'
    pin_memory = True
    cudnn.benchmark = True
    device = torch.device(devicename)

    if devicename == 'cpu':
        # if run on cpu, disable cudnn benchmark and do not pin memory
        cudnn.benchmark = False
        pin_memory = False
        # only run on one core
        torch.set_num_threads(1) 
    
    if args.model == 'M2UNet':
        model = m2unet()
    if args.model == 'DRIU' and dataset == 'DRIVE':
        model = DRIU()
    if args.model == 'DRIU' and dataset == 'CHASE_DB1':
        model = DRIU()
    if args.model == 'ERFNet':
        model = ERFNet(1)   
    if args.model == 'UNet':
        model = UNet(in_channels=3, n_classes=1, depth=5, wf=6, padding=True,batch_norm=False, up_mode='upconv')
        
    state_dict = torch.load(str(state_dict_path),map_location=devicename)
    model.load_state_dict(state_dict,strict=True)
    model.eval()
    # list of all files include path 
    file_paths = get_file_lists(image_file_path)
    # list of file names only
    file_names = list(map(lambda x: x.stem,file_paths))
    # dataloader
    dataloader = DataLoader(dataset = RetinaDataset(file_paths,dataset)
                            ,batch_size = batch_size
                            ,shuffle=False
                            ,num_workers=1
                            ,pin_memory=pin_memory
                           )
    
    

    run(model,dataloader,batch_size,threshold,device,save_prob,save_binary_mask,prediction_output_path,file_names)

if __name__ == '__main__':
    main()