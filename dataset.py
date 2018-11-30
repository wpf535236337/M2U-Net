from pathlib import Path
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as VF

def get_file_lists(image_file_path):
    """
    Args:
        image_file_path
    returns:
        list of file names in path
    """
    file_paths = np.array(sorted(list(image_file_path.glob('*'))))
    return file_paths

class RetinaDataset(Dataset):
        """
        Args: 
            image_file names: a list of image file names with path
            dataset: 'DRIVE' or 'CHASE_DB1'
        """
        def __init__(self, file_paths: list,dataset):
            self.file_paths = file_paths
            self.dataset = dataset

        def __len__(self):
            return len(self.file_paths)

        def __getitem__(self,idx):
            # pick a file
            img_file_name = str(self.file_paths[idx]) # pick a file
            img = Image.open(img_file_name)
            
            if self.dataset == 'DRIVE':
                # take center crop to multiple of 32
                img = VF.center_crop(img,(544,544))
            if self.dataset == 'CHASE_DB1':
                img = np.array(img)
                img = img[:,18:978]

            return VF.to_tensor(img)