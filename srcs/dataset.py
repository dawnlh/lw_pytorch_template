
#---------------------------------------
from utils.utils import get_file_path
from utils.image import augment_img
from torch.utils.data import Dataset 
import cv2
import numpy as np
 
# =================
# Datasets
# =================

class NoisyImg(Dataset):
    def __init__(self, data_dir, tform=['all'], sigma=0):
        # param
        self.data_dir= data_dir
        self.tform_op = tform
        self.sigma = sigma

        # get image paths and load images
        ext = ['jpg', 'png', 'tif', 'bmp']
        self.img_paths, self.img_num, skip_num = get_file_path(data_dir, ext)
        # print(f'---> total dataset image num: {self.img_num}')
    
    def augment_img(self,img, tform_op):
        return augment_img(img, tform_op=tform_op)
    
    def add_noise(self, img, sigma):
        # add gaussian noise to image
        if isinstance(img, np.int8):
            img = img.astype(np.float32)/255
        img_noise = img + np.random.normal(0, sigma, img.shape)
        return img_noise.astype(np.float32)
    
    def __getitem__(self,index):
         
        img_k = cv2.imread(self.img_paths[index])
        img_k = cv2.cvtColor(img_k, cv2.COLOR_BGR2RGB)
        img_k = img_k.astype(np.float32)/255
        img_k = self.augment_img(img_k, self.tform_op)
        img_noisy = self.add_noise(img_k, self.sigma)
        return img_noisy.transpose(2, 0, 1), img_k.transpose(2, 0, 1)
    def __len__(self,):
        return self.img_num

class NoisyImgRealexp(Dataset):
    def __init__(self,data_dir,*args,**kwargs):
 
        self.data_dir= data_dir
 
        # get image paths and load images
        ext = ['jpg', 'png', 'tif', 'bmp']
        self.img_paths, self.img_num, skip_num = get_file_path(data_dir, ext)
        # print(f'---> total dataset image num: {self.img_num}')
         
    def __getitem__(self,index):
         
        img_k = cv2.imread(self.img_paths[index])
        img_k = cv2.cvtColor(img_k, cv2.COLOR_BGR2RGB)
        img_k = img_k.astype(np.float32)/255
        img_gt = np.zeros_like(img_k)
        return img_k.transpose(2, 0, 1), img_gt.transpose(2, 0, 1)
    def __len__(self,):
        return self.img_num


def build_dataset(cfg_dataset, status='train'):
    if status in ['train', 'test', 'simuexp']:
        return NoisyImg(**cfg_dataset)
    elif status in 'realexp':
        return NoisyImgRealexp(**cfg_dataset)
    else:
        raise NotImplementedError(
            f"status {status} not implemented")