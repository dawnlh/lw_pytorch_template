from utils.utils import get_file_path
from utils.image import augment_img
from torch.utils.data import Dataset
import cv2
import numpy as np


# =================
# build dataset
# =================

def build_dataset(cfg_dataset):
    name = cfg_dataset.pop('name')
    if name in DATASETS:
        return DATASETS.get(name)(**cfg_dataset)
    else:
        raise NotImplementedError(f'{name} is not implemented in the DATASETS list')

DATASETS = {}

def add2dataset(cls):
    if cls.__name__ in DATASETS:
        raise ValueError(f'{cls.__name__} is already in the DATASETS list')
    else:
        DATASETS[cls.__name__] = cls
    return cls

# =================
# datasets
# =================

@add2dataset
class NoisyImg(Dataset):

    def __init__(self, status, data_dir, tform=['all'], sigma=0):
        # param
        self.status = status
        self.data_dir = data_dir
        self.tform_op = tform
        self.sigma = sigma

        # get image paths and load images
        ext = ['jpg', 'png', 'tif', 'bmp']
        self.img_paths, self.img_num, skip_num = get_file_path(data_dir, ext)
        # print(f'---> total dataset image num: {self.img_num}')

    def augment_img(self, img, tform_op):
        return augment_img(img, tform_op=tform_op)

    def add_noise(self, img, sigma):
        # add gaussian noise to image
        if isinstance(img, np.int8):
            img = img.astype(np.float32) / 255
        img_noise = img + np.random.normal(0, sigma, img.shape)
        return img_noise.astype(np.float32)

    def __getitem__(self, index):

        img_k = cv2.imread(self.img_paths[index])
        img_k = cv2.cvtColor(img_k, cv2.COLOR_BGR2RGB)
        img_k = img_k.astype(np.float32) / 255
        if self.status == 'train':
            img_gt = self.augment_img(img_k, self.tform_op)
            img_noisy = self.add_noise(img_gt, self.sigma)
        if self.status == 'test':
            img_noisy = img_k
            img_gt = np.zeros_like(img_k)
        return img_noisy.transpose(2, 0, 1), img_gt.transpose(2, 0, 1)

    def __len__(self,):
        return self.img_num

