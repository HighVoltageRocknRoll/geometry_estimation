from __future__ import print_function, division
import os
import torch
from torch.autograd import Variable
from skimage import io
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from geotnf.transformation import GeometricTnf
from pythonME.me_handler import MEHandler

class Dataset3DME(Dataset):
    
    """
    
    3D Motion Vectors dataset
    

    Args:
        csv_file (string): Path to the csv file with image names and transformations.
        dataset_path (string): Directory with the images.
        input_size (2-tuple): Size of input images
        crop (float): Cropping factor (from edges before ME)
        
    """

    def __init__(self, csv_file, dataset_path, input_size=(1080,1920), crop=0.2, use_conf=False):

        self.h, self.w = input_size
        self.use_conf = use_conf
        self.pairs = pd.read_csv(csv_file)
        if self.pairs.iloc[0,0].endswith('.npy') or self.pairs.iloc[0,0].endswith('.npz'):
            self.calculate_me = False
        else:
            self.calculate_me = True
        if self.calculate_me:
            self.img_L_names = self.pairs.iloc[:,0]
            self.img_R_names = self.pairs.iloc[:,1]
            self.affine_simple_values = self.pairs.iloc[:, 2:].values.astype('float')
            self.me_handler = MEHandler(int(self.h-self.h*crop), int(self.w-self.w*crop), loss_metric='colorindependent', runs_to_warm_up=1)
            self.crop = crop
        else:
            self.mv_names = self.pairs.iloc[:,0]
            self.affine_simple_values = self.pairs.iloc[:, 1:].values.astype('float')
        self.dataset_path = dataset_path         

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        if self.calculate_me:
            if self.use_conf:
                raise NotImplementedError('Confidence calculating in model has not implemented yet')
            # read images
            image_L = io.imread(os.path.join(self.dataset_path, self.img_L_names[idx]))
            image_R = io.imread(os.path.join(self.dataset_path, self.img_R_names[idx]))

            # Calculate Motion Vectors
            mv_L2R, mv_R2L = self.me_handler.calculate_disparity(image_L, image_R)
        
        else:
            # read .npz file with Motion Vectors
            reader = np.load(os.path.join(self.dataset_path, self.mv_names[idx]))
            mv_L2R = reader['l2r']
            mv_R2L = reader['r2l']
            if self.use_conf:
                conf = reader['conf']
                if conf.ndim == 2:
                    conf = conf[None, ...]

        mv_L2R = torch.Tensor(mv_L2R.astype(np.float32))
        mv_R2L = torch.Tensor(mv_R2L.astype(np.float32))
        affine_simple_values = torch.Tensor(self.affine_simple_values[idx, :].astype(np.float32))
        
        sample = {'mv_L2R': mv_L2R, 'mv_R2L': mv_R2L, 'affine_simple_values': affine_simple_values}

        if self.use_conf:
            conf = torch.Tensor(conf.astype(np.float32))
            sample['confidence'] = conf
        
        return sample