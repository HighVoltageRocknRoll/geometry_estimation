from __future__ import print_function, division
import torch
import os
from os.path import exists, join, basename
from skimage import io
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset
from geotnf.transformation import GeometricTnf
from torch.autograd import Variable
from geotnf.transformation import homography_mat_from_4_pts
from pythonME.me_handler import MEHandler

class Warper(object):
    def __init__(self, H, W, geometric_model='affine_simple_4', crop=0.2):
        if geometric_model == 'affine_simple_4':
            self.add_tx = True
        elif geometric_model == 'affine_simple':
            self.add_ty = False
        else:
            raise NotImplementedError('Specified geometric model is unsupported')

        self.H = H
        self.W = W
        self.H_off = int(H * crop) // 2
        self.W_off = int(W * crop) // 2

        tx = (2 * np.random.rand(1) - 1) * 0.1 * self.W # between -0.1*W and 0.1*W
        if np.random.randint(0, 5) == 4:
            # 25% of pairs are set to identity transform with disparity
            self.mat = np.array([[1.0, 0.0, tx], [0.0, 1.0, 0.0]], dtype=np.float32)
            self.theta = np.array([0.0, 1.0, 0.0, tx / self.W])
        else:
            rotate_value = (np.random.rand() - 0.5) * 2 * 0.75 # between -0.75 and 0.75
            scale_value = 1 + (np.random.rand() - 0.5) * 2 * 0.015 # between 0.985 and 1.015
            shift_value = (np.random.rand() - 0.5) * 2 * 10 # between -10 and 10
            self.mat = cv2.getRotationMatrix2D((W//2, H//2), rotate_value, scale_value)
            self.mat[1,2] += shift_value
            self.mat[0,2] += tx
            self.theta = np.array([rotate_value, scale_value, shift_value / self.H, tx / self.W])
        
        if not self.add_tx:
            self.mat[0, 2] = 0.0
            self.theta = self.theta[:3]
    
    def crop(self, img):
        return img[self.H_off:-self.H_off, self.W_off:-self.W_off]
    
    def warp(self, img_L, img_R):
        return self.crop(img_L), self.crop(cv2.warpAffine(img_R, self.mat, dsize = (self.W, self.H)))
        
    def get_theta(self):
        return self.theta


class SynthDatasetME(Dataset):
    """
    
    Motion Vectors dataset between synthetically generated image pair for training with strong supervision
    
    Args:
            dataset_csv_path (string): Path to the csv file with image names and transformations.
            dataset_csv_file (string): Filename of the csv file with image names and transformations.
            dataset_image_path (string): Directory with all the images.
            h, w (int): size of input images for ME initialization.
            crop (float): crop factor after image warping.
            
    Returns:
            Dict: {
                    'mv_L2R': Motion Vectors from source (assumed as Left view) to warped (Right view),
                    'mv_R2L': Motion Vectors backward,
                    'theta_GT': desired transformation
                  }
            
    """

    def __init__(self,
                 dataset_csv_path, 
                 dataset_csv_file, 
                 dataset_image_path, 
                 h, w,
                 crop,
                 use_conf,
                 geometric_model='affine_simple_4', 
                 random_sample=True):
    
        # read csv file
        self.train_data = pd.read_csv(os.path.join(dataset_csv_path,dataset_csv_file))
        self.random_sample = random_sample
        self.use_conf = use_conf
        self.img_names = self.train_data.iloc[:,0]
        h_cropped = h - h*crop
        w_cropped = w - w*crop
        if self.random_sample==False:
            self.theta_array = self.train_data.iloc[:, 1:].values.astype('float')
        else:
            self.me_handler = MEHandler(h_cropped, w_cropped, loss_metric='colorindependent', runs_to_warm_up=1)
            self.crop = crop
        
        self.grid = np.stack(np.indices((h_cropped, w_cropped), dtype=np.float32)[::-1], axis=0)
        # copy arguments
        self.dataset_image_path = dataset_image_path
        self.geometric_model = geometric_model
        
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        if self.random_sample:
            if self.use_conf:
                raise NotImplementedError('Calculating confidence in synth dataset has not implemented yet')
            # read image
            img_name = os.path.join(self.dataset_image_path, self.img_names[idx])
            image_L = io.imread(img_name)

            # Warp with random transformations
            h, w, _ = image_L.shape
            warper = Warper(h, w, geometric_model=self.geometric_model, crop=self.crop)
            image_L, image_R = warper.warp(image_L, image_L.copy())
            theta = warper.get_theta()

            # Calculate Motion Vectors
            mv_L2R, mv_R2L = self.me_handler.calculate_disparity(image_L, image_R)

            # permute order to CHW
            # mv_L2R = mv_L2R.transpose(2,0,1)
            # mv_R2L = mv_R2L.transpose(2,0,1)
        else:
            # read .npz file with Motion Vectors
            reader = np.load(os.path.join(self.dataset_image_path, self.img_names[idx]))
            mv_L2R = reader['l2r']
            mv_R2L = reader['r2l']
            if self.use_conf:
                conf = reader['conf']
                if conf.ndim == 2:
                    conf = conf[None, ...]
            theta = self.theta_array[idx, :]

        # make arrays float tensor for subsequent processing
        mv_L2R = torch.Tensor(mv_L2R.astype(np.float32))
        mv_R2L = torch.Tensor(mv_R2L.astype(np.float32))
        grid = torch.Tensor(self.grid)
        theta = torch.Tensor(theta.astype(np.float32))

        sample = {'mv_L2R': mv_L2R, 'mv_R2L': mv_R2L, 'grid': grid, 'theta_GT': theta}
        
        if self.use_conf:
            conf = torch.Tensor(conf.astype(np.float32))
            sample['confidence'] = conf

        return sample
