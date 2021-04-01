from __future__ import print_function, division
import torch
import os
from os.path import exists, join, basename
from skimage import io
import warnings
import pandas as pd
import numpy as np
try:
    import cv2
    from pythonME.me_handler import MEHandler
except:
    print("Modules cv2, pyME are not loaded: warping or ME on training may not work")
    pass
from torch.utils.data import Dataset
from geotnf.transformation import GeometricTnf
from torch.autograd import Variable
from geotnf.transformation import homography_mat_from_4_pts


class Warper(object):
    """
    
    Image pair warping class (using cv2)

    
    Args:
            H, W (int): shape of input images.
            geometric_model (str): type of geometric model {affine_simple, affine_simple_4}
            crop (float): crop factor after image warping.
            
    """
    def __init__(self, H, W, geometric_model='affine_simple_4', crop=0.2):
        if geometric_model == 'affine_simple_4':
            self.add_tx = True
        elif geometric_model == 'affine_simple':
            self.add_tx = False
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
            rotate_value = (np.random.rand() - 0.5) * 2 * 0.75 # between -0.75 and 0.75. means angle, not radians
            scale_value = 1 + (np.random.rand() - 0.5) * 2 * 0.015 # between 0.985 and 1.015
            shift_value = (np.random.rand() - 0.5) * 2 * 10 # between -10 and 10. means pixels
            self.mat = cv2.getRotationMatrix2D((W//2, H//2), rotate_value, scale_value)
            self.mat[1,2] += shift_value
            self.mat[0,2] += tx
            self.theta = np.array([rotate_value, scale_value, shift_value / self.H, tx / self.W])
        
        if not self.add_tx:
            self.mat[0,2] -= tx
            self.theta = self.theta[:3]
    
    def crop(self, img):
        return img[self.H_off:-self.H_off, self.W_off:-self.W_off]
    
    def warp(self, img_L, img_R):
        return self.crop(img_L), self.crop(cv2.warpAffine(img_R, self.mat, dsize = (self.W, self.H)))
        
    def get_theta(self):
        return self.theta


class MEDataset(Dataset):
    """
    
    Motion Vectors dataset between image pair 

    
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
                 input_height, input_width,
                 crop,
                 use_conf,
                 use_random_patch,
                 normalize_inputs,
                 geometric_model='affine_simple_4', 
                 random_sample=True,
                 load_images=False):
    
        # read csv file
        self.csv = pd.read_csv(os.path.join(dataset_csv_path, dataset_csv_file))
        self.random_sample = random_sample
        self.use_random_patch = use_random_patch
        self.normalize_inputs = normalize_inputs

        h_cropped = int(input_height - input_height*crop)
        w_cropped = int(input_width - input_width*crop)
        self.grid = np.stack(np.indices((h_cropped, w_cropped), dtype=np.float32)[::-1], axis=0)[..., ::4, ::4]

        if self.csv.iloc[0,0].endswith('.npy') or self.csv.iloc[0,0].endswith('.npz'):
            self.image_input = False
            if not self.random_sample:
                self.mv_names = self.csv.iloc[:,0]
                self.theta = self.csv.iloc[:, 1:].values.astype('float')
            else:
                raise ValueError('Incorrect attempt for using ME Dataset')
        else:
            ### Reading images and calculating ME (+conf) for them
            ### Not used for now.
            warnings.warn("Using ME Dataset with images as input.")
            self.img_L_names = self.csv.iloc[:,0]
            if not self.random_sample:
                self.img_R_names = self.csv.iloc[:,1]
                self.theta = self.csv.iloc[:, 2:].values.astype('float')
            self.me_handler = MEHandler(h_cropped, w_cropped, loss_metric='colorindependent', runs_to_warm_up=1)

        # copy args
        self.dataset_path = dataset_image_path
        self.geometric_model = geometric_model
        self.use_conf = use_conf
        self.crop = crop
        self.load_images = load_images

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        if not self.image_input:
             # read .npz file with Motion Vectors
            reader = np.load(os.path.join(self.dataset_path, self.mv_names[idx]))
            mv_L2R = reader['l2r']
            mv_R2L = reader['r2l']
            if self.use_conf:
                conf_L = reader['conf_l']
                conf_R = reader['conf_r']
                if conf_L.ndim == 2:
                    conf_L = conf_L[None, ...]
                if conf_R.ndim == 2:
                    conf_R = conf_R[None, ...]
            if self.load_images:
                img_R_orig = io.imread(os.path.join(
                    self.dataset_path, 'images',
                    '.'.join(['_'.join(self.mv_names[idx].split('.')[0].split('_')[:-1] + ['0']), 'png'])))
                img_R = io.imread(os.path.join(self.dataset_path,  'images',
                                               '.'.join([self.mv_names[idx].split('.')[0], 'png'])))
            theta = self.theta[idx, :]
        else:
            if self.use_conf:
                raise NotImplementedError('Calculating confidence in ME dataset has not implemented yet')
            image_L = io.imread(os.path.join(self.dataset_path, self.img_L_names[idx]))
            if self.random_sample:
                # Warp with random transformations
                h, w, _ = image_L.shape
                warper = Warper(h, w, geometric_model=self.geometric_model, crop=self.crop)
                image_L, image_R = warper.warp(image_L, image_L.copy())
                theta = warper.get_theta()
            else:
                image_R = io.imread(os.path.join(self.dataset_path, self.img_R_names[idx]))
                theta = self.theta[idx, :]
            # Calculate Motion Vectors
            mv_L2R, mv_R2L = self.me_handler.calculate_disparity(image_L, image_R)

        grid = self.grid

        if self.normalize_inputs:
            _, h, w = mv_L2R.shape
            space_w = np.linspace(-1.0, 1.0, num=w)
            space_h = np.linspace(-1.0, 1.0, num=h)
            grid = np.stack(np.meshgrid(space_w, space_h))

            mv_L2R[0] /= w/2
            mv_L2R[1] /= h/2

            mv_R2L[0] /= w/2
            mv_R2L[1] /= h/2

        # make arrays float tensor for subsequent processing
        grid_L2R = torch.Tensor((grid + mv_L2R).astype(np.float32))
        grid_R2L = torch.Tensor((grid + mv_R2L).astype(np.float32))
        mv_L2R = torch.Tensor(mv_L2R.astype(np.float32))
        mv_R2L = torch.Tensor(mv_R2L.astype(np.float32))
        grid = torch.Tensor(grid)
        theta = torch.Tensor(theta.astype(np.float32))

        sample = {
            'mv_L2R': mv_L2R,
            'mv_R2L': mv_R2L,
            'grid': grid,
            'grid_L2R': grid_L2R,
            'grid_R2L': grid_R2L,
            'theta_GT': theta,
            'affine_simple_values': theta,
        }
        
        if self.use_conf:
            conf_L = torch.Tensor(conf_L.astype(np.float32))
            conf_R = torch.Tensor(conf_R.astype(np.float32))
            sample['conf_L'] = conf_L
            sample['conf_R'] = conf_R

        if self.load_images:
            img_R_orig = torch.Tensor(img_R_orig.astype(np.float32) / 255.0)
            img_R = torch.Tensor(img_R.astype(np.float32) / 255.0)
            sample['img_R_orig'] = torch.unsqueeze(img_R_orig, dim=0)
            sample['img_R'] = torch.unsqueeze(img_R, dim=0)

        return sample
