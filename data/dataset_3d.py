from __future__ import print_function, division
import os
import torch
from torch.autograd import Variable
from skimage import io
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from geotnf.transformation import GeometricTnf

class Dataset3D(Dataset):
    
    """
    
    Proposal Flow image pair dataset
    

    Args:
        csv_file (string): Path to the csv file with image names and transformations.
        dataset_path (string): Directory with the images.
        output_size (2-tuple): Desired output size
        transform (callable): Transformation for post-processing the training pair (eg. image normalization)
        
    """

    def __init__(self, csv_file, dataset_path, output_size=(240,240), transform=None):

        self.out_h, self.out_w = output_size
        self.pairs = pd.read_csv(csv_file)
        self.img_A_names = self.pairs.iloc[:,0]
        self.img_B_names = self.pairs.iloc[:,1]
        self.affine_simple_values = self.pairs.iloc[:, 2:].values.astype('float')
        self.dataset_path = dataset_path         
        self.transform = transform
        # no cuda as dataset is called from CPU threads in dataloader and produces confilct
        self.affineTnf = GeometricTnf(out_h=self.out_h, out_w=self.out_w, use_cuda = False) 
              
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # get pre-processed images
        image_A,im_size_A = self.get_image(self.img_A_names,idx)
        image_B,im_size_B = self.get_image(self.img_B_names,idx)

        affine_simple_values = torch.Tensor(self.affine_simple_values[idx, :].astype(np.float32))
        
        sample = {'source_image': image_A, 'target_image': image_B, 'source_im_size': im_size_A, 'target_im_size': im_size_B, 'affine_simple_values': affine_simple_values}
        
        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_image(self,img_name_list,idx):
        img_name = os.path.join(self.dataset_path, img_name_list[idx])
        image = io.imread(img_name)
        
        # get image size
        im_size = np.asarray(image.shape)
        
        # convert to torch Variable
        image = np.expand_dims(image.transpose((2,0,1)),0)
        image = torch.Tensor(image.astype(np.float32))
        image_var = Variable(image,requires_grad=False)
        
        # Resize image using bilinear sampling with identity affine tnf
        image = self.affineTnf(image_var).data.squeeze(0)
        
        im_size = torch.Tensor(im_size.astype(np.float32))
        
        return (image, im_size)