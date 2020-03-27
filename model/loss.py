from __future__ import print_function, division
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from geotnf.point_tnf import PointTnf
from geotnf.transformation import affine_mat_from_simple

def get_rotate_matrix(theta):
    # cos_alpha = torch.cos(theta)# / 180.0 * np.pi)
    cos_alpha = torch.ones_like(theta, requires_grad=False)
    sin_alpha = torch.sin(theta)# / 180.0 * np.pi)
    zero = torch.zeros_like(theta, requires_grad=False)
    return torch.stack((
        cos_alpha, -sin_alpha, zero,
        sin_alpha, cos_alpha, zero
    ), dim=1)

def get_scale_matrix(theta):
    zero = torch.zeros_like(theta, requires_grad=False)
    return torch.stack((
        theta, zero, zero,
        zero, theta, zero
    ), dim=1)

def get_shift_y_matrix(theta):
    one = torch.ones_like(theta, requires_grad=False)
    zero = torch.zeros_like(theta, requires_grad=False)
    return torch.stack((
        one, zero, zero,
        zero, one, theta
    ), dim=1)

class TransformedGridLoss(nn.Module):
    def __init__(self, geometric_model='affine', use_cuda=True, grid_size=20):
        super(TransformedGridLoss, self).__init__()
        self.geometric_model = geometric_model
        # define virtual grid of points to be transformed
        axis_coords = np.linspace(-1,1,grid_size)
        self.N = grid_size*grid_size
        X,Y = np.meshgrid(axis_coords,axis_coords)
        X = np.reshape(X,(1,1,self.N))
        Y = np.reshape(Y,(1,1,self.N))
        P = np.concatenate((X,Y),1)
        self.P = Variable(torch.FloatTensor(P),requires_grad=False)
        self.pointTnf = PointTnf(use_cuda=use_cuda)
        if use_cuda:
            self.P = self.P.cuda()

    def forward(self, theta, theta_GT):
        # expand grid according to batch size
        batch_size = theta.size(0)
        P = self.P.expand(batch_size,2,self.N)
        # compute transformed grid points using estimated and GT tnfs
        if self.geometric_model == 'affine_simple' or self.geometric_model == 'affine_simple_4':
            theta_aff = affine_mat_from_simple(theta)
            theta_aff_GT = affine_mat_from_simple(theta_GT)
        elif self.geometric_model == 'rotate':
            theta_aff = get_rotate_matrix(theta)
            theta_aff_GT = get_rotate_matrix(theta_GT)
        elif self.geometric_model == 'scale':
            theta_aff = get_scale_matrix(theta)
            theta_aff_GT = get_scale_matrix(theta_GT)
        elif self.geometric_model == 'shift_y':
            theta_aff = get_shift_y_matrix(theta)
            theta_aff_GT = get_shift_y_matrix(theta_GT)
        else:
            raise NotImplementedError('Specified geometric model is unsupported')

        P_prime = self.pointTnf.affPointTnf(theta_aff,P)
        P_prime_GT = self.pointTnf.affPointTnf(theta_aff_GT,P)
        # compute MSE loss on transformed grid points
        loss = torch.sum(torch.pow(P_prime - P_prime_GT,2),1)
        loss = torch.mean(loss)
        return loss


class MixedLoss(nn.Module):
    def __init__(self, alpha=1000, geometric_model='affine', use_cuda=True, grid_size=20):
        super(MixedLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.grid = TransformedGridLoss(geometric_model=geometric_model, use_cuda=use_cuda, grid_size=grid_size)
        self.alpha = alpha
        self.mse_weight = Variable(torch.FloatTensor([1.0, 40.0, 10.0]),requires_grad=False)
        if use_cuda:
            self.mse_weight = self.mse_weight.cuda()

    def forward(self, theta, theta_GT):

        loss = self.mse(theta * self.mse_weight, theta_GT * self.mse_weight) + self.alpha * self.grid(theta, theta_GT)
        return loss

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive

class SplitLoss(nn.Module):
    def __init__(self, geometric_model='affine_simple', use_cuda=True, grid_size=20):
        super(SplitLoss, self).__init__()

        self.rotate_mse = nn.MSELoss()
        self.scale_mse = nn.MSELoss()
        self.shift_mse = nn.MSELoss()

        self.rotate_grid = TransformedGridLoss(geometric_model='rotate', use_cuda=use_cuda, grid_size=grid_size)
        self.scale_grid = TransformedGridLoss(geometric_model='scale', use_cuda=use_cuda, grid_size=grid_size)
        # self.shift_grid = TransformedGridLoss(geometric_model='shift_y', use_cuda=use_cuda, grid_size=grid_size)

        # self.weight = torch.tensor([1.0, 200.0, 2.0, 5000.0, 200.0, 1.0], requires_grad=False)
        self.weight = torch.tensor([1.0, 200.0, 2.0, 1.0, 200.0, 1.0], requires_grad=False)
        if use_cuda:
            self.weight = self.weight.cuda()

    def forward(self, theta, theta_GT):
        loss = self.rotate_mse(theta[:, 0], theta_GT[:, 0])# * self.weight[0] + \
            #    self.rotate_grid(theta[:, 0], theta_GT[:, 0]) * self.weight[3]
            #    self.scale_mse(theta[:, 1], theta_GT[:, 1]) * self.weight[1] + \
            #    self.scale_grid(theta[:, 1], theta_GT[:, 1]) * self.weight[4] # + \
            #    self.shift_mse(theta[:, 2], theta_GT[:, 2]) * self.weight[2] + \
            #    self.shift_grid(theta[:, 2], theta_GT[:, 2]) * self.weight[5]
        # Contrastive_part
        if theta.size(1) > 4:
            loss += self.rotate_mse(theta[:, 0], theta[:, 3]) * self.weight[0] + \
               self.scale_mse(theta[:, 1], theta[:, 4]) * self.weight[1] + \
               self.shift_mse(theta[:, 2], theta[:, 5]) * self.weight[2]
                   
        return loss
