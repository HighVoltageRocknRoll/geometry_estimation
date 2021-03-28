from __future__ import print_function, division
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from geotnf.point_tnf import PointTnf
from geotnf.transformation import affine_mat_from_simple

def get_rotate_matrix(theta):
    cos_alpha = torch.cos(theta / 180.0 * np.pi)
    # cos_alpha = torch.ones_like(theta, requires_grad=False)
    sin_alpha = torch.sin(theta / 180.0 * np.pi)
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

def get_vqmt3d_matrix(rotate_angle, scale_val):
    cos_alpha = scale_val
    sin_alpha = torch.sin(rotate_angle / 180.0 * np.pi)
    zero = torch.zeros_like(rotate_angle, requires_grad=False)
    return torch.stack((
        cos_alpha, -sin_alpha, zero,
        sin_alpha, cos_alpha, zero
    ), dim=1)

class SequentialGridLoss(nn.Module):
    def __init__(self, use_cuda=True, grid_size=20):
        super(SequentialGridLoss, self).__init__()
        self.N = grid_size * grid_size
        # define virtual grid of points to be transformed
        axis_coords = np.linspace(-1, 1, grid_size)
        X,Y = np.meshgrid(axis_coords, axis_coords)
        X = X.ravel()[None, None, ...]
        Y = Y.ravel()[None, None, ...]
        P = np.concatenate((X, Y), axis=1)
        self.P = torch.tensor(P, dtype=torch.float32, requires_grad=False)
        self.pointTnf = PointTnf(use_cuda=use_cuda)
        self.weights = torch.tensor([5000.0, 3000.0, 3000.0], requires_grad=False)
        if use_cuda:
            self.P = self.P.cuda()
            self.weights = self.weights.cuda()

    def warp_and_mse(self, mat, mat_GT, P, P_GT):
        P_warp = self.pointTnf.affPointTnf(mat, P)
        P_warp_GT = self.pointTnf.affPointTnf(mat_GT, P_GT)
        torch.nn.MSELoss()
        loss = torch.sum(torch.pow(P_warp - P_warp_GT, 2), 1)
        loss = torch.mean(loss)
        return loss, P_warp, P_warp_GT

    def forward(self, theta, theta_GT):
        # expand grid according to batch size
        batch_size = theta.size(0)
        P = self.P.expand(batch_size, 2, self.N)

        rotate_mat = get_rotate_matrix(theta[:, 0])
        rotate_mat_GT = get_rotate_matrix(theta_GT[:, 0])
        loss_rotate, P_rotate, P_rotate_GT = self.warp_and_mse(rotate_mat, rotate_mat_GT, P, P)

        scale_mat = get_scale_matrix(theta[:, 1])
        scale_mat_GT = get_scale_matrix(theta_GT[:, 1])
        loss_scale, P_scale, P_scale_GT = self.warp_and_mse(scale_mat, scale_mat_GT, P_rotate, P_rotate_GT)

        shift_mat = get_shift_y_matrix(theta[:, 2])
        shift_mat_GT = get_shift_y_matrix(theta_GT[:, 2])
        loss_shift, P_shift, P_shift_GT = self.warp_and_mse(shift_mat, shift_mat_GT, P_scale, P_scale_GT)

        return self.weights[0] * loss_rotate + self.weights[1] * loss_scale + self.weights[2] * loss_shift

class WeightedMSELoss(nn.Module):
    def __init__(self, use_cuda=True):
        super(WeightedMSELoss, self).__init__()

        self.mse = nn.MSELoss()
        self.weights = torch.tensor([1.0, 10000.0, 10000.0], requires_grad=False)
        self.saved_values = torch.zeros_like(self.weights)
        if use_cuda:
            self.weights = self.weights.cuda()

    def forward(self, theta, theta_GT):
        return torch.sum(torch.stack([self.weights[i] * self.mse(theta[:, i], theta_GT[:, i])
                                      for i in range(len(self.weights))]))

class SplitLoss(nn.Module):
    def __init__(self, use_cuda=True, grid_size=20):
        super(SplitLoss, self).__init__()

        self.rotate_mse = nn.MSELoss()
        self.scale_mse = nn.MSELoss()
        self.shift_mse = nn.MSELoss()

        self.weighted_mse = WeightedMSELoss(use_cuda=use_cuda)
        self.sequential_grid = SequentialGridLoss(use_cuda=use_cuda, grid_size=grid_size)

    def forward(self, theta, theta_GT):
        loss = self.weighted_mse(theta, theta_GT) + self.sequential_grid(theta, theta_GT)
        
        if theta.size(1) > 4:
            loss += self.rotate_mse(theta[:, 0], theta[:, 3]) * self.weight[0] + \
               self.scale_mse(theta[:, 1], theta[:, 4]) * self.weight[1] + \
               self.shift_mse(theta[:, 2], theta[:, 5]) * self.weight[2]
                   
        return loss
