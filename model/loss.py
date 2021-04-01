from __future__ import print_function, division
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from geotnf.point_tnf import PointTnf
from geotnf.transformation import affine_mat_from_simple
from torch.nn.functional import grid_sample, affine_grid


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


def get_cv_matrix(center, theta):
    angle = theta[:, 0] / 180.0 * np.pi
    alpha = theta[:, 1] * torch.cos(angle)
    beta = theta[:, 1] * torch.sin(angle)
    t_x = (1.0 - alpha) * center[0] - beta * center[1]
    t_y = beta * center[0] + (1.0 - alpha) * center[1] + theta[:, 2]

    return torch.stack((
        alpha, beta, t_x,
        -beta, alpha, t_y
    ), dim=1)


def normalize_transform(transform, width, height):
    one = torch.ones_like(transform[:, 0], requires_grad=False)
    zero = torch.zeros_like(transform[:, 0], requires_grad=False)
    transform_sqr = torch.stack([*transform.unbind(dim=1),
                                 zero, zero, one], dim=1).reshape(-1, 3, 3)
    inv_transform = torch.inverse(transform_sqr)

    norm_transform = torch.zeros_like(transform)
    norm_transform[:, 0] = inv_transform[:, 0, 0]
    norm_transform[:, 1] = inv_transform[:, 0, 1] * height / width
    norm_transform[:, 2] = inv_transform[:, 0, 2] * 2 / width + norm_transform[:, 0] + norm_transform[:, 1] - 1.0
    norm_transform[:, 3] = inv_transform[:, 1, 0] * width / height
    norm_transform[:, 4] = inv_transform[:, 1, 1]
    norm_transform[:, 5] = inv_transform[:, 1, 2] * 2 / height + norm_transform[:, 3] + norm_transform[:, 4] - 1.0

    return norm_transform


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
        if use_cuda:
            self.weights = self.weights.cuda()

    def forward(self, theta, theta_GT):
        return torch.sum(torch.stack([self.weights[i] * self.mse(theta[:, i], theta_GT[:, i])
                                      for i in range(len(self.weights))]))


class ReconstructionLoss(nn.Module):
    def __init__(self, width, height, shift_norm, p=1, use_cuda=True):
        super(ReconstructionLoss, self).__init__()
        self.width = width
        self.height = height
        self.shift_norm = shift_norm
        self.p = p
        self.EPS = 1e-16
        self.mask = torch.ones((1, height, width), dtype=torch.float32)
        if use_cuda:
            self.mask = self.mask.cuda()

    def forward(self, theta, img_R, img_R_orig):
        theta[:, 2] = self.shift_norm * theta[:, 2]
        cv_transform = get_cv_matrix((self.width // 2, self.height // 2), theta)
        transform = normalize_transform(cv_transform, self.width, self.height)
        grid = affine_grid(transform.reshape(-1, 2, 3), img_R.shape, align_corners=True)
        warped_img_R_orig = grid_sample(img_R_orig, grid, align_corners=True)
        mask = self.mask.expand(img_R.shape[0], *self.mask.shape)
        warped_mask = grid_sample(mask, grid, align_corners=True)
        warped_mask = warped_mask[:, 0, :, :]

        losses = (torch.sum(warped_mask * torch.norm(img_R - warped_img_R_orig, p=self.p, dim=1),
                            dim=[1, 2]) + self.EPS) / (torch.sum(warped_mask, dim=[1, 2]) + self.EPS)

        return torch.mean(losses)


class SplitLoss(nn.Module):
    def __init__(self, use_cuda=True, grid_size=20):
        super(SplitLoss, self).__init__()

        self.rotate_mse = nn.MSELoss()
        self.scale_mse = nn.MSELoss()
        self.shift_mse = nn.MSELoss()

        self.weighted_mse = WeightedMSELoss(use_cuda=use_cuda)
        self.sequential_grid = SequentialGridLoss(use_cuda=use_cuda, grid_size=grid_size)

        self.weights = torch.tensor([1.0, 10000.0, 10000.0], requires_grad=False)
        if use_cuda:
            self.weights = self.weights.cuda()

    def forward(self, theta, theta_GT):
        loss = self.weighted_mse(theta, theta_GT) + self.sequential_grid(theta, theta_GT)
        
        if theta.size(1) > 4:
            loss += self.rotate_mse(theta[:, 0], theta[:, 3]) * self.weights[0] + \
               self.scale_mse(theta[:, 1], theta[:, 4]) * self.weights[1] + \
               self.shift_mse(theta[:, 2], theta[:, 5]) * self.weights[2]
                   
        return loss
