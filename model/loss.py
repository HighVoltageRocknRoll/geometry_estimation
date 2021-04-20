from __future__ import print_function, division
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from geotnf.point_tnf import PointTnf
from geotnf.transformation import affine_mat_from_simple
from torch.nn.functional import grid_sample, affine_grid
from .cv_transform import get_cv_matrix, normalize_transform


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
        self.weights = [5000.0, 3000.0, 3000.0]
        if use_cuda:
            self.P = self.P.cuda()

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
        self.weights = [1.0, 10000.0, 10000.0]

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
        self.weight = 10.0
        self.EPS = 1e-16
        self.mask = torch.ones((1, height, width), dtype=torch.float32, requires_grad=False)
        if use_cuda:
            self.mask = self.mask.cuda()

    def forward(self, theta, img_R, img_R_orig):
        cv_transform = get_cv_matrix((self.width // 2, self.height // 2),
                                     theta[:, 0], theta[:, 1], self.shift_norm * theta[:, 2])
        transform = normalize_transform(cv_transform, self.width, self.height)
        grid = affine_grid(transform.reshape(-1, 2, 3), img_R.shape, align_corners=True)
        warped_img_R_orig = grid_sample(img_R_orig, grid, align_corners=True)
        mask = self.mask.expand(img_R.shape[0], *self.mask.shape)
        warped_mask = grid_sample(mask, grid, align_corners=True)
        warped_mask = warped_mask[:, 0, :, :]

        losses = self.weight * (torch.sum(warped_mask * torch.norm(img_R - warped_img_R_orig, p=self.p, dim=1),
                                          dim=[1, 2]) + self.EPS) / (torch.sum(warped_mask, dim=[1, 2]) + self.EPS)

        return torch.mean(losses)


class CombinedLoss(nn.Module):
    def __init__(self, args, use_cuda=True):
        super(CombinedLoss, self).__init__()

        if args.use_weighted_mse_loss:
            self.weighted_mse = WeightedMSELoss(use_cuda=use_cuda)
        else:
            self.weighted_mse = None

        if args.use_grid_loss:
            self.sequential_grid = SequentialGridLoss(use_cuda=use_cuda)
        else:
            self.sequential_grid = None

        if args.use_reconstruction_loss:
            self.reconstruction = ReconstructionLoss(int(np.rint(args.input_width * (1 - args.crop_factor) / 16) * 16),
                                                     int(np.rint(args.input_height * (1 - args.crop_factor) / 16) * 16),
                                                     args.input_height,
                                                     use_cuda=use_cuda)
        else:
            self.reconstruction = None
            
        if args.use_siamese:
            self.siamese = WeightedMSELoss(use_cuda=use_cuda)
        else:
            self.siamese = None

    def forward(self, theta, theta_GT, img_R=None, img_R_orig=None):
        loss_parts = dict()
        loss = 0.0
        if self.weighted_mse is not None:
            weighted_mse_loss = self.weighted_mse(theta[:, :3], theta_GT)
            loss_parts['weighted_mse'] = weighted_mse_loss.clone().detach()
            loss += weighted_mse_loss
        if self.sequential_grid is not None:
            sequential_grid_loss = self.sequential_grid(theta[:, :3], theta_GT)
            loss_parts['sequential_grid'] = sequential_grid_loss.clone().detach()
            loss += sequential_grid_loss
        if self.reconstruction is not None:
            reconstruction_loss = self.reconstruction(theta[:, :3], img_R, img_R_orig)
            loss_parts['reconstruction'] = reconstruction_loss.clone().detach()
            loss += reconstruction_loss
        if self.siamese is not None:
            siamese_loss = self.siamese(theta[:, :3], theta[:, 3:])
            loss_parts['siamese'] = siamese_loss.clone().detach()
            loss += siamese_loss

        return loss, loss_parts
    