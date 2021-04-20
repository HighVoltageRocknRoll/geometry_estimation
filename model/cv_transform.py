from __future__ import print_function, division
import numpy as np
import torch


def get_cv_matrix(center, angle, scale, shift):
    angle = angle / 180.0 * np.pi
    alpha = scale * torch.cos(angle)
    beta = scale * torch.sin(angle)
    t_x = (1.0 - alpha) * center[0] - beta * center[1]
    t_y = beta * center[0] + (1.0 - alpha) * center[1] + shift

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
