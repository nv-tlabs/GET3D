# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import math
import numpy as np
from training.math_utils_torch import *


def create_camera_from_angle(phi, theta, sample_r, device='cuda'):
    '''
    :param phi: rotation angle of the camera
    :param theta:  rotation angle of the camera
    :param sample_r: distance from the camera to the origin
    :param device:
    :return:
    '''
    phi = torch.clamp(phi, 1e-5, math.pi - 1e-5)

    camera_origin = torch.zeros((phi.shape[0], 3), device=device)
    camera_origin[:, 0:1] = sample_r * torch.sin(phi) * torch.cos(theta)
    camera_origin[:, 2:3] = sample_r * torch.sin(phi) * torch.sin(theta)
    camera_origin[:, 1:2] = sample_r * torch.cos(phi)

    forward_vector = normalize_vecs(camera_origin)

    world2cam_matrix = create_my_world2cam_matrix(forward_vector, camera_origin, device=device)
    return world2cam_matrix, forward_vector, camera_origin, phi, theta


def sample_camera(camera_data_mode, n, device='cuda'):
    # We use this function to sample the camera for training the generator
    # When we're rendering the dataset, the camera is random sampled from
    # a uniform distribution (see `render_shapenet_data/render_shapenet.py`
    # for details of camera distribution)
    if camera_data_mode == 'shapenet_car' or camera_data_mode == 'shapenet_chair' or camera_data_mode == 'shapenet_motorbike' \
            or camera_data_mode == 'ts_house':
        horizontal_stddev = math.pi  # here means horizontal rotation
        vertical_stddev = (math.pi / 180) * 15
        horizontal_mean = math.pi  ######## [horizon range [0, 2pi]]
        vertical_mean = (math.pi / 180) * 75
        mode = 'uniform'
        radius_range = [1.2, 1.2]
    elif camera_data_mode == 'ts_animal':
        horizontal_stddev = math.pi  # here means horizontal rotation
        vertical_stddev = (math.pi / 180) * (45.0 / 2.0)

        horizontal_mean = math.pi  ######## [horizon range [0, 2pi]]
        vertical_mean = (math.pi / 180) * ((90.0 + 90.0 - 45.0) / 2.0)
        mode = 'uniform'
        radius_range = [1.2, 1.2]
    elif camera_data_mode == 'renderpeople':
        horizontal_stddev = math.pi  # here means horizontal rotation
        vertical_stddev = (math.pi / 180) * 10

        horizontal_mean = math.pi  ######## [horizon range [0, 2pi]]
        vertical_mean = (math.pi / 180) * 80
        mode = 'uniform'
        radius_range = [1.2, 1.2]

    else:
        raise NotImplementedError

    camera_origin, rotation_angle, elevation_angle = sample_camera_positions(
        device, n=n, r=radius_range,
        horizontal_stddev=horizontal_stddev,
        vertical_stddev=vertical_stddev,
        horizontal_mean=horizontal_mean,
        vertical_mean=vertical_mean, mode=mode,
    )

    forward_vector = normalize_vecs(camera_origin)
    # Camera is always looking at the Origin point
    world2cam_matrix = create_my_world2cam_matrix(forward_vector, camera_origin, device=device)
    return world2cam_matrix, forward_vector, camera_origin, rotation_angle, elevation_angle


def sample_camera_positions(
        device, n=1, r=[1.0, 1.0], horizontal_stddev=1.0, vertical_stddev=1.0,
        horizontal_mean=math.pi * 0.5,
        vertical_mean=math.pi * 0.5, mode='normal'):
    """
    Samples n random locations along a sphere of radius r. Uses the specified distribution.
    Theta is yaw in radians (-pi, pi)
    Phi is pitch in radians (0, pi)
    """
    if mode == 'uniform':
        theta = (torch.rand((n, 1), device=device) - 0.5) * 2 * horizontal_stddev + horizontal_mean
        phi = (torch.rand((n, 1), device=device) - 0.5) * 2 * vertical_stddev + vertical_mean

    elif mode == 'spherical_uniform':
        theta = (torch.rand((n, 1), device=device) - .5) * 2 * horizontal_stddev + horizontal_mean
        v_stddev, v_mean = vertical_stddev / math.pi, vertical_mean / math.pi
        v = ((torch.rand((n, 1), device=device) - .5) * 2 * v_stddev + v_mean)
        v = torch.clamp(v, 1e-5, 1 - 1e-5)
        phi = torch.arccos(1 - 2 * v)
    else:
        raise NotImplementedError
    phi = torch.clamp(phi, 1e-5, math.pi - 1e-5)

    output_points = torch.zeros((n, 3), device=device)
    sample_r = torch.rand((n, 1), device=device)
    sample_r = sample_r * r[0] + (1 - sample_r) * r[1]

    compute_theta = -theta - 0.5 * math.pi

    output_points[:, 0:1] = sample_r * torch.sin(phi) * torch.cos(compute_theta)
    output_points[:, 2:3] = sample_r * torch.sin(phi) * torch.sin(compute_theta)
    output_points[:, 1:2] = sample_r * torch.cos(phi)
    rotation_angle = theta
    elevation_angle = phi
    return output_points, rotation_angle, elevation_angle


def lookAt(eye, at, up):
    a = eye - at
    b = up
    w = normalize_vecs(a)
    u = torch.cross(b, w, dim=-1)

    u = normalize_vecs(u)
    v = torch.cross(w, u, dim=-1)

    translate = np.array(
        [[1, 0, 0, -eye[0]],
         [0, 1, 0, -eye[1]],
         [0, 0, 1, -eye[2]],
         [0, 0, 0, 1]]).astype(np.float32)
    rotate = np.array(
        [[u[0], u[1], u[2], 0],
         [v[0], v[1], v[2], 0],
         [w[0], w[1], w[2], 0],
         [0, 0, 0, 1]]).astype(np.float32)
    return np.matmul(rotate, translate)


def create_my_world2cam_matrix(forward_vector, origin, device=None):
    """Takes in the direction the camera is pointing and the camera origin and returns a cam2world matrix."""

    forward_vector = normalize_vecs(forward_vector)
    up_vector = torch.tensor([0, 1, 0], dtype=torch.float, device=device).expand_as(forward_vector)

    left_vector = normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))

    up_vector = normalize_vecs(torch.cross(forward_vector, left_vector, dim=-1))

    new_t = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    new_t[:, :3, 3] = -origin
    new_r = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    new_r[:, :3, :3] = torch.cat(
        (left_vector.unsqueeze(dim=1), up_vector.unsqueeze(dim=1), forward_vector.unsqueeze(dim=1)), dim=1)
    world2cam = new_r @ new_t
    return world2cam


def create_cam2world_matrix(forward_vector, origin, device=None):
    """Takes in the direction the camera is pointing and the camera origin and returns a cam2world matrix."""

    forward_vector = normalize_vecs(forward_vector)
    up_vector = torch.tensor([0, 1, 0], dtype=torch.float, device=device).expand_as(forward_vector)

    left_vector = normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))

    up_vector = normalize_vecs(torch.cross(forward_vector, left_vector, dim=-1))

    rotation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = torch.stack((-left_vector, up_vector, -forward_vector), dim=-1)
    # rotation_matrix[:, :3, :3] = torch.stack((left_vector, up_vector, forward_vector), dim=-1)

    translation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = origin

    cam2world = translation_matrix @ rotation_matrix
    return cam2world


def create_world2cam_matrix(forward_vector, origin, device):
    """Takes in the direction the camera is pointing and the camera origin and returns a world2cam matrix."""
    cam2world = create_cam2world_matrix(forward_vector, origin, device=device)
    world2cam = torch.inverse(cam2world)
    return world2cam
