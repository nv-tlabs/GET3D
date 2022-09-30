# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import math
import numpy as np
import os
import argparse
import multiprocessing as mp
from multiprocessing import Pool
import trimesh
import tqdm
import torch
import nvdiffrast.torch as dr
import kaolin as kal

parser = argparse.ArgumentParser(description='sample surface points from mesh')
parser.add_argument(
    '--n_proc', type=int, default=0,
    help='Number of processes to run in parallel'
         '(0 means sequential execution).')
parser.add_argument(
    '--in_dir', type=str,
    help='Path to input directory.')
parser.add_argument(
    '--n_points', type=int, default=100000,
    help='Number of points to sample per model.')
parser.add_argument(
    '--n_views', type=int, default=100,
    help='Number of views per model.')
parser.add_argument(
    '--image_height', type=int, default=640,
    help='Depth image height.')
parser.add_argument(
    '--image_width', type=int, default=640,
    help='Depth image width.')
parser.add_argument(
    '--focal_length_x', type=float, default=640,
    help='Focal length in x direction.')
parser.add_argument(
    '--focal_length_y', type=float, default=640,
    help='Focal length in y direction.')
parser.add_argument(
    '--principal_point_x', type=float, default=320,
    help='Principal point location in x direction.')
parser.add_argument(
    '--principal_point_y', type=float, default=320,
    help='Principal point location in y direction.')
parser.add_argument("--shape_root", type=str, required=True, help="path to the save resules shapenet dataset")
parser.add_argument("--save_root", type=str, required=True, help="path to the split shapenet dataset")

options = parser.parse_args()

# create array for inverse mapping
coordspx2 = np.stack(np.nonzero(np.ones((options.image_height, options.image_width))), -1).astype(np.float32)
coordspx2 = coordspx2[:, ::-1]
fusion_intrisics = np.array(
    [
        [options.focal_length_x, 0, options.principal_point_x],
        [0, options.focal_length_y, options.principal_point_y],
        [0, 0, 1]
    ])
glctx = dr.RasterizeGLContext()


def CalcLinearZ(depth):
    # depth = depth * 2 - 1
    zFar = 100.0
    zNear = 0.1
    linear = zNear / (zFar - depth * (zFar - zNear)) * zFar
    return linear


def projection_cv_new(fx, fy, cx, cy, width, height, n=1.0, f=50.0):
    return np.array(
        [[-2 * fx / width, 0.0, (width - 2 * cx) / width, 0.0],
         [0.0, -2 * fy / height, (height - 2 * cy) / height, 0.0],
         [0.0, 0.0, (-f - n) / (f - n), -2.0 * f * n / (f - n)],
         [0.0, 0.0, -1.0, 0.0]])


def interpolate(attr, rast, attr_idx, rast_db=None):
    return dr.interpolate(
        attr.contiguous(), rast, attr_idx, rast_db=rast_db,
        diff_attrs=None if rast_db is None else 'all')


def render_nvdiffrast(v_pos, tris, T_bx4x4):
    # T_bx4x4 - world to cam
    proj = projection_cv_new(
        fx=options.focal_length_x, fy=options.focal_length_y, cx=options.principal_point_x,
        cy=options.principal_point_y,
        width=options.image_width, height=options.image_height, n=0.1, f=100.0)

    fix = torch.eye(4, dtype=torch.float32, device='cuda')
    fix[2, 2] = -1
    fix[1, 1] = -1
    fix[0, 0] = -1
    fix = fix.unsqueeze(0).repeat(T_bx4x4.shape[0], 1, 1)

    proj = torch.tensor(proj, dtype=torch.float32, device='cuda').unsqueeze(0).repeat(T_bx4x4.shape[0], 1, 1)
    T_world_cam_bx4x4 = torch.bmm(fix, T_bx4x4)
    mvp = torch.bmm(proj, T_world_cam_bx4x4)
    v_pos_clip = torch.matmul(
        torch.nn.functional.pad(v_pos, pad=(0, 1), mode='constant', value=1.0),
        torch.transpose(mvp, 1, 2))
    rast, db = dr.rasterize(
        glctx, torch.tensor(v_pos_clip, dtype=torch.float32, device='cuda'), tris.int(),
        (options.image_height, options.image_width))

    v_pos_cam = torch.matmul(
        torch.nn.functional.pad(v_pos, pad=(0, 1), mode='constant', value=1.0),
        torch.transpose(T_world_cam_bx4x4, 1, 2))
    gb_pos_cam, _ = interpolate(v_pos_cam, rast, tris.int())
    depth_maps = gb_pos_cam[..., 2].abs()
    return depth_maps


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(
                    trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert (isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh


def render(mesh_v, mesh_f, Rs):
    """
    Render the given mesh using the generated views.

    :param base_mesh: mesh to render
    :type base_mesh: mesh.Mesh
    :param Rs: rotation matrices
    :type Rs: [numpy.ndarray]
    :return: depth maps
    :rtype: numpy.ndarray
    """
    T_bx4x4 = torch.zeros((options.n_views, 4, 4), dtype=torch.float32, device='cuda')
    T_bx4x4[:, 3, 3] = 1
    T_bx4x4[:, 2, 3] = 1
    T_bx4x4[:, :3, :3] = torch.tensor(Rs, dtype=torch.float32, device='cuda')
    depthmaps = render_nvdiffrast(
        mesh_v,
        mesh_f, T_bx4x4)
    return depthmaps


def get_points():
    """
    :param n_points: number of points
    :type n_points: int
    :return: list of points
    :rtype: numpy.ndarray
    """

    rnd = 1.
    points = []
    offset = 2. / options.n_views
    increment = math.pi * (3. - math.sqrt(5.))

    for i in range(options.n_views):
        y = ((i * offset) - 1) + (offset / 2)
        r = math.sqrt(1 - pow(y, 2))

        phi = ((i + rnd) % options.n_views) * increment

        x = math.cos(phi) * r
        z = math.sin(phi) * r

        points.append([x, y, z])
    return np.array(points)


def get_views(semi_sphere=True):
    """
    Generate a set of views to generate depth maps from.

    :param n_views: number of views per axis
    :type n_views: int
    :return: rotation matrices
    :rtype: [numpy.ndarray]
    """

    Rs = []
    points = get_points()
    if semi_sphere:
        points[:, 2] = -np.abs(points[:, 2]) - 0.1

    for i in range(points.shape[0]):
        longitude = - math.atan2(points[i, 0], points[i, 1])
        latitude = math.atan2(points[i, 2], math.sqrt(points[i, 0] ** 2 + points[i, 1] ** 2))

        R_x = np.array(
            [[1, 0, 0],
             [0, math.cos(latitude), -math.sin(latitude)],
             [0, math.sin(latitude), math.cos(latitude)]])
        R_y = np.array(
            [[math.cos(longitude), 0, math.sin(longitude)],
             [0, 1, 0],
             [-math.sin(longitude), 0, math.cos(longitude)]])
        R = R_x @ R_y
        Rs.append(R)

    return Rs


def fusion(depthmaps, Rs):
    """
    Fuse the rendered depth maps.

    :param depthmaps: depth maps
    :type depthmaps: numpy.ndarray
    :param Rs: rotation matrices corresponding to views
    :type Rs: [numpy.ndarray]
    :return: (T)SDF
    :rtype: numpy.ndarray
    """

    # sample points inside mask
    sample_per_view = options.n_points // options.n_views
    sample_bxn = torch.zeros((options.n_views, sample_per_view), device='cuda', dtype=torch.long)
    for i in range(len(Rs)):
        mask = depthmaps[i] > 0
        valid_idx = torch.nonzero(mask.reshape(-1)).squeeze(-1)
        idx = list(range(valid_idx.shape[0]))
        np.random.shuffle(idx)
        idx = idx[:sample_per_view]
        sample_bxn[i] = torch.tensor(valid_idx[idx])

    depthmaps = torch.gather(depthmaps.reshape(options.n_views, -1), 1, sample_bxn)

    inv_Ks_bx3x3 = torch.tensor(np.linalg.inv(fusion_intrisics), dtype=torch.float32, device='cuda').unsqueeze(
        0).repeat(options.n_views, 1, 1)
    T_bx4x4 = torch.zeros((options.n_views, 4, 4), dtype=torch.float32, device='cuda')
    T_bx4x4[:, 3, 3] = 1
    T_bx4x4[:, 2, 3] = 1
    T_bx4x4[:, :3, :3] = torch.tensor(Rs, dtype=torch.float32, device='cuda')
    inv_T_bx4x4 = torch.inverse(T_bx4x4)

    tf_coords_bxpx2 = torch.tensor(coordspx2.copy(), dtype=torch.float32, device='cuda').unsqueeze(0).repeat(
        options.n_views, 1, 1)
    tf_coords_bxpx2 = torch.gather(tf_coords_bxpx2, 1, sample_bxn.unsqueeze(-1).repeat(1, 1, 2))

    tf_coords_bxpx3 = torch.cat([tf_coords_bxpx2, torch.ones_like(tf_coords_bxpx2[..., :1])], -1)
    tf_coords_bxpx3 *= depthmaps.reshape(options.n_views, -1, 1)
    tf_cam_bxpx3 = torch.bmm(inv_Ks_bx3x3, tf_coords_bxpx3.transpose(1, 2)).transpose(1, 2)
    tf_cam_bxpx4 = torch.cat([tf_cam_bxpx3, torch.ones_like(tf_cam_bxpx3[..., :1])], -1)
    tf_world_bxpx3 = torch.bmm(inv_T_bx4x4, tf_cam_bxpx4.transpose(1, 2)).transpose(1, 2)[..., :3]

    return tf_world_bxpx3.reshape(-1, 3)


def normalize(vertices, faces, normalized_scale=0.9):
    vertices = vertices.cuda()
    scale = (vertices.max(dim=0)[0] - vertices.min(dim=0)[0]).max()
    mesh_v1 = vertices / scale * normalized_scale
    mesh_f1 = faces.long().cuda()
    return mesh_v1, mesh_f1


def sample_surface_pts(path):
    mesh_path, output_pth, debug = path
    mesh = kal.io.obj.import_mesh(mesh_path)
    if mesh.vertices.shape[0] == 0:
        return
    mesh_v = mesh.vertices
    mesh_v, mesh_f = normalize(mesh_v, mesh.faces, normalized_scale=0.9)

    # generate camera matrices
    Rs = get_views()
    # get depth images
    depths = render(mesh_v, mesh_f, Rs)
    # project to world space
    try:
        pcd = fusion(depths, Rs)
    except:
        return
    pcd = pcd.cpu().numpy()

    np.savez(output_pth, pcd=pcd)
    if debug:
        pcd = trimesh.points.PointCloud(pcd)
        pcd.export(output_pth.replace('.npz', '.ply'))


if __name__ == '__main__':
    mp.set_start_method('spawn')
    shapenet_root = options.shape_root
    save_root = options.save_root

    debug = False
    model_list = sorted(os.listdir(shapenet_root))[:7500]
    os.makedirs(save_root, exist_ok=True)
    cmds = [(os.path.join(shapenet_root, id), os.path.join(save_root, id.replace('.obj', '.npz')), debug) for id in
            model_list]
    if options.n_proc == 0:
        for filepath in tqdm.tqdm(cmds):
            sample_surface_pts(filepath)
    else:
        with Pool(options.n_proc) as p:
            list(tqdm.tqdm(p.imap(sample_surface_pts, cmds), total=len(cmds)))
