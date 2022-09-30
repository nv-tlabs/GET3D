# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import argparse
import numpy as np
import torch
import os
import random
import glob
from tqdm import tqdm
import kaolin as kal
import point_cloud_utils as pcu


def seed_everything(seed):
    if seed < 0:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def sample_point_with_mesh_name(name, n_sample=2048, normalized_scale=1.0):
    if 'npz' in name:
        pcd = np.load(name)['pcd']
        if pcd.shape[0] != 1:  # The first dimension is 1
            pcd = pcd[np.newaxis, :, :]
        pcd = pcd[:, :n_sample, :]
        return torch.from_numpy(pcd).float().cuda()

    if 'ply' in name:
        v = pcu.load_mesh_v(name)
        point_clouds = np.random.permutation(v)[:n_sample, :]
        scale = 0.9
        if 'chair' in name:
            scale = 0.7
        if 'animal' in name:
            scale = 0.7
        if 'car' in name:
            normalized_scale = 0.9  # We sample the car using 0.9 surface
        point_clouds = point_clouds / scale * normalized_scale  # Make them in the same scale
        return torch.from_numpy(point_clouds).float().cuda().unsqueeze(dim=0)

    mesh_1 = kal.io.obj.import_mesh(name)
    if mesh_1.vertices.shape[0] == 0:
        return None
    vertices = mesh_1.vertices.cuda()
    scale = (vertices.max(dim=0)[0] - vertices.min(dim=0)[0]).max()
    mesh_v1 = vertices / scale * normalized_scale
    mesh_f1 = mesh_1.faces.cuda()
    points, _ = kal.ops.mesh.sample_points(mesh_v1.unsqueeze(dim=0), mesh_f1, n_sample)
    return points.cuda()


def chamfer_distance(ref_pcs, sample_pcs, batch_size):
    all_rec_pcs = []
    n_sample = 2048
    normalized_scale = 1.0
    for name in tqdm(ref_pcs):
        all_rec_pcs.append(sample_point_with_mesh_name(name, n_sample, normalized_scale=normalized_scale))
    all_sample_pcs = []
    for name in tqdm(sample_pcs):
        # This is generated
        all_sample_pcs.append(sample_point_with_mesh_name(name, n_sample, normalized_scale=normalized_scale))
    all_rec_pcs = [p for p in all_rec_pcs if p is not None]
    all_sample_pcs = [p for p in all_sample_pcs if p is not None]
    all_rec_pcs = torch.cat(all_rec_pcs, dim=0)
    all_sample_pcs = torch.cat(all_sample_pcs, dim=0)
    all_cd = []
    for i_ref_p in tqdm(range(len(ref_pcs))):
        ref_p = all_rec_pcs[i_ref_p]
        cd_lst = []
        for sample_b_start in range(0, len(sample_pcs), batch_size):
            sample_b_end = min(len(sample_pcs), sample_b_start + batch_size)
            sample_batch = all_sample_pcs[sample_b_start:sample_b_end]

            batch_size_sample = sample_batch.size(0)
            chamfer = kal.metrics.pointcloud.chamfer_distance(
                ref_p.unsqueeze(dim=0).expand(batch_size_sample, -1, -1),
                sample_batch)
            cd_lst.append(chamfer)
        cd_lst = torch.cat(cd_lst, dim=0)
        all_cd.append(cd_lst.unsqueeze(dim=0))
    all_cd = torch.cat(all_cd, dim=0)
    return all_cd


def compute_all_metrics(sample_pcs, ref_pcs, batch_size, save_name=None):
    M_rs_cd = chamfer_distance(ref_pcs, sample_pcs, batch_size)
    import pickle
    pickle.dump(M_rs_cd.data.cpu().numpy(), open(save_name, 'wb'))


def evaluate(args):
    # Set the random seed
    seed_everything(41)

    with open(args.split_path) as f:
        split_models = f.readlines()
        split_models = [model.rstrip() for model in split_models]
    if 'animal' in args.dataset_path:
        ref_path = [os.path.join(args.dataset_path, s, s + '.obj') for s in split_models]
    elif '02958343_surface_pcd_all' in args.dataset_path:
        ref_path = [os.path.join(args.dataset_path, s + '.npz') for s in split_models]  ###
    else:
        ref_path = [os.path.join(args.dataset_path, s, 'model.obj') for s in split_models]

    gen_path = args.gen_path
    if args.use_npz:
        gen_models = glob.glob(os.path.join(gen_path, '*.npz'))
    else:
        gen_models = glob.glob(os.path.join(gen_path, '*.obj'))
    gen_models = sorted(gen_models)

    ref_path = ref_path[:7500]
    gen_models = gen_models[:args.n_shape]
    with torch.no_grad():
        compute_all_metrics(gen_models, ref_path, args.batch_size, args.save_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_name", type=str, required=True, help="path to the save results")
    parser.add_argument("--split_path", type=str, required=True, help="path to the split shapenet dataset")
    parser.add_argument("--dataset_path", type=str, required=True, help="path to the original shapenet dataset")
    parser.add_argument("--gen_path", type=str, required=True, help="path to the generated models")
    parser.add_argument("--n_points", type=int, default=2048, help="Number of points used for evaluation")
    parser.add_argument("--batch_size", type=int, default=50, help="batch size to compute chamfer distance")
    parser.add_argument("--n_shape", type=int, default=7500, help="number of shapes for evaluations")
    parser.add_argument("--use_npz", type=bool, default=False, help="whether the generated shape is npz or not")
    args = parser.parse_args()
    evaluate(args)
