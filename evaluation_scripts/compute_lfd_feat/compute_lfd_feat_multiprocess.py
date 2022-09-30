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
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool
import kaolin as kal
import point_cloud_utils as pcu


def seed_everything(seed):
    if seed < 0:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_mesh_v(mesh_name, normalized_scale=0.9):
    if mesh_name.endswith('obj'):
        mesh_1 = kal.io.obj.import_mesh(mesh_name)
        vertices = mesh_1.vertices.cpu().numpy()
        mesh_f1 = mesh_1.faces.cpu().numpy()
    elif mesh_name.endswith('ply'):
        vertices, mesh_f1 = pcu.load_mesh_vf(mesh_name)
    else:
        raise NotImplementedError

    if vertices.shape[0] == 0:
        return None, None

    scale = (vertices.max(axis=0) - vertices.min(axis=0)).max()
    mesh_v1 = vertices / scale * normalized_scale
    return mesh_v1, mesh_f1


from lfd_me import MeshEncoder
from functools import partial


def align_mesh_feature(mesh_name, align_feature_sample_folder):
    mesh_fodler = mesh_name.split('/')[-3:]
    mesh_fodler[-1] = mesh_fodler[-1].split('.')[0]
    mesh_fodler = '/'.join(mesh_fodler)
    mesh_fodler = os.path.join(align_feature_sample_folder, mesh_fodler)
    if not os.path.exists(mesh_fodler):
        os.makedirs(mesh_fodler)
    if os.path.exists(os.path.join(mesh_fodler, 'mesh_q4_v1.8.art')) and os.path.getsize(
            os.path.join(mesh_fodler, 'mesh_q4_v1.8.art')) > 1000:
        temp_dir_path = Path(mesh_fodler)
        file_name = 'mesh'
        temp_path = temp_dir_path / "{}.obj".format(file_name)
        path = temp_path.with_suffix("").as_posix()
        return path

    mesh_v, mesh_f = load_mesh_v(mesh_name, normalized_scale=1.0)
    if mesh_v is None:
        return None  # No face here

    mesh = MeshEncoder(mesh_v, mesh_f, folder=mesh_fodler, file_name='mesh', )
    mesh.align_mesh()
    return mesh.get_path()


def compute_lfd_feture(sample_pcs, n_process, save_path):
    align_feature_sample_folder = save_path
    os.makedirs(align_feature_sample_folder, exist_ok=True)
    print('==> one model')
    align_mesh_feature(sample_pcs[0], align_feature_sample_folder)
    N_process = n_process
    path_list = []
    if n_process == 0:
        for i in tqdm(range(len(sample_pcs))):
            align_mesh_feature(sample_pcs[i], align_feature_sample_folder)
        exit()
    print('==> multi process')
    pool = Pool(N_process)
    for x in tqdm(
            pool.imap_unordered(partial(align_mesh_feature, align_feature_sample_folder=align_feature_sample_folder), sample_pcs),
            total=len(sample_pcs)):
        path_list.append(x)
    pool.close()
    pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_path", type=str, required=True, help="path to the generated models")
    parser.add_argument("--save_path", type=str, required=True, help="path to save the generated features for each model")
    parser.add_argument("--n_models", type=int, default=-1, help="Number of models used for evaluation")
    parser.add_argument("--n_process", type=int, default=32, help="Number of process used for evaluation")

    args = parser.parse_args()
    path_list = [

    ]
    n_model_list = [

    ]

    if args.gen_path == 'all':
        models = []
        assert len(path_list) == len(n_model_list)
        for path, n_model in zip(path_list, n_model_list):
            objs = os.listdir(path)
            objs = sorted(objs)
            objs = [os.path.join(path, obj) for obj in objs if obj.endswith('obj')]
            select_obj = objs[:n_model]
            models.extend(select_obj)
    else:
        models = sorted(os.listdir(args.gen_path))
        if 'gt_shape' in args.gen_path:
            if 'animal' in args.gen_path:
                models = [os.path.join(args.gen_path, m, m + '.obj') for m in models if 'json' not in m]
            else:
                models = [os.path.join(args.gen_path, m, 'model.obj') for m in models]
        else:
            models = [os.path.join(args.gen_path, m) for m in models if m.endswith('obj')]
        if args.n_models != -1:
            models = models[:args.n_models]
    compute_lfd_feture(models, args.n_process, args.save_path)
