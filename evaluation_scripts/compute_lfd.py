# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import numpy as np
import torch
import os
from tqdm import tqdm
from load_data.interface import LoadData


def read_all_data(folder_list, load_data, add_model_str=True, add_ori_name=False):
    all_data = []

    for f in tqdm(folder_list):
        if add_model_str:
            result = load_data.run(os.path.join(f, 'model', 'mesh'))
        elif add_ori_name:
            result = load_data.run(os.path.join(f, f.split('/')[-1], 'mesh'))
        else:
            result = load_data.run(os.path.join(f, 'mesh'))

        all_data.append(result)
    q8_table = all_data[0][0]
    align_10 = all_data[0][1]
    dest_ArtCoeff = [r[2][np.newaxis, :] for r in all_data]
    dest_FdCoeff_q8 = [r[3][np.newaxis, :] for r in all_data]
    dest_CirCoeff_q8 = [r[4][np.newaxis, :] for r in all_data]
    dest_EccCoeff_q8 = [r[5][np.newaxis, :] for r in all_data]
    SRC_ANGLE = 10
    ANGLE = 10
    CAMNUM = 10
    ART_COEF = 35
    FD_COEF = 10
    n_shape = len(all_data)
    dest_ArtCoeff = torch.from_numpy(np.ascontiguousarray(np.concatenate(dest_ArtCoeff, axis=0))).int().cuda().reshape(n_shape, SRC_ANGLE, CAMNUM, ART_COEF)
    dest_FdCoeff_q8 = torch.from_numpy(np.ascontiguousarray(np.concatenate(dest_FdCoeff_q8, axis=0))).int().cuda().reshape(n_shape, ANGLE, CAMNUM, FD_COEF)
    dest_CirCoeff_q8 = torch.from_numpy(np.ascontiguousarray(np.concatenate(dest_CirCoeff_q8, axis=0))).int().cuda().reshape(n_shape, ANGLE, CAMNUM)
    dest_EccCoeff_q8 = torch.from_numpy(np.ascontiguousarray(np.concatenate(dest_EccCoeff_q8, axis=0))).int().cuda().reshape(n_shape, ANGLE, CAMNUM)
    q8_table = torch.from_numpy(np.ascontiguousarray(q8_table)).int().cuda().reshape(256, 256)
    align_10 = torch.from_numpy(np.ascontiguousarray(align_10)).int().cuda().reshape(60, 20)  ##
    return q8_table.contiguous(), align_10.contiguous(), dest_ArtCoeff.contiguous(), \
           dest_FdCoeff_q8.contiguous(), dest_CirCoeff_q8.contiguous(), dest_EccCoeff_q8.contiguous()


def compute_lfd_all(src_dir, tgt_dir, split_path, debug=False, save_name=None):
    load_data = LoadData()
    src_folder_list = sorted(os.listdir(src_dir))
    tgt_folder_list = sorted(os.listdir(tgt_dir))
    src_folder_list = [os.path.join(src_dir, f) for f in src_folder_list]
    tgt_folder_list = [os.path.join(tgt_dir, f) for f in tgt_folder_list]
    if split_path is not None:
        with open(args.split_path) as f:
            split_models = f.readlines()
            split_models = [model.rstrip() for model in split_models]
        # new_tgt_folder_list = []
        tgt_folder_list = [os.path.join(tgt_dir, f) for f in split_models]
    if debug:
        import ipdb
        ipdb.set_trace()
        import pickle
        src_folder_list = src_folder_list[:100]
        tgt_folder_list = tgt_folder_list[:100]

    add_ori_name = False
    add_model_str = True

    q8_table, align_10, src_ArtCoeff, src_FdCoeff_q8, src_CirCoeff_q8, src_EccCoeff_q8 = \
        read_all_data(src_folder_list, load_data, add_model_str=False)
    q8_table, align_10, tgt_ArtCoeff, tgt_FdCoeff_q8, tgt_CirCoeff_q8, tgt_EccCoeff_q8 = \
        read_all_data(tgt_folder_list, load_data, add_model_str=add_model_str, add_ori_name=add_ori_name)  ###

    from lfd_all_compute.lfd import LFD
    lfd = LFD()
    lfd_matrix = lfd.forward(
        q8_table, align_10, src_ArtCoeff, src_FdCoeff_q8, src_CirCoeff_q8, src_EccCoeff_q8,
        tgt_ArtCoeff, tgt_FdCoeff_q8, tgt_CirCoeff_q8, tgt_EccCoeff_q8)
    print(lfd_matrix)
    mmd = lfd_matrix.float().min(dim=0)[0].mean()
    mmd_swp = lfd_matrix.float().min(dim=1)[0].mean()
    print(mmd)
    print(mmd_swp)
    import pickle
    if save_name is None:
        save_name = 'tmp.pkl'
    pickle.dump(lfd_matrix.data.cpu().numpy(), open(save_name, 'wb'))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_name", type=str, required=True, help="path to the save resules shapenet dataset")
    parser.add_argument("--split_path", type=str, required=True, help="path to the split shapenet dataset")
    parser.add_argument("--dataset_path", type=str, required=True, help="path to the preprocessed shapenet dataset")
    parser.add_argument("--gen_path", type=str, required=True, help="path to the generated models")
    args = parser.parse_args()
    save_path = '/'.join(args.save_name.split('/')[:-1])
    os.makedirs(save_path, exist_ok=True)
    compute_lfd_all(
        args.gen_path,
        args.dataset_path,
        args.split_path,
        debug=False,
        save_name=args.save_name)
