# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

# !/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
from tqdm import tqdm


def calculate_lfd_distance(
        q8_table, align_10, src_ArtCoeff, src_FdCoeff_q8, src_CirCoeff_q8, src_EccCoeff_q8,
        tgt_ArtCoeff, tgt_FdCoeff_q8, tgt_CirCoeff_q8, tgt_EccCoeff_q8):
    with torch.no_grad():
        src_ArtCoeff = src_ArtCoeff.unsqueeze(dim=1).unsqueeze(dim=1).expand(-1, 10, 10, -1, -1, -1)
        tgt_ArtCoeff = tgt_ArtCoeff.unsqueeze(dim=3).unsqueeze(dim=3).expand(-1, -1, -1, 10, 10, -1)
        art_distance = q8_table[src_ArtCoeff.reshape(-1).long(), tgt_ArtCoeff.reshape(-1).long()]
        art_distance = art_distance.reshape(
            src_ArtCoeff.shape[0], src_ArtCoeff.shape[1], src_ArtCoeff.shape[2],
            src_ArtCoeff.shape[3],
            src_ArtCoeff.shape[4], src_ArtCoeff.shape[5])
        art_distance = torch.sum(art_distance, dim=-1)

        src_FdCoeff_q8 = src_FdCoeff_q8.unsqueeze(dim=1).unsqueeze(dim=1).expand(-1, 10, 10, -1, -1, -1)
        tgt_FdCoeff_q8 = tgt_FdCoeff_q8.unsqueeze(dim=3).unsqueeze(dim=3).expand(-1, -1, -1, 10, 10, -1)
        fd_distance = q8_table[src_FdCoeff_q8.reshape(-1).long(), tgt_FdCoeff_q8.reshape(-1).long()]
        fd_distance = fd_distance.reshape(
            src_FdCoeff_q8.shape[0], src_FdCoeff_q8.shape[1], src_FdCoeff_q8.shape[2],
            src_FdCoeff_q8.shape[3], src_FdCoeff_q8.shape[4], src_FdCoeff_q8.shape[5])
        fd_distance = torch.sum(fd_distance, dim=-1) * 2.0

        src_CirCoeff_q8 = src_CirCoeff_q8.unsqueeze(dim=1).unsqueeze(dim=1).expand(-1, 10, 10, -1, -1)
        tgt_CirCoeff_q8 = tgt_CirCoeff_q8.unsqueeze(dim=3).unsqueeze(dim=3).expand(-1, -1, -1, 10, 10)
        cir_distance = q8_table[src_CirCoeff_q8.reshape(-1).long(), tgt_CirCoeff_q8.reshape(-1).long()]
        cir_distance = cir_distance.reshape(
            src_CirCoeff_q8.shape[0], src_CirCoeff_q8.shape[1],
            src_CirCoeff_q8.shape[2],
            src_CirCoeff_q8.shape[3], src_CirCoeff_q8.shape[4])
        cir_distance = cir_distance * 2.0
        src_EccCoeff_q8 = src_EccCoeff_q8.unsqueeze(dim=1).unsqueeze(dim=1).expand(-1, 10, 10, -1, -1)
        tgt_EccCoeff_q8 = tgt_EccCoeff_q8.unsqueeze(dim=3).unsqueeze(dim=3).expand(-1, -1, -1, 10, 10)
        ecc_distance = q8_table[src_EccCoeff_q8.reshape(-1).long(), tgt_EccCoeff_q8.reshape(-1).long()]
        ecc_distance = ecc_distance.reshape(
            src_EccCoeff_q8.shape[0], src_EccCoeff_q8.shape[1],
            src_EccCoeff_q8.shape[2], src_EccCoeff_q8.shape[3],
            src_EccCoeff_q8.shape[4])
        cost = art_distance + fd_distance + cir_distance + ecc_distance
        # find the cloest matching
        # cost shape: batch_size x src_camera x src_angle x dst_camera x dst_angle
        cost = cost.permute(0, 1, 3, 2, 4).long()
        align_n = align_10[:, :10].reshape(-1)
        cost_bxsrc_cxdst_cxsrc_axdst_a = cost
        align_err = torch.gather(
            input=cost_bxsrc_cxdst_cxsrc_axdst_a,
            index=align_n.reshape(1, 1, 1, 60 * 10, 1).expand(
                cost.shape[0], cost.shape[1],
                cost.shape[2], 60 * 10, 10).long(),
            dim=3)
        align_err = align_err.reshape(cost.shape[0], cost.shape[1], cost.shape[2], 60, 10, 10)
        sum_diag = 0
        for i in range(10):
            sum_diag += align_err[:, :, :, :, i, i]
        sum_diag = sum_diag.reshape(cost.shape[0], -1)
        dist = torch.min(sum_diag, dim=-1)[0]
    return dist


class LightFieldDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx, q8_table, align_10, src_ArtCoeff, src_FdCoeff_q8, src_CirCoeff_q8, src_EccCoeff_q8,
            tgt_ArtCoeff, tgt_FdCoeff_q8, tgt_CirCoeff_q8, tgt_EccCoeff_q8):
        n = src_ArtCoeff.shape[0]
        m = tgt_ArtCoeff.shape[0]
        ##############
        # This is only calculating one pair of distance

        all_dist = []
        with torch.no_grad():
            for i in tqdm(range(n)):
                start_idx = 0
                n_all_run = tgt_ArtCoeff.shape[0]
                n_each_run = 1000
                one_run_d = []
                while start_idx < n_all_run:
                    end_idx = min(n_all_run, start_idx + n_each_run)
                    run_length = end_idx - start_idx
                    d = calculate_lfd_distance(
                        q8_table, align_10,
                        src_ArtCoeff[i:i + 1].expand(run_length, -1, -1, -1),
                        src_FdCoeff_q8[i:i + 1].expand(run_length, -1, -1, -1),
                        src_CirCoeff_q8[i:i + 1].expand(run_length, -1, -1),
                        src_EccCoeff_q8[i:i + 1].expand(run_length, -1, -1),
                        tgt_ArtCoeff[start_idx:end_idx],
                        tgt_FdCoeff_q8[start_idx:end_idx],
                        tgt_CirCoeff_q8[start_idx:end_idx],
                        tgt_EccCoeff_q8[start_idx:end_idx])
                    start_idx += end_idx
                    one_run_d.append(d)
                d = torch.cat(one_run_d, dim=0)
                all_dist.append(d.unsqueeze(dim=0))
        dist = torch.cat(all_dist, dim=0)

        return dist

    @staticmethod
    def backward(ctx, graddist):
        raise NotImplementedError
        return None, None, None, None, None, None, None, None, None, None


class LFD(torch.nn.Module):
    def forward(
            self, q8_table, align_10, src_ArtCoeff, src_FdCoeff_q8, src_CirCoeff_q8, src_EccCoeff_q8,
            tgt_ArtCoeff, tgt_FdCoeff_q8, tgt_CirCoeff_q8, tgt_EccCoeff_q8):
        return LightFieldDistanceFunction.apply(
            q8_table, align_10, src_ArtCoeff, src_FdCoeff_q8, src_CirCoeff_q8, src_EccCoeff_q8,
            tgt_ArtCoeff, tgt_FdCoeff_q8, tgt_CirCoeff_q8, tgt_EccCoeff_q8)
