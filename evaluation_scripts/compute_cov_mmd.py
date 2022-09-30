# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import os
import pickle
import numpy as np


def get_score(results, use_same_numer_for_test=False):
    if use_same_numer_for_test:
        results = results[:, :results.shape[0]]
    mmd = results.min(axis=1).mean()
    min_ref = results.argmin(axis=0)
    unique_idx = np.unique(min_ref)
    cov = float(len(unique_idx)) / results.shape[0]
    if mmd < 1:
        # Chamfer distance
        mmd = mmd * 1000  # for showing results
    return mmd, cov * 100


def get_one_result(file_name, filter_idx=None, transpose=False):
    results = pickle.load(open(file_name, 'rb'))
    if 'lfd' in file_name:
        transpose = True
    if transpose:
        results = results.T
    if filter_idx is not None:
        results = results[filter_idx, :]
    results = results[:, :results.shape[0] * 5]  # Generation is 5 time of the testing set
    mmd, cov = get_score(results, use_same_numer_for_test=False)
    print('MMD: %.2f, COV: %2.2f' % (mmd, cov))


def get_score_one_line(cd_name, lfd_name, filter_idx=None):
    results = pickle.load(open(cd_name, 'rb'))
    if filter_idx is not None:
        results = results[filter_idx, :]
    results = results[:, :results.shape[0] * 5]  # Generation is 5 time of the testing set
    cd_mmd, cd_cov = get_score(results, use_same_numer_for_test=False)
    results = pickle.load(open(lfd_name, 'rb'))
    results = results.T
    if filter_idx is not None:
        results = results[filter_idx, :]
    results = results[:, :results.shape[0] * 5]  # Generation is 5 time of the testing set
    lfd_mmd, lfd_cov = get_score(results, use_same_numer_for_test=False)
    print('==> %s' % (cd_name))
    print('%2.2f & %2.2f & %d & %.2f' % (lfd_cov, cd_cov, lfd_mmd, cd_mmd))


if __name__ == '__main__':
    get_score_one_line('results/our/cd.pkl', 'results/our/lfd.pkl', None)
