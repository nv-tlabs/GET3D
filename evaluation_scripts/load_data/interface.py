# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import os
import ctypes as c
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))


class LoadData:
    def __init__(self):
        self.lib = c.cdll.LoadLibrary(os.path.join(dir_path, 'run.so'))
        self.lib.run.argtypes = [c.POINTER(c.c_ubyte), c.POINTER(c.c_ubyte), c.c_char_p, c.POINTER(c.c_ubyte),
                                 c.POINTER(c.c_ubyte), c.POINTER(c.c_ubyte), c.POINTER(c.c_ubyte)]

    def run(self, file_name, normalize=False):
        q8_table = np.zeros(256 * 256, dtype=np.ubyte)
        align10 = np.zeros(60 * 20, dtype=np.ubyte)
        dest_ArtCoeff = np.zeros(10 * 10 * 35, dtype=np.ubyte)
        dest_FdCoeff_q8 = np.zeros(10 * 10 * 10, dtype=np.ubyte)
        dest_CirCoeff_q8 = np.zeros(10 * 10, dtype=np.ubyte)
        dest_EccCoeff_q8 = np.zeros(10 * 10, dtype=np.ubyte)

        q8_table_p = np.ascontiguousarray(q8_table).ctypes.data_as(c.POINTER(c.c_ubyte))
        align10_p = np.ascontiguousarray(align10).ctypes.data_as(c.POINTER(c.c_ubyte))
        dest_ArtCoeff_p = np.ascontiguousarray(dest_ArtCoeff).ctypes.data_as(c.POINTER(c.c_ubyte))
        dest_FdCoeff_q8_p = np.ascontiguousarray(dest_FdCoeff_q8).ctypes.data_as(c.POINTER(c.c_ubyte))
        dest_CirCoeff_q8_p = np.ascontiguousarray(dest_CirCoeff_q8).ctypes.data_as(c.POINTER(c.c_ubyte))
        dest_EccCoeff_q8_p = np.ascontiguousarray(dest_EccCoeff_q8).ctypes.data_as(c.POINTER(c.c_ubyte))
        file_name_p = file_name.encode('utf-8')
        self.lib.run(q8_table_p, align10_p, file_name_p, dest_ArtCoeff_p, dest_FdCoeff_q8_p, dest_CirCoeff_q8_p, dest_EccCoeff_q8_p)
        return q8_table, align10, dest_ArtCoeff, dest_FdCoeff_q8, dest_CirCoeff_q8, dest_EccCoeff_q8
