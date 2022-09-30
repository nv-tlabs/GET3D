# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
'''
Function is modified based on https://github.com/kacperkan/light-field-distance
'''
import argparse
import sys
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh

SIMILARITY_TAG = b"SIMILARITY:"
CURRENT_DIR = Path(__file__).parent
CURRENT_DIR = '/opt/conda/lib/python3.8/site-packages/light_field_distance-0.0.9-py3.8.egg/lfd/Executable'

GENERATED_FILES_NAMES = [
    "all_q4_v1.8.art",
    "all_q8_v1.8.art",
    "all_q8_v1.8.cir",
    "all_q8_v1.8.ecc",
    "all_q8_v1.8.fd",
]

OUTPUT_NAME_TEMPLATES = [
    "{}_q4_v1.8.art",
    "{}_q8_v1.8.art",
    "{}_q8_v1.8.cir",
    "{}_q8_v1.8.ecc",
    "{}_q8_v1.8.fd",
]


class MeshEncoder:
    """Class holding an object and preprocessing it using an external cmd."""

    def __init__(self, vertices: np.ndarray, triangles: np.ndarray, folder=None, file_name=None):
        """Instantiate the class.

        It instantiates an empty, temporary folder that will hold any
        intermediate data necessary to calculate Light Field Distance.

        Args:
            vertices: np.ndarray of vertices consisting of 3 coordinates each.
            triangles: np.ndarray where each entry is a vector with 3 elements.
                Each element correspond to vertices that create a triangle.
        """
        self.mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        if folder is None:
            folder = tempfile.mkdtemp()
        if file_name is None:
            file_name = uuid.uuid4()
        self.temp_dir_path = Path(folder)
        self.file_name = file_name
        self.temp_path = self.temp_dir_path / "{}.obj".format(self.file_name)
        self.mesh.export(self.temp_path.as_posix())

    def get_path(self) -> str:
        """Get path of the object.

        Commands require that an object is represented without any extension.

        Returns:
            Path to the temporary object created in the file system that
            holds the Wavefront OBJ data of the object.
        """
        return self.temp_path.with_suffix("").as_posix()

    def align_mesh(self):
        """Create data of a 3D mesh to calculate Light Field Distance.

        It runs an external command that create intermediate files and moves
        these files to created temporary folder.

        Returns:
            None
        """
        run_dir = self.temp_dir_path
        # copy_file = []
        copy_file = ['3DAlignment', 'align10.txt', 'q8_table', '12_0.obj',
                     '12_1.obj',
                     '12_2.obj',
                     '12_3.obj',
                     '12_4.obj',
                     '12_5.obj',
                     '12_6.obj',
                     '12_7.obj',
                     '12_8.obj',
                     '12_9.obj', ]
        for f in copy_file:
            os.system(
                'cp %s %s' % (os.path.join(CURRENT_DIR, f),
                              os.path.join(run_dir, f)))
        process = subprocess.Popen(
            ['./3DAlignment', self.temp_path.with_suffix("").as_posix()],
            cwd=run_dir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        output, err = process.communicate()

        if len(err) > 0:
            print(err)
            sys.exit(1)

        for file, out_file in zip(
                GENERATED_FILES_NAMES, OUTPUT_NAME_TEMPLATES
        ):
            shutil.move(
                os.path.join(run_dir, file),
                (
                        self.temp_dir_path / out_file.format(self.file_name)
                ).as_posix(),
            )
        for f in copy_file:
            os.system('rm -rf %s' % (os.path.join(run_dir, f)))

        os.system('rm -rf %s' % (self.temp_path.as_posix()))
