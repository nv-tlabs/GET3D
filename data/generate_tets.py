# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import os
import numpy as np

'''
This code segment shows how to use Quartet: https://github.com/crawforddoran/quartet, 
to generate a tet grid 
1) Download, compile and run Quartet as described in the link above. Example usage `quartet meshes/cube.obj 0.5 cube_5.tet`
2) Run the function below to generate a file `cube_32_tet.tet`
'''


def generate_tetrahedron_grid_file(res=32, root='..'):
    frac = 1.0 / res
    command = 'cd %s/quartet; ' % (root) + \
              './quartet meshes/cube.obj %f meshes/cube_%f_tet.tet -s meshes/cube_boundary_%f.obj' % (frac, res, res)
    os.system(command)


'''
This code segment shows how to convert from a quartet .tet file to compressed npz file
'''


def generate_tetrahedrons(res=50, root='..'):
    tetrahedrons = []
    vertices = []
    if res > 1.0:
        res = 1.0 / res

    root_path = os.path.join(root, 'quartet/meshes')
    file_name = os.path.join(root_path, 'cube_%f_tet.tet' % (res))

    # generate tetrahedron is not exist files
    if not os.path.exists(file_name):
        command = 'cd %s/quartet; ' % (root) + \
                  './quartet meshes/cube.obj %f meshes/cube_%f_tet.tet -s meshes/cube_boundary_%f.obj' % (res, res, res)
        os.system(command)

    with open(file_name, 'r') as f:
        line = f.readline()
        line = line.strip().split(' ')
        n_vert = int(line[1])
        n_t = int(line[2])
        for i in range(n_vert):
            line = f.readline()
            line = line.strip().split(' ')
            assert len(line) == 3
            vertices.append([float(v) for v in line])
        for i in range(n_t):
            line = f.readline()
            line = line.strip().split(' ')
            assert len(line) == 4
            tetrahedrons.append([int(v) for v in line])

    assert len(tetrahedrons) == n_t
    assert len(vertices) == n_vert
    vertices = np.asarray(vertices)
    vertices[vertices <= (0 + res / 4.0)] = 0  # determine the boundary point
    vertices[vertices >= (1 - res / 4.0)] = 1  # determine the boundary point
    np.savez_compressed('%d_compress' % (res), vertices=vertices, tets=tetrahedrons)
    return vertices, tetrahedrons
