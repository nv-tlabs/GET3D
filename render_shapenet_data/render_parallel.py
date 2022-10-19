# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import os
import argparse
import subprocess
import time

parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument(
    '--save_folder', type=str, default='./tmp',
    help='path for saving rendered image')
parser.add_argument(
    '--dataset_folder', type=str, default='./tmp',
    help='path for downloaded 3d dataset folder')
parser.add_argument(
    '--blender_root', type=str, default='./tmp',
    help='path for blender')
args = parser.parse_args()

save_folder = args.save_folder
dataset_folder = args.dataset_folder
blender_root = args.blender_root

synset_list = [
    # '02958343',  # Car
    '03001627',  # Chair
    '03790512'  # Motorbike
]
scale_list = [
    # 0.9,
    0.7,
    0.9
]

for synset, obj_scale in zip(synset_list, scale_list):
    file_list = sorted(os.listdir(os.path.join(dataset_folder, synset)))
    idx = 0
    start_time = time.time()
    while idx < len(file_list):
        print("Done with %d/%d" % (idx, len(file_list)))

        stdout = open('stdout.txt', 'w')
        stderr = open('stderr.txt', 'w')
        file = file_list[idx]
        render_cmd = '%s -b -P render_shapenet.py -- --output %s %s  --scale %f --views 24 --resolution 1024 --gpu 0' % (
            blender_root, save_folder, os.path.join(dataset_folder, synset, file, 'model.obj'), obj_scale
        )
        p0 = subprocess.Popen(render_cmd, shell=True, stdout=stdout, stderr=stderr)
        idx += 1

        file = file_list[idx]
        render_cmd = '%s -b -P render_shapenet.py -- --output %s %s  --scale %f --views 24 --resolution 1024 --gpu 1' % (
            blender_root, save_folder, os.path.join(dataset_folder, synset, file, 'model.obj'), obj_scale
        )
        p1 = subprocess.Popen(render_cmd, shell=True, stdout=stdout, stderr=stderr)
        idx += 1

        file = file_list[idx]
        render_cmd = '%s -b -P render_shapenet.py -- --output %s %s  --scale %f --views 24 --resolution 1024 --gpu 2' % (
            blender_root, save_folder, os.path.join(dataset_folder, synset, file, 'model.obj'), obj_scale
        )
        p2 = subprocess.Popen(render_cmd, shell=True, stdout=stdout, stderr=stderr)
        idx += 1

        file = file_list[idx]
        render_cmd = '%s -b -P render_shapenet.py -- --output %s %s  --scale %f --views 24 --resolution 1024 --gpu 3' % (
            blender_root, save_folder, os.path.join(dataset_folder, synset, file, 'model.obj'), obj_scale
        )
        p3 = subprocess.Popen(render_cmd, shell=True, stdout=stdout, stderr=stderr)
        idx += 1

        file = file_list[idx]
        render_cmd = '%s -b -P render_shapenet.py -- --output %s %s  --scale %f --views 24 --resolution 1024 --gpu 4' % (
            blender_root, save_folder, os.path.join(dataset_folder, synset, file, 'model.obj'), obj_scale
        )
        p4 = subprocess.Popen(render_cmd, shell=True, stdout=stdout, stderr=stderr)
        idx += 1

        file = file_list[idx]
        render_cmd = '%s -b -P render_shapenet.py -- --output %s %s  --scale %f --views 24 --resolution 1024 --gpu 5' % (
            blender_root, save_folder, os.path.join(dataset_folder, synset, file, 'model.obj'), obj_scale
        )
        p5 = subprocess.Popen(render_cmd, shell=True, stdout=stdout, stderr=stderr)
        idx += 1

        file = file_list[idx]
        render_cmd = '%s -b -P render_shapenet.py -- --output %s %s  --scale %f --views 24 --resolution 1024 --gpu 6' % (
            blender_root, save_folder, os.path.join(dataset_folder, synset, file, 'model.obj'), obj_scale
        )
        p6 = subprocess.Popen(render_cmd, shell=True, stdout=stdout, stderr=stderr)
        idx += 1

        file = file_list[idx]
        render_cmd = '%s -b -P render_shapenet.py -- --output %s %s  --scale %f --views 24 --resolution 1024 --gpu 7' % (
            blender_root, save_folder, os.path.join(dataset_folder, synset, file, 'model.obj'), obj_scale
        )
        p7 = subprocess.Popen(render_cmd, shell=True, stdout=stdout, stderr=stderr)
        idx += 1

        p0.wait()
        p1.wait()
        p2.wait()
        p3.wait()
        p4.wait()
        p5.wait()
        p6.wait()
        p7.wait()

    end_time = time.time()
    print('Time for rendering %d models: %f' % (len(file_list), end_time - start_time))
