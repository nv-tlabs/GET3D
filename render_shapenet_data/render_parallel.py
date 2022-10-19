# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import os
import argparse
import json
import time
import subprocess

parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument(
    '--save_folder', type=str, default='./shapenet_rendered',
    help='path for saving rendered image')
parser.add_argument(
    '--dataset_list', type=str, default='./dataset_list.json',
    help='path to a json linking datasets')
parser.add_argument(
    '--blender_root', type=str, default='blender',
    help='path for blender')
parser.add_argument(
    '--shapenet_version', type=str, default='1',
    help='ShapeNet version 1 or 2')
parser.add_argument(
    '--num_views', type=str, default='24',
    help='Number of views to capture per object')
parser.add_argument(
    '--engine', type=str, default='CYCLES',
    help='Use CYCLES or EEVEE - CYCLES is a realistic path tracer (slow), EEVEE is a real-time renderer (fast)')
parser.add_argument(
    '--quiet_mode', type=bool, default=1,
    help='Route output of console to log file')
parser.add_argument(
    '--headless', action='store_true', default=False, help='Run blender in headless mode')
args = parser.parse_args()

engine = args.engine
quiet_mode = args.quiet_mode
save_folder = args.save_folder
dataset_list = args.dataset_list
blender_root = args.blender_root
shapenet_version = args.shapenet_version
num_views = args.num_views

if args.headless and args.engine == 'EEVEE':
    from pyvirtualdisplay import Display
    Display().start()

# check if dataset_list exists, throw error if not
if not os.path.exists(dataset_list):
    raise ValueError('dataset_list does not exist!')

# check if save_folder exists
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

scale_list = []
path_list = []

# read and parse json file at dataset_list.json
with open(dataset_list, 'r') as f:
    dataset = json.load(f)

# example json entry:
#   {
#     "name": "Car",
#     "id": "02958343",
#     "scale": 0.9,
#     "directory": "./shapenet/02958343"
#   }
for entry in dataset:
    scale_list.append(entry['scale'])
    path_list.append(entry['directory'])


# for shapenet v2, we normalize the model location
if shapenet_version == '2':
    for obj_scale, dataset_folder in zip(scale_list, path_list):
        file_list = sorted(os.listdir(os.path.join(dataset_folder)))
        for file in file_list:
            # check if file_list+'/models' exists
            if os.path.exists(os.path.join(dataset_folder, file, 'models')):
                # move all files in file_list+'/models' to file_list
                os.system('mv ' + os.path.join(dataset_folder, file, 'models/*') + ' ' + os.path.join(dataset_folder, file))
                # remove file_list+'/models' if it exists
                os.system('rm -rf ' + os.path.join(dataset_folder, file, 'models'))
            material_file = os.path.join(dataset_folder, file, 'model_normalized.mtl')
            # read material_file as a text file, replace any instance of '../images' with './images'
            with open(material_file, 'r') as f:
                material_file_text = f.read()
            material_file_text = material_file_text.replace('../images', './images')
            # write the modified text to material_file
            with open(material_file, 'w') as f:
                f.write(material_file_text)

# ShapeNetCore v2 normalizes the scale and orientation of the models and the names are changed as a result
model_name = 'model.obj'
if shapenet_version == '2':
    model_name = 'model_normalized.obj'

suffix = ''
if(args.quiet_mode == '1'):
    suffix = ' >> tmp.out'

for obj_scale, dataset_folder in zip(scale_list, path_list):
    file_list = sorted(os.listdir(os.path.join(dataset_folder)))
    idx = 0
    start_time = time.time()
    while idx < len(file_list):
        print("Done with %d/%d" % (idx, len(file_list)))

        stdout = open('stdout.txt', 'w')
        stderr = open('stderr.txt', 'w')
        file = file_list[idx]
        render_cmd = '%s -b -P render_shapenet.py -- --output %s %s  --scale %f --views %s --engine %s%s --gpu 0' % (
            blender_root, save_folder, os.path.join(dataset_folder, file, model_name), obj_scale, num_views, engine, suffix
        )
        p0 = subprocess.Popen(render_cmd, shell=True, stdout=stdout, stderr=stderr)
        idx += 1

        file = file_list[idx]
        render_cmd = '%s -b -P render_shapenet.py -- --output %s %s  --scale %f --views %s --engine %s%s --gpu 1' % (
            blender_root, save_folder, os.path.join(dataset_folder, file, model_name), obj_scale, num_views, engine, suffix
        )
        p1 = subprocess.Popen(render_cmd, shell=True, stdout=stdout, stderr=stderr)
        idx += 1

        file = file_list[idx]
        render_cmd = '%s -b -P render_shapenet.py -- --output %s %s  --scale %f --views %s --engine %s%s --gpu 2' % (
            blender_root, save_folder, os.path.join(dataset_folder, file, model_name), obj_scale, num_views, engine, suffix
        )
        p2 = subprocess.Popen(render_cmd, shell=True, stdout=stdout, stderr=stderr)
        idx += 1

        file = file_list[idx]
        render_cmd = '%s -b -P render_shapenet.py -- --output %s %s  --scale %f --views %s --engine %s%s --gpu 3' % (
            blender_root, save_folder, os.path.join(dataset_folder, file, model_name), obj_scale, num_views, engine, suffix
        )
        p3 = subprocess.Popen(render_cmd, shell=True, stdout=stdout, stderr=stderr)
        idx += 1

        file = file_list[idx]
        render_cmd = '%s -b -P render_shapenet.py -- --output %s %s  --scale %f --views %s --engine %s%s --gpu 4' % (
            blender_root, save_folder, os.path.join(dataset_folder, file, model_name), obj_scale, num_views, engine, suffix
        )
        p4 = subprocess.Popen(render_cmd, shell=True, stdout=stdout, stderr=stderr)
        idx += 1

        file = file_list[idx]
        render_cmd = '%s -b -P render_shapenet.py -- --output %s %s  --scale %f --views %s --engine %s%s --gpu 5' % (
            blender_root, save_folder, os.path.join(dataset_folder, file, model_name), obj_scale, num_views, engine, suffix
        )
        p5 = subprocess.Popen(render_cmd, shell=True, stdout=stdout, stderr=stderr)
        idx += 1

        file = file_list[idx]
        render_cmd = '%s -b -P render_shapenet.py -- --output %s %s  --scale %f --views %s --engine %s%s --gpu 6' % (
            blender_root, save_folder, os.path.join(dataset_folder, file, model_name), obj_scale, num_views, engine, suffix
        )
        p6 = subprocess.Popen(render_cmd, shell=True, stdout=stdout, stderr=stderr)
        idx += 1

        file = file_list[idx]
        render_cmd = '%s -b -P render_shapenet.py -- --output %s %s  --scale %f --views %s --engine %s%s --gpu 7' % (
            blender_root, save_folder, os.path.join(dataset_folder, file, model_name), obj_scale, num_views, engine, suffix
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