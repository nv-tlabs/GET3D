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
import subprocess
from multiprocessing.pool import ThreadPool
import subprocess

# Connect EFS
# /home/user/mirage-dev/GET3D/render_shapenet_data/mirageml-dev/aman/experiements/GET3D/render_shapenet_data
# sudo sshfs ubuntu@ec2-3-95-21-26.compute-1.amazonaws.com:/home/ubuntu/mirage-dev/  /home/user/mirage-dev/GET3D/render_shapenet_data/mirageml-dev -o IdentityFile=/home/user/mirage-dev/GET3D/render_shapenet_data/mirage-omniverse.pem -o allow_other

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
args = parser.parse_args()


engine = args.engine
quiet_mode = args.quiet_mode
save_folder = args.save_folder
dataset_list = args.dataset_list
blender_root = args.blender_root
shapenet_version = args.shapenet_version
num_views = args.num_views

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

import pdb;
for obj_scale, dataset_folder in zip(scale_list, path_list):
    file_list = sorted(os.listdir(os.path.join(dataset_folder)))
    num = None  # set to the number of workers you want (it defaults to the cpu count of your machine)
    tp = ThreadPool(num)
    def work(file):
        output_dir = "/home/user/mirage-dev/GET3D/render_shapenet_data/mirageml-dev/aman/experiements/GET3D/shapenet_rendered"
        camera_dir = os.path.abspath(os.path.join(save_folder, "camera", dataset_folder.split("/")[-1], file))
        camera_save_dir = os.path.join(output_dir, "camera", dataset_folder.split("/")[-1], file)
        img_dir = os.path.abspath(os.path.join(save_folder, "img", dataset_folder.split("/")[-1], file))
        img_save_dir = os.path.join(output_dir, "img", dataset_folder.split("/")[-1], file)

        if os.path.exists(camera_save_dir) and os.path.exists(img_save_dir):
            print("Files Exist on EFS; ", file)
            if os.path.exists(camera_dir) and os.path.exists(img_dir):
                print("Removing Local: ",file)
                subprocess.call(["rm", "-rf", camera_dir])
                subprocess.call(["rm", "-rf", img_dir])
            return
        elif os.path.exists(camera_dir) and os.path.exists(img_dir):
            print("Files Exist Locally Moving to EFS: ", file)
            subprocess.call(["mv", camera_dir, camera_save_dir])
            subprocess.call(["mv", img_dir, img_save_dir])
            return

        print("Rendering: ", file)
        render_cmd = '%s -b -P render_shapenet.py -- --output %s %s  --scale %f --views %s --engine %s%s' % (
            blender_root, save_folder, os.path.join(dataset_folder, file, model_name), obj_scale, num_views, engine, suffix
        )
        os.system(render_cmd)

        print("Moving:", camera_dir, camera_save_dir)
        subprocess.call(["mv", camera_dir, camera_save_dir])
        print("Moving", img_dir, img_save_dir)
        subprocess.call(["mv", img_dir, img_save_dir])

    for idx, file in enumerate(file_list):
        tp.apply_async(work, (file,))

    tp.close()
    tp.join()



