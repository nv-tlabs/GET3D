# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import argparse, sys, os, math, re
import bpy
from mathutils import Vector
import numpy as np

parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument(
    '--views', type=int, default=24,
    help='number of views to be rendered')
parser.add_argument(
    'obj', type=str,
    help='Path to the obj file to be rendered.')
parser.add_argument(
    '--output_folder', type=str, default='/tmp',
    help='The path the output will be dumped to.')
parser.add_argument(
    '--scale', type=float, default=1,
    help='Scaling factor applied to model. Depends on size of mesh.')
parser.add_argument(
    '--format', type=str, default='PNG',
    help='Format of files generated. Either PNG or OPEN_EXR')
parser.add_argument(
    '--engine', type=str, default='CYCLES',
    help='Blender internal engine for rendering. either CYCLES or EEVEE, ...')

argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)

if args.engine == 'CYCLES':
    bpy.ops.wm.open_mainfile(filepath=os.path.abspath("./blender/cycles_renderer.blend"))
else:
    bpy.ops.wm.open_mainfile(filepath=os.path.abspath("./blender/eevee_renderer.blend"))

# Set up rendering
context = bpy.context
scene = bpy.context.scene
render = bpy.context.scene.render

def bounds(obj, local=False):
    local_coords = obj.bound_box[:]
    om = obj.matrix_world

    if not local:
        worldify = lambda p: om @ Vector(p[:])
        coords = [worldify(p).to_tuple() for p in local_coords]
    else:
        coords = [p[:] for p in local_coords]

    rotated = zip(*coords[::-1])

    push_axis = []
    for (axis, _list) in zip('xyz', rotated):
        info = lambda: None
        info.max = max(_list)
        info.min = min(_list)
        info.distance = info.max - info.min
        push_axis.append(info)

    import collections

    originals = dict(zip(['x', 'y', 'z'], push_axis))

    o_details = collections.namedtuple('object_details', 'x y z')
    return o_details(**originals)


imported_object = bpy.ops.import_scene.obj(filepath=args.obj, use_edges=False, use_smooth_groups=False, split_mode='OFF')

for this_obj in bpy.data.objects:
    if this_obj.type == "MESH":
        this_obj.select_set(True)
        bpy.context.view_layer.objects.active = this_obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.split_normals()

bpy.ops.object.mode_set(mode='OBJECT')
print(len(bpy.context.selected_objects))
obj = bpy.context.selected_objects[0]
context.view_layer.objects.active = obj

mesh_obj = obj
scale = args.scale
factor = max(mesh_obj.dimensions[0], mesh_obj.dimensions[1], mesh_obj.dimensions[2]) / scale
print('size of object:')
print(mesh_obj.dimensions)
print(factor)
object_details = bounds(mesh_obj)
print(
    object_details.x.min, object_details.x.max,
    object_details.y.min, object_details.y.max,
    object_details.z.min, object_details.z.max,
)
print(bounds(mesh_obj))
mesh_obj.scale[0] /= factor
mesh_obj.scale[1] /= factor
mesh_obj.scale[2] /= factor
bpy.ops.object.transform_apply(scale=True)

# Get reference to camera and empty (rotation pivot)
cam = scene.objects['Camera']
cam_empty = scene.objects['Empty']

stepsize = 360.0 / args.views
rotation_mode = 'XYZ'

model_identifier = os.path.split(os.path.split(args.obj)[0])[1]
print('model identifier: ' + model_identifier)
synset_idx = args.obj.split('/')[-3]
print('synset idx: ' + synset_idx)

img_folder = os.path.join(os.path.abspath(args.output_folder), 'img', synset_idx, model_identifier)
camera_folder = os.path.join(os.path.abspath(args.output_folder), 'camera', synset_idx, model_identifier)

os.makedirs(img_folder, exist_ok=True)
os.makedirs(camera_folder, exist_ok=True)

rotation_angle_list = np.random.rand(args.views)
elevation_angle_list = np.random.rand(args.views)
rotation_angle_list = rotation_angle_list * 360
elevation_angle_list = elevation_angle_list * 30
np.save(os.path.join(camera_folder, 'rotation'), rotation_angle_list)
np.save(os.path.join(camera_folder, 'elevation'), elevation_angle_list)

for i in range(0, args.views):
    cam_empty.rotation_euler[2] = math.radians(rotation_angle_list[i])
    cam_empty.rotation_euler[0] = math.radians(elevation_angle_list[i])

    print("Rotation {}, {}".format((stepsize * i), math.radians(stepsize * i)))
    render_file_path = os.path.join(img_folder, '%03d.png' % (i))
    scene.render.filepath = render_file_path
    bpy.ops.render.render(write_still=True)
