# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from __future__ import print_function
import math, sys, random, argparse, json, os, tempfile
from datetime import datetime as dt
from collections import Counter

from math import pi

"""
Renders random scenes using Blender, each with with a random number of objects;
each object has a random size, position, color, and shape. Objects will be
nonintersecting but may partially occlude each other. Output images will be
written to disk as PNGs, and we will also write a JSON file for each image with
ground-truth scene information.

This file expects to be run from Blender like this:

blender --background --python render_images.py -- [arguments to this script]
"""

INSIDE_BLENDER = True
try:
    import bpy, bpy_extras, _cycles
    from mathutils import Vector
except ImportError as e:
    INSIDE_BLENDER = False
if INSIDE_BLENDER:
    try:
        import utils
    except ImportError as e:
        print("\nERROR")
        print("Running render_images.py from Blender and cannot import utils.py.")
        print("You may need to add a .pth file to the site-packages of Blender's")
        print("bundled python with a command like this:\n")
        print("echo $PWD >> $BLENDER/$VERSION/python/lib/python3.5/site-packages/clevr.pth")
        print("\nWhere $BLENDER is the directory where Blender is installed, and")
        print("$VERSION is your Blender version (such as 2.78).")
        sys.exit(1)

parser = argparse.ArgumentParser()

# Input options
parser.add_argument('--base_scene_blendfile', default='data/base_scene.blend',
                    help="Base blender file on which all scenes are based; includes " +
                         "ground plane, lights, and camera.")
parser.add_argument('--properties_json', default='data/properties_3drec.json',
                    help="JSON file defining objects, materials, sizes, and colors. " +
                         "The \"colors\" field maps from CLEVR color names to RGB values; " +
                         "The \"sizes\" field maps from CLEVR size names to scalars used to " +
                         "rescale object models; the \"materials\" and \"shapes\" fields map " +
                         "from CLEVR material and shape names to .blend files in the " +
                         "--object_material_dir and --shape_dir directories respectively.")
parser.add_argument('--shape_dir', default='data/shapes',
                    help="Directory where .blend files for object models are stored")
parser.add_argument('--material_dir', default='data/materials',
                    help="Directory where .blend files for materials are stored")
parser.add_argument('--shape_color_combos_json', default=None,
                    help="Optional path to a JSON file mapping shape names to a list of " +
                         "allowed color names for that shape. This allows rendering images " +
                         "for CLEVR-CoGenT.")
parser.add_argument('--mode',
                    help="Select one mode from ['rnd', 'basic, stack']. ")

# Settings for objects
parser.add_argument('--min_objects', default=3, type=int,
                    help="The minimum number of objects to place in each scene")
parser.add_argument('--max_objects', default=10, type=int,
                    help="The maximum number of objects to place in each scene")
parser.add_argument('--min_dist', default=0.25, type=float,
                    help="The minimum allowed distance between object centers")
parser.add_argument('--margin', default=0.4, type=float,
                    help="Along all cardinal directions (left, right, front, back), all " +
                         "objects will be at least this distance apart. This makes resolving " +
                         "spatial relationships slightly less ambiguous.")
parser.add_argument('--min_pixels_per_object', default=200, type=int,
                    help="All objects will have at least this many visible pixels in the " +
                         "final rendered images; this ensures that no objects are fully " +
                         "occluded by other objects.")
parser.add_argument('--max_retries', default=500, type=int,
                    help="The number of times to try placing an object before giving up and " +
                         "re-placing all objects in the scene.")
parser.add_argument('--n_views', default=1, type=int,
                    help="The number of images with the same but moved set of objects")

# Output settings
parser.add_argument('--start_idx', default=0, type=int,
                    help="The index at which to start for numbering rendered images. Setting " +
                         "this to non-zero values allows you to distribute rendering across " +
                         "multiple machines and recombine the results later.")
parser.add_argument('--num_images', default=5, type=int,
                    help="The number of images to render")
parser.add_argument('--filename_prefix', default='CLEVR',
                    help="This prefix will be prepended to the rendered images and JSON scenes")
parser.add_argument('--split', default='new',
                    help="Name of the split for which we are rendering. This will be added to " +
                         "the names of rendered images, and will also be stored in the JSON " +
                         "scene structure for each image.")
parser.add_argument('--output_dir', default='../output/train',
                    help="The directory where output [images, instances, scene, (blend)] will be" +
                         "stored. It will be created if it does not exist.")
parser.add_argument('--save_blendfiles', type=int, default=0,
                    help="Setting --save_blendfiles 1 will cause the blender scene file for " +
                         "each generated image to be stored in the directory specified by " +
                         "the --output_blend_dir flag. These files are not saved by default " +
                         "because they take up ~5-10MB each.")
parser.add_argument('--version', default='1.0',
                    help="String to store in the \"version\" field of the generated JSON file")
parser.add_argument('--license',
                    default="Creative Commons Attribution (CC-BY 4.0)",
                    help="String to store in the \"license\" field of the generated JSON file")
parser.add_argument('--date', default=dt.today().strftime("%m/%d/%Y"),
                    help="String to store in the \"date\" field of the generated JSON file; " +
                         "defaults to today's date")

# Rendering options
parser.add_argument('--use_gpu', default=0, type=int,
                    help="Setting --use_gpu 1 enables GPU-accelerated rendering using CUDA. " +
                         "You must have an NVIDIA GPU with the CUDA toolkit installed for " +
                         "to work.")
parser.add_argument('--width', default=64, type=int,
                    help="The width (in pixels) for the rendered images [320]")
parser.add_argument('--height', default=64, type=int,
                    help="The height (in pixels) for the rendered images [240]")
parser.add_argument('--color_bg', default=0, type=int,
                    help="Setting --color_bg 1 results in randomly colored background ")
parser.add_argument('--key_light_jitter', default=.0, type=float,
                    help="The magnitude of random jitter to add to the key light position.")
parser.add_argument('--fill_light_jitter', default=.0, type=float,
                    help="The magnitude of random jitter to add to the fill light position.")
parser.add_argument('--back_light_jitter', default=.0, type=float,
                    help="The magnitude of random jitter to add to the back light position.")
parser.add_argument('--camera_jitter', default=.0, type=float,
                    help="The magnitude of random jitter to add to the camera position")
parser.add_argument('--render_num_samples', default=512, type=int,
                    help="The number of samples to use when rendering. Larger values will " +
                         "result in nicer images but will cause rendering to take longer.")
parser.add_argument('--render_min_bounces', default=8, type=int,
                    help="The minimum number of bounces to use for rendering.")
parser.add_argument('--render_max_bounces', default=8, type=int,
                    help="The maximum number of bounces to use for rendering.")
parser.add_argument('--render_tile_size', default=256, type=int,
                    help="The tile size to use for rendering. This should not affect the " +
                         "quality of the rendered image but may affect the speed; CPU-based " +
                         "rendering may achieve better performance using smaller tile sizes " +
                         "while larger tile sizes may be optimal for GPU-based rendering.")
parser.add_argument('--data_randomization', default=0, type=int,
                    help="Apply noise during rendering process (color of bg+objects, light sources)")


def main(args):
    time_start = dt.now()

    num_digits = 6
    prefix = '%s_%s_' % (args.filename_prefix, args.split)
    img_template = '%s%%0%dd_0.png' % (prefix, num_digits)
    inst_template = '%s%%0%dd_0.png' % (prefix, num_digits)
    depth_template = '%s%%0%dd_0_0001.exr' % (prefix, num_digits)
    scene_template = '%s%%0%dd.json' % (prefix, num_digits)
    blend_template = '%s%%0%dd.blend' % (prefix, num_digits)
    obj_template = '%s%%0%dd_obj.off' % (prefix, num_digits)
    img_template = os.path.join(args.output_dir, 'images', img_template)
    inst_template = os.path.join(args.output_dir, 'instances', inst_template)
    depth_template = os.path.join(args.output_dir, 'depth', depth_template)
    scene_template = os.path.join(args.output_dir, 'scenes', scene_template)
    blend_template = os.path.join(args.output_dir, 'blendfiles', blend_template)
    obj_template = os.path.join(args.output_dir, 'objects/original', obj_template)

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.isdir(os.path.join(args.output_dir, 'images')):
        os.makedirs(os.path.join(args.output_dir, 'images'))
    if not os.path.isdir(os.path.join(args.output_dir, 'instances')):
        os.makedirs(os.path.join(args.output_dir, 'instances'))
    if not os.path.isdir(os.path.join(args.output_dir, 'scenes')):
        os.makedirs(os.path.join(args.output_dir, 'scenes'))
    if args.save_blendfiles == 1 and not os.path.isdir(os.path.join(args.output_dir, 'blendfiles')):
        os.makedirs(os.path.join(args.output_dir, 'blendfiles'))
    if not os.path.isdir(os.path.join(args.output_dir, 'objects')):
        os.makedirs(os.path.join(args.output_dir, 'objects'))
    if not os.path.isdir(os.path.join(args.output_dir, 'obj_files')):
        os.makedirs(os.path.join(args.output_dir, 'obj_files'))
    if not os.path.isdir(os.path.join(args.output_dir, 'objects/original')):
        os.makedirs(os.path.join(args.output_dir, 'objects/original'))

    if not args.mode in ['rnd', 'basic', 'stack']:
        print('[ERROR] Mode ', args.mode, ' not in [rnd, basic].')
        exit(0)

    all_scene_paths = []
    for i in range(args.num_images):
        # For cluster: break rendering after some time
        time_delta = dt.now() - time_start
        time_delta_hours = time_delta.seconds // 3600
        if time_delta_hours >= 23:
            print("Break rendering at scene {}, after running time {}.".format(i, str(time_delta)))
            exit(3)

        img_path = img_template % (i + args.start_idx)
        inst_path = inst_template % (i + args.start_idx)
        depth_path = depth_template % (i + args.start_idx)
        scene_path = scene_template % (i + args.start_idx)
        all_scene_paths.append(scene_path)
        blend_path = None
        if args.save_blendfiles == 1:
            blend_path = blend_template % (i + args.start_idx)
        if os.path.exists(scene_path) and os.path.exists(img_path) and os.path.exists(inst_path) and os.path.exists(depth_path):
            continue
        obj_path = obj_template % (i + args.start_idx)
        while True:
            num_objects = random.randint(args.min_objects, args.max_objects)
            valid = render_scene(args,
                                 num_objects=num_objects,
                                 output_index=(i + args.start_idx),
                                 output_split=args.split,
                                 output_image=img_path,
                                 output_instance=inst_path,
                                 output_scene=scene_path,
                                 output_blendfile=blend_path,
                                 output_objfile=obj_path,
                                 )
            if valid:
                break

    # After rendering all images, combine the JSON files for each scene into a
    # single JSON file.
    all_scenes = []
    for scene_path in all_scene_paths:
        with open(scene_path, 'r') as f:
            all_scenes.append(json.load(f))
    output = {
        'info': {
            'date': args.date,
            'version': args.version,
            'split': args.split,
            'license': args.license,
        },
        'scenes': all_scenes
    }
    with open(os.path.join(args.output_dir, 'CLEVR_scenes'), 'w') as f:
        json.dump(output, f)


def render_scene(args,
                 num_objects=5,
                 output_index=0,
                 output_split='none',
                 output_image='render.png',
                 output_instance='render_inst.png',
                 output_scene='render_json',
                 output_blendfile=None,
                 output_objfile='obj.off'
                 ):

    # Load the main blendfile
    bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)

    # Load materials
    utils.load_materials(args.material_dir)

    # Set render arguments so we can get pixel coordinates later.
    # We use functionality specific to the CYCLES renderer so BLENDER_RENDER
    # cannot be used.
    render_args = bpy.context.scene.render
    render_args.engine = "CYCLES"
    render_args.filepath = output_image
    render_args.resolution_x = args.width
    render_args.resolution_y = args.height
    render_args.resolution_percentage = 100
    render_args.tile_x = args.render_tile_size
    render_args.tile_y = args.render_tile_size

    if args.use_gpu == 1:
        # Blender changed the API for enabling CUDA at some point
        if bpy.app.version < (2, 78, 0):
            bpy.context.user_preferences.system.compute_device_type = 'CUDA'
            bpy.context.user_preferences.system.compute_device = 'CUDA_0'
        else:
            cycles_prefs = bpy.context.user_preferences.addons['cycles'].preferences
            cycles_prefs.compute_device_type = 'CUDA'

    # Some CYCLES-specific stuff
    bpy.data.worlds['World'].cycles.sample_as_light = True
    bpy.context.scene.cycles.blur_glossy = 2.0
    bpy.context.scene.cycles.samples = args.render_num_samples
    bpy.context.scene.cycles.transparent_min_bounces = args.render_min_bounces
    bpy.context.scene.cycles.transparent_max_bounces = args.render_max_bounces
    if args.use_gpu == 1:
        bpy.context.scene.cycles.device = 'GPU'

    # This will give ground-truth information about the scene and its objects
    scene_struct = {
        'split': output_split,
        'image_index': output_index,
        'image_filename': os.path.basename(output_image),
        'instance_filename': os.path.basename(output_instance),
        'objects': [],
        'directions': {},
    }

    # Put a plane on the ground so we can compute cardinal directions
    bpy.ops.mesh.primitive_plane_add(radius=5)
    plane = bpy.context.object

    def rand(L):
        return 2.0 * L * (random.random() - 0.5)

    # Add random jitter to camera position
    if args.camera_jitter > 0:
        for i in range(3):
            bpy.data.objects['Camera'].location[i] += rand(args.camera_jitter)
    for i in range(3):
        bpy.data.objects['Camera'].location[i] = 0.5 * bpy.data.objects['Camera'].location[i]

    # Rt = get_3x4_RT_matrix_from_blender(bpy.data.objects['Camera'])
    # # TODO: new camera pose
    if args.mode == 'stack':
        for c in bpy.data.objects['Camera'].constraints:
            bpy.data.objects['Camera'].constraints.remove(c)
        bpy.data.objects['Camera'].rotation_euler[0] = math.pi * (75./180.)
        bpy.data.objects['Camera'].rotation_euler[1] = 0.
        bpy.data.objects['Camera'].rotation_euler[2] = math.pi * (90./180.)
        bpy.data.objects['Camera'].location[0] = 5.5
        bpy.data.objects['Camera'].location[1] = 0.
        bpy.data.objects['Camera'].location[2] = 3.
        bpy.data.objects['Camera'].data.lens = 30
        bpy.data.objects['Ground'].rotation_euler[2] = math.pi * (180./180.)
        bpy.data.objects['Ground'].location[0] = -17.
        bpy.data.objects['Ground'].location[1] = -25.

    # Get camera information [WARNING] camera output is updated only after rendering (i.e. do this at later point)
    # K = get_calibration_matrix_K_from_blender(bpy.data.objects['Camera'].data)
    # Rt = get_3x4_RT_matrix_from_blender(bpy.data.objects['Camera'])

    # Figure out the left, up, and behind directions along the plane and record
    # them in the scene structure
    camera = bpy.data.objects['Camera']
    plane_normal = plane.data.vertices[0].normal
    cam_behind = camera.matrix_world.to_quaternion() * Vector((0, 0, -1))
    cam_left = camera.matrix_world.to_quaternion() * Vector((-1, 0, 0))
    cam_up = camera.matrix_world.to_quaternion() * Vector((0, 1, 0))
    plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
    plane_left = (cam_left - cam_left.project(plane_normal)).normalized()
    plane_up = cam_up.project(plane_normal).normalized()

    # Delete the plane; we only used it for normals anyway. The base scene file
    # contains the actual ground plane.
    utils.delete_object(plane)

    # Save all six axis-aligned directions in the scene struct
    scene_struct['directions']['behind'] = tuple(plane_behind)
    scene_struct['directions']['front'] = tuple(-plane_behind)
    scene_struct['directions']['left'] = tuple(plane_left)
    scene_struct['directions']['right'] = tuple(-plane_left)
    scene_struct['directions']['above'] = tuple(plane_up)
    scene_struct['directions']['below'] = tuple(-plane_up)

    # Add random jitter to lamp positions
    if args.key_light_jitter > 0:
        for i in range(3):
            bpy.data.objects['Lamp_Key'].location[i] += rand(args.key_light_jitter)
    if args.back_light_jitter > 0:
        for i in range(3):
            bpy.data.objects['Lamp_Back'].location[i] += rand(args.back_light_jitter)
    if args.fill_light_jitter > 0:
        for i in range(3):
            bpy.data.objects['Lamp_Fill'].location[i] += rand(args.fill_light_jitter)

    # set lightning to background
    if not args.data_randomization == 1 or True:
        utils.set_layer(bpy.data.objects['Lamp_Key'], 2)
        utils.set_layer(bpy.data.objects['Lamp_Fill'], 2)
        utils.set_layer(bpy.data.objects['Lamp_Back'], 2)

    # # BG Color
    if args.data_randomization==1 or args.color_bg == 1:
        # rgb = [random.uniform(0, 1) for _ in range(3)]

        def clip(val):
            return max(min(val, 1.), 0.)
        eps = 0.05
        noise = random.gauss(0., eps)
        rgb = [clip(0.9 + noise) for _ in range(3)]

        mat = bpy.data.materials.new("PKHG")
        mat.diffuse_color = rgb
        o = bpy.data.objects['Ground']
        o.active_material = mat

    # Now make some (random) objects
    objects, blender_objects = add_random_objects(scene_struct, num_objects, args, output_index)

    # Render the scene and dump the scene data structure
    while True:
        # try:
        #     Save obj (mesh file)
        #     for i, obj in enumerate(blender_objects):
        #         bpy.context.scene.objects.active = bpy.data.objects[obj.name]
        #
        #         objfile = output_objfile.replace('_obj', '_' + str(i))
        #         # objfile = output_objfile.replace('_obj', '_' + obj.name)
        #         print('-- Save object to', objfile)
        #         bpy.ops.export_mesh.off(filepath=objfile)
        #         bpy.ops.export_mesh.ply(filepath=objfile.replace('.off', '.ply'))

            # Multiple images with moved objects
            for n in range(args.n_views):

                objects = move_objects(scene_struct, objects, blender_objects)

                # Movement was not valid to often.. - remove earlier files
                if objects is None:
                    print('-- Remove earlier files..')
                    prev_files = []
                    for i, obj in enumerate(blender_objects):
                        prev_files.append(output_objfile.replace('_obj', '_' + str(i)))
                        # prev_files.append(output_objfile.replace('_obj', '_' + obj.name))
                    for n_prev in range(n):
                        prev_output_image = output_image.replace('0.png', str(n_prev) + '.png')
                        prev_files.append(prev_output_image)
                        prev_files.append(prev_output_image.replace('images', 'depth').replace('.png', '_0001.png'))
                        prev_files.append(output_instance.replace('0.png', str(n_prev) + '.png'))
                    for f in prev_files:
                        if os.path.exists(f):
                            os.remove(f)
                    return False

                # print(bpy.data.objects['Camera'].location)
                # if n == 0:
                #     bpy.data.objects['Camera'].location[0] = bpy.data.objects['Camera'].location[0] * 1.5
                #     bpy.data.objects['Camera'].location[1] = bpy.data.objects['Camera'].location[1] * 1.4
                #     bpy.data.objects['Camera'].location[2] = bpy.data.objects['Camera'].location[2] * 1.6
                #
                #     for i, obj in enumerate(blender_objects):
                #         obj_name = obj.name
                #         bpy.context.scene.objects.active = bpy.data.objects[obj_name]
                #         print(obj_name)
                #         objects[i]['3d_coords'].append(tuple(bpy.context.object.location))
                #         objects[i]['rotation'].append(bpy.context.object.rotation_euler[2])
                #         objects[i]['pixel_coords'].append(utils.get_camera_coords(camera, bpy.context.object.location))

                # Render image (+ depth, instance seg.)
                new_output_image = output_image.replace('0.png', str(n)+'.png')

                new_output_depth = new_output_image.split('/')[-1].replace('.png', '_')
                render_depth_exr(os.path.join(args.output_dir, 'depth'), new_output_depth)
                # render_depth(os.path.join(args.output_dir, 'depth'), new_output_depth)

                new_output_inst = output_instance.replace('0.png', str(n)+'.png')
                _ = render_shadeless(blender_objects, path=new_output_inst, info_objects=objects)

                render_args.filepath = new_output_image
                bpy.ops.render.render(write_still=True)
                
                obj_filepath = new_output_image.replace('/images/', '/obj_files/')
                obj_filepath = obj_filepath.replace('png', 'obj')
                bpy.ops.export_scene.obj(filepath=obj_filepath, use_materials=True, use_triangles=True, use_normals=True, use_edges=True)

            break

        # except Exception as e:
        #     print(e)

    scene_struct['objects'] = objects
    scene_struct['relationships'] = compute_all_relationships(scene_struct, n_views=args.n_views)

    with open(output_scene, 'w') as f:
        json.dump(scene_struct, f, indent=2)

    if output_blendfile is not None:
        bpy.ops.wm.save_as_mainfile(filepath=output_blendfile)

    return True


def add_random_objects(scene_struct, num_objects, args, output_index):
    """
    Add random objects to the current blender scene
    """

    # Load the property file
    with open(args.properties_json, 'r') as f:
        properties = json.load(f)
        color_name_to_rgba = {}
        for name, rgb in properties['colors'].items():
            def clip(val):
                return max(min(val, 1.), 0.)

            if args.data_randomization:
                eps = 0.05
                rgba = [clip(float(c) / 255.0 + random.gauss(0., eps)) for c in rgb] + [1.0]
            else:
                rgba = [float(c) / 255.0 for c in rgb] + [1.0]
            color_name_to_rgba[name] = rgba
        material_mapping = [(v, k) for k, v in properties['materials'].items()]
        object_mapping = [(v, k) for k, v in properties['shapes'].items()]
        size_mapping = properties['sizes']

    # positions = []
    objects = []
    blender_objects = []
    for i in range(num_objects):

        # Choose random shape, size, and scale transformation factors along different axes
        if args.mode == 'rnd':
            obj_name, obj_name_out = random.choice(object_mapping)
            r = random.uniform(size_mapping['small'], size_mapping['large'])
            scale_vars = [random.uniform(0.5, 1.0) for _ in range(3)]
            scale_vars_max = max(scale_vars)
            scale_vars_norm = [sv/scale_vars_max for sv in scale_vars]
        elif args.mode == 'stack':
            obj_name, obj_name_out = random.choice(object_mapping)
            r = random.uniform(size_mapping['small'], size_mapping['large'])
            scale_vars = [1.0 for _ in range(3)]
            scale_vars_max = max(scale_vars)
            scale_vars_norm = [sv / scale_vars_max for sv in scale_vars]
        elif args.mode == 'basic':
            scale_vars_opt = [[1.0, 1.0, 1.0],
                              [1.0, 0.75, 1.0],
                              [0.5, 1.0, 1.0],
                              [0.75, 0.75, 1.0],
                              [0.5, 0.5, 1.0],
                              [1.0, 1.0, 0.75],
                              [1.0, 1.0, 0.5],
                              [0.75, 1.0, 0.5],
                              [1.0, 0.5, 0.75]]
            obj_name, obj_name_out = object_mapping[int(output_index / len(scale_vars_opt))]
            r = 1.
            scale_vars_norm = scale_vars_opt[output_index % len(scale_vars_opt)]

        # Choose random color and shape
        color_name, rgba = random.choice(list(color_name_to_rgba.items()))

        # color_name = 'rnd'
        # rgba = [random.uniform(0, 1) for _ in range(3)] + [1.]
        color_inst_list = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0.75, 0.75, 0.], [0., 0.75, 0.75], [0.75, 0., 0.75]]
        color_inst = color_inst_list[i]

        # For cube, adjust the size a bit
        if obj_name == 'Cube':
            r /= math.sqrt(2)

        # Actually add the object to the scene
        utils.add_object(args.shape_dir, obj_name, r, (0., 0.), theta=0., scale_vars=scale_vars_norm)
        obj = bpy.context.object
        blender_objects.append(obj)

        # Disable shadow
        obj.cycles_visibility.shadow = False
        if not args.data_randomization:
            obj.cycles_visibility.transmission = False
            obj.cycles_visibility.diffuse = False
            obj.cycles_visibility.glossy = False
            obj.cycles_visibility.volume_scatter = False

        # Attach a random material
        mat_name, mat_name_out = random.choice(material_mapping)
        utils.add_material(mat_name, Color=rgba)

        objects.append({
            'shape': obj_name_out,
            'size': r,
            'scale_vars': scale_vars_norm,
            'material': mat_name_out,
            '3d_coords': [],
            'rotation': [],
            'pixel_coords': [],
            'color': color_name,
            'color_rgb': rgba[:3],
            'color_rgb_inst': color_inst,
        })

    # OPTIONAL: Check that all objects are at least partially visible in the rendered image

    return objects, blender_objects     # [{object_info}], [bpy.context.object]


def move_objects(scene_struct, objects, blender_objects):
    """
    Sample new extrinsic coordinates for all objects in the scene.
    Ensure that there are no overlapping objects
    """

    camera = bpy.data.objects['Camera']

    prev_obj_positions = []

    for i, obj in enumerate(blender_objects):
        obj_name = obj.name
        bpy.context.scene.objects.active = bpy.data.objects[obj_name]
        r = objects[i]['size']

        # Choose random orientation for the object
        if args.mode == 'rnd' or args.mode =='stack':
            theta = 2.0 * math.pi * (random.random() - 0.5)
        elif args.mode == 'basic':
            theta = math.pi/4.
        obj.rotation_euler[2] = theta

        # Try to place the object, ensuring that we don't intersect any existing
        # objects and that we are more than the desired margin away from all existing
        # objects along all cardinal directions.
        num_tries = 0
        while True:
            # If we try and fail to place an object too many times, then delete all
            # the objects in the scene and start over. TODO
            num_tries += 1
            if num_tries > args.max_retries:
                print('To many attempts to move objects!')
                return None
            if args.mode == 'rnd':
                x, y = [random.uniform(-1.5, 1.5) for _ in range(2)]  # TODO: setting position
            elif args.mode == 'basic':
                x, y = [1, 0]
                # x, y = random.choice([[1, 1.5], [1, -1.5], [-1, 1.5], [-1, -1.5]])
            elif args.mode == 'stack':
                if i > 0:
                    prev_obj_3dcoords = objects[i - 1]['3d_coords']
                    prev_obj_size = objects[i - 1]['size']
                    prev_obj_scalevars = objects[i - 1]['scale_vars']
                    cur_obj_size = objects[i]['size']
                    cur_obj_scalevars = objects[i]['scale_vars']
                    if random.random() <= 0.25:
                        off = 0.25
                        x = prev_obj_3dcoords[0][0] + random.uniform(-off*prev_obj_scalevars[0], off*prev_obj_scalevars[0])
                        y = prev_obj_3dcoords[0][1] + random.uniform(-off*prev_obj_scalevars[1], off*prev_obj_scalevars[1])
                        z = prev_obj_3dcoords[0][2] + prev_obj_size*prev_obj_scalevars[2] + \
                            cur_obj_size * cur_obj_scalevars[2]
                        bpy.context.object.location[2] = z
                    else:
                        x = random.uniform(0., 1.5)  #(-1.5, 1.5)  # (-1., 1.)
                        y = random.uniform(-1.5, 1.5)
                        z = cur_obj_size * cur_obj_scalevars[2]
                else:
                    x = random.uniform(-1., 1.)
                    y = random.uniform(-1.5, 1.5)
            bpy.context.object.location[0] = x
            bpy.context.object.location[1] = y

            # Check to make sure the new object is further than min_dist from all
            # other objects, and further than margin along the four cardinal directions
            dists_good = True
            margins_good = True
            for (xx, yy, zz, rr) in prev_obj_positions:
                dx, dy = x - xx, y - yy
                dist = math.sqrt(dx * dx + dy * dy)
                flag_stack = False
                if dist - r - rr < args.min_dist:
                    if args.mode == 'stack':
                        dz = z - zz
                        eps = 1e-8
                        if dz < prev_obj_size * prev_obj_scalevars[2] + cur_obj_size * cur_obj_scalevars[2] - eps:
                            dists_good = False
                            break
                        else:
                            flag_stack = True
                    else:
                        dists_good = False
                        break
                for direction_name in ['left', 'right', 'front', 'behind']:
                    direction_vec = scene_struct['directions'][direction_name]
                    assert direction_vec[2] == 0
                    margin = dx * direction_vec[0] + dy * direction_vec[1]
                    if 0 < margin < args.margin and not flag_stack:  # TODO: direction for stacked objects
                        print('- Overlapping Objects:', num_tries, margin, args.margin, direction_name)
                        margins_good = False
                        break
                if not margins_good:
                    break

            if dists_good and margins_good:
                break

        # bpy.context.object.rotation_euler[2] = theta
        # bpy.context.object.location[0] = x
        # bpy.context.object.location[1] = y

        objects[i]['3d_coords'].append(tuple(bpy.context.object.location))
        objects[i]['rotation'].append(theta)
        objects[i]['pixel_coords'].append(utils.get_camera_coords(camera, bpy.context.object.location))

        # prev_obj_positions.append((x, y, r))
        prev_obj_positions.append(([bpy.context.object.location[i] for i in range(3)]+[r]))

    return objects


def compute_all_relationships(scene_struct, eps=0.2, n_views=2):
    """
    Computes relationships between all pairs of objects in the scene.

    Returns a dictionary mapping string relationship names to lists of lists of
    integers, where output[rel][i] gives a list of object indices that have the
    relationship rel with object i. For example if j is in output['left'][i] then
    object j is left of object i.

    Update:
    output[rel_name][img_n][obj_i] -> list
    - if j in output[rel_name][img_n][obj_i] i.e. object j is rel_name (e.g. left) of object i
    """
    all_relationships = {}
    for name, direction_vec in scene_struct['directions'].items():
        if name == 'above' or name == 'below':
            continue
        all_relationships[name] = [[] for _ in range(n_views)]
        for v in range(n_views):
            for i, obj1 in enumerate(scene_struct['objects']):
                coords1 = obj1['3d_coords']
                related = set()
                for j, obj2 in enumerate(scene_struct['objects']):
                    if obj1 == obj2: continue
                    coords2 = obj2['3d_coords']
                    diff = [coords2[v][k] - coords1[v][k] for k in [0, 1, 2]]
                    dot = sum(diff[k] * direction_vec[k] for k in [0, 1, 2])
                    if dot > eps:
                        related.add(j)
                all_relationships[name][v].append(sorted(list(related)))
    return all_relationships


def render_shadeless(blender_objects, path='flat.png', info_objects=None):
    """
    Render a version of the scene with shading disabled and unique materials
    assigned to all objects, and return a set of all colors that should be in the
    rendered image. The image itself is written to path. This is used to ensure
    that all objects will be visible in the final rendered scene.
    """
    render_args = bpy.context.scene.render

    # Cache the render args we are about to clobber
    old_filepath = render_args.filepath
    old_engine = render_args.engine
    old_use_antialiasing = render_args.use_antialiasing

    # Override some render settings to have flat shading
    render_args.filepath = path
    #render_args.engine = 'BLENDER_RENDER'
    render_args.use_antialiasing = False

    # # Move the lights and ground to layer 2 so they don't render
    # if args.data_randomization == 1:
    #     utils.set_layer(bpy.data.objects['Lamp_Key'], 2)
    #     utils.set_layer(bpy.data.objects['Lamp_Fill'], 2)
    #     utils.set_layer(bpy.data.objects['Lamp_Back'], 2)
    utils.set_layer(bpy.data.objects['Ground'], 2)

    # Add random shadeless materials to all objects
    object_colors = set()
    old_materials = []
    for i, obj in enumerate(blender_objects):
        old_materials.append(obj.data.materials[0])
        bpy.ops.material.new()
        mat = bpy.data.materials['Material']
        mat.name = 'Material_%d' % i
        if info_objects is None:
            while True:
                r, g, b = [random.random() for _ in range(3)]
                if (r, g, b) not in object_colors: break
            object_colors.add((r, g, b))
            mat.diffuse_color = [r, g, b]
        else:
            # mat.diffuse_color = info_objects[i]['color_rgb']
            mat.diffuse_color = info_objects[i]['color_rgb_inst']
        mat.use_shadeless = True
        obj.data.materials[0] = mat

    # Render the scene
    bpy.ops.render.render(write_still=True)

    # Undo the above; first restore the materials to objects
    for mat, obj in zip(old_materials, blender_objects):
        obj.data.materials[0] = mat

    # # Move the lights and ground back to layer 0    # out-commented for simple lightning condition
    # if args.data_randomization == 1:
    #     utils.set_layer(bpy.data.objects['Lamp_Key'], 0)
    #     utils.set_layer(bpy.data.objects['Lamp_Fill'], 0)
    #     utils.set_layer(bpy.data.objects['Lamp_Back'], 0)
    utils.set_layer(bpy.data.objects['Ground'], 0)

    # Set the render settings back to what they were
    render_args.filepath = old_filepath
    render_args.engine = old_engine
    render_args.use_antialiasing = old_use_antialiasing

    return object_colors


def render_depth_exr(base_path, name):
    # Set up rendering of depth map:
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    # clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)

    # create input render layer node
    rl = tree.nodes.new('CompositorNodeRLayers')

    fileOutput = tree.nodes.new(type="CompositorNodeOutputFile")
    fileOutput.base_path = base_path
    fileOutput.file_slots[0].path = name
    fileOutput.format.file_format = 'OPEN_EXR'

    links.new(rl.outputs[2], fileOutput.inputs[0])


def render_depth_png(base_path, name):
    # Set up rendering of depth map:
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    # clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)

    # create input render layer node
    rl = tree.nodes.new('CompositorNodeRLayers')

    map = tree.nodes.new(type="CompositorNodeMapRange")
    # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
    map.inputs[1].default_value = 0.0   # Min value here
    map.inputs[2].default_value = 16.0  # Max value here
    map.inputs[3].default_value = 0.0
    map.inputs[4].default_value = 1.0

    links.new(rl.outputs[2], map.inputs[0])

    invert = tree.nodes.new(type="CompositorNodeInvert")
    links.new(map.outputs[0], invert.inputs[1])

    fileOutput = tree.nodes.new(type="CompositorNodeOutputFile")
    fileOutput.base_path = base_path
    fileOutput.file_slots[0].path = name
    # links.new(invert.outputs[0], fileOutput.inputs[0])
    links.new(map.outputs[0], fileOutput.inputs[0])


def get_calibration_matrix_K_from_blender(camd):

    ## https://www.rojtberg.net/1601/from-blender-to-opencv-camera-and-back/
    # get the relevant data
    cam = camd
    scene = bpy.context.scene
    # assume image is not scaled
    assert scene.render.resolution_percentage == 100
    # assume angles describe the horizontal field of view
    assert cam.sensor_fit != 'VERTICAL'

    f_in_mm = cam.lens
    sensor_width_in_mm = cam.sensor_width

    w = scene.render.resolution_x
    h = scene.render.resolution_y

    pixel_aspect = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x

    f_x = f_in_mm / sensor_width_in_mm * w
    f_y = f_x * pixel_aspect

    # yes, shift_x is inverted. WTF blender?
    c_x = w * (0.5 - cam.shift_x)
    # and shift_y is still a percentage of width..
    c_y = h * 0.5 + w * cam.shift_y

    K = [[f_x, 0, c_x],
         [0, f_y, c_y],
         [0, 0, 1]]

    print('f_in_mm', f_in_mm)
    print('sensor_width_in_mm', sensor_width_in_mm)
    print('[u/v]_pixels (resolution_[]_in_px)', w, h)
    print('pixel_aspect_ratio', pixel_aspect)
    print('f_x/y', f_x, f_y)
    print('c_x, c_y', c_x, c_y)
    print('Camera Callibration', K)

    return K


def get_3x4_RT_matrix_from_blender(cam):

    # location, rotation = cam.matrix_world.decompose()[0:2]
    location = cam.location
    print('location', location)

    rotmode = cam.rotation_mode
    order = rotmode if rotmode not in {'QUATERNION', 'AXIS_ANGLE'} else 'XYZ'  # 'XYZ' would be default
    rot_euler = cam.matrix_world.to_euler(order)
    print('rotation (euler)', rot_euler)
    print('rotation (degrees)', math.degrees(rot_euler.x), math.degrees(rot_euler.y), math.degrees(rot_euler.z))
    R_world2bcam = rot_euler.to_matrix()
    print('R_world2bcam', R_world2bcam)


if __name__ == '__main__':
    if INSIDE_BLENDER:
        # Run normally
        argv = utils.extract_args()
        args = parser.parse_args(argv)
        main(args)
    elif '--help' in sys.argv or '-h' in sys.argv:
        parser.print_help()
    else:
        print('This script is intended to be called from blender like this:')
        print()
        print('blender --background --python render_images.py -- [args]')
        print()
        print('You can also run as a standalone python script to view all')
        print('arguments like this:')
        print()
        print('python render_images.py --help')

    exit(0)
