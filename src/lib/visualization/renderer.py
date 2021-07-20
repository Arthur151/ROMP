# -*- coding: utf-8 -*-

# brought from https://github.com/mkocabas/VIBE/blob/master/lib/utils/renderer.py
import sys, os
import json
import torch
import math
import trimesh
import pickle
import platform
'''
if 'Ubuntu' in platform.version():
    print('In Ubuntu, using osmesa mode for rendering')
    os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
else:
    print('In other system, using egl mode for rendering')
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
'''
plt = platform.system()
if plt != "Windows":
    if 'Ubuntu' in platform.version():
        print('In Ubuntu, using osmesa mode for rendering')
        os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
    else:
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
        
        
import numpy as np
import pyrender
from pyrender.constants import RenderFlags
root_dir = os.path.join(os.path.dirname(__file__),'..')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
import config
import constants
from config import args

colors = {
    'pink': [.7, .7, .9],
    'neutral': [.9, .9, .8], #[.7, .7, .6],#
    'capsule': [.7, .75, .5],
    'yellow': [.5, .7, .75],
}

class WeakPerspectiveCamera(pyrender.Camera):
    def __init__(self,
                 scale,
                 translation,
                 znear=pyrender.camera.DEFAULT_Z_NEAR,
                 zfar=None,
                 name=None):
        super(WeakPerspectiveCamera, self).__init__(
            znear=znear,
            zfar=zfar,
            name=name,
        )
        self.scale = scale
        self.translation = translation

    def get_projection_matrix(self, width=None, height=None):
        P = np.eye(4)
        P[0, 0] = self.scale[0]
        P[1, 1] = self.scale[1]
        P[0, 3] = self.translation[0] * self.scale[0]
        P[1, 3] = -self.translation[1] * self.scale[1]
        P[2, 2] = -1
        return P


class Renderer:
    def __init__(self, faces, resolution=(224,224), orig_img=False, wireframe=False):
        self.resolution = resolution

        self.faces = faces
        self.orig_img = orig_img
        self.wireframe = wireframe
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=self.resolution[0],
            viewport_height=self.resolution[1],
            point_size=1.0)

        # set the scene
        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))

        #light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=5)
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2)

        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, -1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [0, 1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [1, 1, 2]
        self.scene.add(light, pose=light_pose)

    def __call__(self, verts, cam=[1,1,0,0], angle=None, axis=None, mesh_filename=None, color=[1.0, 1.0, 0.9]):

        mesh = trimesh.Trimesh(vertices=verts, faces=self.faces, process=False)

        Rx = trimesh.transformations.rotation_matrix(math.radians(180), [1, 0, 0])
        mesh.apply_transform(Rx)

        if mesh_filename is not None:
            mesh.export(mesh_filename)

        if angle and axis:
            R = trimesh.transformations.rotation_matrix(math.radians(angle), axis)
            mesh.apply_transform(R)

        sx, sy, tx, ty = cam

        camera = WeakPerspectiveCamera(
            scale=[sx, sy],
            translation=[tx, ty],
            zfar=1000.
        )

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor=(color[0], color[1], color[2], 1.0)
        )

        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        mesh_node = self.scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        cam_node = self.scene.add(camera, pose=camera_pose)

        if self.wireframe:
            render_flags = RenderFlags.RGBA | RenderFlags.ALL_WIREFRAME
        else:
            render_flags = RenderFlags.RGBA

        image, _ = self.renderer.render(self.scene, flags=render_flags)
        #valid_mask = (rgb[:, :, -1] > 0)[:, :, np.newaxis]
        #output_img = rgb[:, :, :-1] * valid_mask + (1 - valid_mask) * img
        #image = output_img.astype(np.uint8)

        self.scene.remove_node(mesh_node)
        self.scene.remove_node(cam_node)

        return image

def get_renderer(test=False,resolution = (512,512,3),part_segment=False):
    faces = pickle.load(open(os.path.join(args().smpl_model_path,'smpl','SMPL_NEUTRAL.pkl'),'rb'), encoding='latin1')['f']
    renderer = Renderer(faces,resolution=resolution[:2])
    
    return renderer

if __name__ == '__main__':
    get_renderer(test=True,model_type='smpl')
