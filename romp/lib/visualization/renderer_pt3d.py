# -*- coding: utf-8 -*-
# brought from https://github.com/mkocabas/VIBE/blob/master/lib/utils/renderer.py
import sys, os
import json
import torch
from torch import nn
import pickle
# Data structures and functions for rendering
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    FoVOrthographicCameras,
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex)

import numpy as np
import config
import constants
from config import args

mesh_color_table = {
    'pink': [.7, .7, .9],
    'neutral': [.9, .9, .8],
    'capsule': [.7, .75, .5],
    'yellow': [.5, .7, .75],
}

class MeshRendererWithDepth(nn.Module):
    def __init__(self, rasterizer, shader):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)
        return (images, fragments.zbuf)

class Renderer(nn.Module):
    def __init__(self, resolution=(512,512), perps=True, R=None, T=None, fov=args().FOV, use_gpu=args().gpu!='-1', with_depth=False):
        super(Renderer, self).__init__()
        self.perps = perps
        self.with_depth = with_depth
        if use_gpu:
            self.device = torch.device('cuda:{}'.format(str(args().gpu).split(',')[0]))
        else:
            self.device = torch.device('cpu')

        if R is None:
            R = torch.Tensor([[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]])
        if T is None:
            T = torch.Tensor([[0., 0., 0.]])

        if self.perps:
            # Initialize a camera.
            self.cameras = FoVPerspectiveCameras(R=R, T=T, fov=fov, device=self.device)
            self.lights = PointLights(ambient_color=((0.56, 0.56, 0.56),),location=torch.Tensor([[0., 0., 0.]]), device=self.device)
        else:
            if args().model_version==1:
                self.cameras = FoVOrthographicCameras(R=R, T=T, znear=0., zfar=100.0, max_y=1.0, min_y=-1.0, max_x=1.0, min_x=-1.0, device=self.device)
            else:
                self.cameras = FoVOrthographicCameras(R=R, T=T, znear=0., zfar=100.0, max_y=2.0, min_y=-2.0, max_x=2.0, min_x=-2.0, device=self.device)
            self.lights = DirectionalLights(ambient_color=((0.6, 0.6, 0.6),),direction=torch.Tensor([[0., -1., 0.]]), device=self.device)

        # Define the settings for rasterization and shading. Here we set the output image to be of size
        # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
        # and blur_radius=0.0. 
        raster_settings = RasterizationSettings(
            image_size=resolution[0], 
            blur_radius=0.0, 
            faces_per_pixel=1, 
            bin_size = 0, # to avoid the warning.
            )

        # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will 
        # interpolate the texture uv coordinates for each vertex, sample from a texture image and 
        # apply the Phong lighting model
        if not with_depth:
            self.renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=self.cameras, 
                    raster_settings=raster_settings
                ),
                shader=SoftPhongShader(
                    device=self.device,
                    cameras=self.cameras,
                    lights=self.lights))
        else:
            self.renderer = MeshRendererWithDepth(
                rasterizer=MeshRasterizer(
                    cameras=self.cameras, 
                    raster_settings=raster_settings
                ),
                shader=SoftPhongShader(
                    device=self.device,
                    cameras=self.cameras,
                    lights=self.lights))

    def __call__(self, verts, faces, colors=torch.Tensor(mesh_color_table['neutral']), merge_meshes=True, cam_params=None):
        assert len(verts.shape) == 3, print('The input verts of visualizer is bounded to be 3-dims (Nx6890 x3) tensor')
        verts, faces = verts.to(self.device), faces.to(self.device)
        verts_rgb = torch.ones_like(verts)
        verts_rgb = set_mesh_color(verts_rgb, colors)
        textures = TexturesVertex(verts_features=verts_rgb)
        verts[:,:,:2] *= -1
        meshes = Meshes(verts, faces, textures)
        if merge_meshes:
            meshes = join_meshes_as_scene(meshes)
        if cam_params is not None:
            if self.perps:
                R, T, fov = cam_params
                new_cam = FoVPerspectiveCameras(R=R, T=T, fov=fov, device=self.device)
            else:
                R, T, xyz_ranges = cam_params
                new_cam = FoVOrthographicCameras(R=R, T=T, **xyz_ranges, device=self.device)
            images = self.renderer(meshes,cameras=new_cam)
        else:
            images = self.renderer(meshes)

        return images

def set_mesh_color(verts_rgb, colors):
    if colors is None:
        colors = torch.Tensor(mesh_color_table['neutral'])
    if len(colors.shape) == 1:
        verts_rgb[:, :] = colors
    elif len(colors.shape) == 2:
        verts_rgb[:, :] = colors.unsqueeze(1)
    return verts_rgb

def get_renderer(test=False,**kwargs):
    renderer = Renderer(**kwargs)
    if test:
        import cv2
        dist = 1/np.tan(np.radians(args().FOV/2.))
        print('dist:', dist)
        model = pickle.load(open(os.path.join(args().smpl_model_path,'smpl','SMPL_NEUTRAL.pkl'),'rb'), encoding='latin1')
        np_v_template = torch.from_numpy(np.array(model['v_template'])).cuda().float()[None]
        face = torch.from_numpy(model['f'].astype(np.int32)).cuda()[None]
        np_v_template = np_v_template.repeat(2,1,1)
        np_v_template[1] += 0.3
        np_v_template[:,:,2] += dist
        face = face.repeat(2,1,1)
        result = renderer(np_v_template, face).cpu().numpy()
        for ri in range(len(result)):
            cv2.imwrite('test{}.png'.format(ri),(result[ri,:,:,:3]*255).astype(np.uint8))
    return renderer

if __name__ == '__main__':
    get_renderer(test=True, perps=True)