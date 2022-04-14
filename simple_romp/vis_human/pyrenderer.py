import os,sys
try:
    import pyrender
except:
    print('To use the pyrender, we are trying to install it via pip install pyrender.')
    print('If you meet any bug in this process, please refer to https://pyrender.readthedocs.io/en/latest/install/index.html to install it by youself.')
    os.system('pip install pyrender')
    import pyrender

import trimesh
import numpy as np


def add_light(scene, light):
    # Use 3 directional lights  
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)


class Py3DR(object):
    def __init__(self, FOV=60, height=512, width=512, focal_length=None):
        self.renderer = pyrender.OffscreenRenderer(height, width)
        if focal_length is None:
            self.focal_length = 1/(np.tan(np.radians(FOV/2)))
        else:
            self.focal_length = focal_length / max(height, width)*2
        self.rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        self.colors = [
                        (.7, .7, .6, 1.),
                        (.7, .5, .5, 1.),  # Pink
                        (.5, .5, .7, 1.),  # Blue
                        (.5, .55, .3, 1.),  # capsule
                        (.3, .5, .55, 1.),  # Yellow
                    ]

    def __call__(self, vertices, triangles, image, mesh_colors=None, f=None, persp=True, camera_pose=None):
        img_height, img_width = image.shape[:2]
        self.renderer.viewport_height = img_height
        self.renderer.viewport_width = img_width
        # Create a scene for each image and render all meshes
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                                ambient_light=(0.3, 0.3, 0.3))

        if camera_pose is None:
            camera_pose = np.eye(4)
        if persp:
            if f is None:
                f = self.focal_length * max(img_height, img_width) / 2
            camera = pyrender.camera.IntrinsicsCamera(fx=f, fy=f, cx=img_width / 2., cy=img_height / 2.)
        else:
            xmag = ymag = np.abs(vertices[:,:,:2]).max() * 1.05
            camera = pyrender.camera.OrthographicCamera(xmag, ymag, znear=0.05, zfar=100.0, name=None)
        scene.add(camera, pose=camera_pose)

        if len(triangles.shape) == 2:
            triangles = [triangles for _ in range(len(vertices))]

        # Create light source
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.5)
        # for every person in the scene
        for n in range(vertices.shape[0]):
            mesh = trimesh.Trimesh(vertices[n], triangles[n])
            mesh.apply_transform(self.rot)
            if mesh_colors is None:
                mesh_color = self.colors[n % len(self.colors)]
            else:
                mesh_color = mesh_colors[n % len(mesh_colors)]
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.2,
                alphaMode='OPAQUE',
                baseColorFactor=mesh_color)
            mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
            scene.add(mesh, 'mesh')

            add_light(scene, light)
        # Alpha channel was not working previously need to check again
        # Until this is fixed use hack with depth image to get the opacity
        color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)

        color = color.astype(np.float32)
        valid_mask = (rend_depth > 0)[:, :, None]
        output_image = (color[:, :, :3] * valid_mask +
                        (1 - valid_mask) * image).astype(np.uint8)

        return output_image, rend_depth

    def delete(self):
        self.renderer.delete()