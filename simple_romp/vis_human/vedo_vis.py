import os,sys
import vedo
from vedo import *
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
import pickle
import cv2

def convert_cam_to_stand_on_image_trans(trans_3d, enlarge_scale=3):
    stand_on_image_trans = np.zeros(3)
    # The x-axis is supposed to be adapted to the proper scale
    stand_on_image_trans[0] = trans_3d[0] * 0.3
    stand_on_image_trans[1] = 0.6 #0.46 #0.5 - trans_3d[1] * 0.2
    stand_on_image_trans[2] = trans_3d[1] * 0.4 #- trans_3d[2]/3  0.4
    stand_on_image_trans *= enlarge_scale
    return stand_on_image_trans

def parse_nvxia_uvmap(uvs, face):
    verts_num = np.max(face) + 1
    uvs_verts = np.zeros((verts_num, 2))
    for uv, f in zip(uvs, face):
        uvs_verts[f] = uv[:,:2]
    #uvs_verts[:,1] = 1-uvs_verts[:,1]
    return uvs_verts


class Vedo_visualizer(object):
    def __init__(self, character='smpl'):
        if character == 'smpl':
            self.faces = pickle.load(open(os.path.join(smpl_model_path,'SMPL_NEUTRAL.pkl'),'rb'), encoding='latin1')['f']
        elif character == 'nvxia':
            params_dict = np.load(os.path.join(nvxia_model_path, 'nvxia.npz'), allow_pickle=True)
            self.faces = np.array([np.array(face) for face in params_dict['polygons']])
            self.texture_file = cv2.imread(os.path.join(nvxia_model_path, 'Kachujin_diffuse.png'))[:,:,::-1]
            self.uvs = parse_nvxia_uvmap(params_dict['uvmap'],self.faces)
        self.scene_bg_color = [240,255,255]
        self.default_camera={'pos':{'far':(0,800,1000), 'close':(0,200,800)}[soi_camera]} 
        # Light([0,800,1000], c='white')
        self.lights = [Light([0,800,1000], intensity=0.6, c='white'), Light([0,-800,1000], intensity=0.6, c='white'), Light([0,800,-1000], intensity=0.6, c='white'), Light([0,-800,-1000], intensity=0.6, c='white')]
        vedo.settings.screeshotLargeImage = True
        vedo.settings.screeshotScale = 2

    def plot_multi_meshes_batch(self, vertices, cam_params, meta_data, reorganize_idx, save_img=True, interactive_show=False, rotate_frames=[]):
        result_save_names = []
        for inds, img_id in enumerate(np.unique(reorganize_idx)):
            single_img_verts_inds = np.array(np.where(reorganize_idx==img_id)[0])
            
            plt = self.plot_multi_meshes(vertices[single_img_verts_inds].detach().cpu().numpy(), \
                cam_params[single_img_verts_inds].detach().cpu().numpy(), meta_data['image'][single_img_verts_inds[0]].cpu().numpy().astype(np.uint8),\
                interactive_show=interactive_show)

            if img_id in rotate_frames:
                result_imgs, rot_angles = self.render_rotating(plt)
                save_names = [os.path.join(output_dir, '3D_meshes-'+os.path.basename(meta_data['imgpath'][single_img_verts_inds[0]]+'_{:03d}.jpg'.format(ra))) for ra in rot_angles]
            else:
                result_imgs = self.render_one_time(plt, self.default_camera)
                save_names = [os.path.join(output_dir, '3D_meshes-'+os.path.basename(meta_data['imgpath'][single_img_verts_inds[0]]+'.jpg'))]

            plt.close()
            result_save_names += save_names
            if save_img:
                for save_name, result_img in zip(save_names, result_imgs):
                    cv2.imwrite(save_name, result_img[:,:,::-1])
            
        return result_save_names

    def plot_multi_meshes(self, vertices, cam_params, img, mesh_colors=None, interactive_show=False, rotate_cam=False):
        plt = Plotter(bg=[240,255,255], axes=0, offscreen=not interactive_show)
        h,w = img.shape[:2]
        pic = Picture(img)
        
        pic.rotateX(-90).z(h//2).x(-w//2)
        verts_enlarge_scale = max(h,w)/5
        cam_enlarge_scale = max(h,w)/3

        plt += pic
        vertices_vis = []

        for inds, (vert, cam) in enumerate(zip(vertices, cam_params)):
            trans_3d = convert_cam_to_stand_on_image_trans(cam, cam_enlarge_scale)#enlarge_scale
            vert[:,1:] *= -1
            vert = vert * verts_enlarge_scale
            vert += trans_3d[None]
            vertices_vis.append(vert)
        vertices_vis = np.stack(vertices_vis, 0)
        
        visulize_list = []
        for inds, vert in enumerate(vertices_vis):
            mesh = Mesh([vert, self.faces])
            if character == 'smpl':
                mesh = mesh.c([255,255,255]).smooth(niter=20)
                if mesh_colors is not None:
                    mesh.c(mesh_colors[inds].astype(np.uint8))
            elif character == 'nvxia':
                mesh.texture(self.texture_file,tcoords=self.uvs).smooth(niter=20)#.lighting('glossy')
            visulize_list.append(mesh)
        plt += visulize_list
        for light in self.lights:
            plt += light
        return plt

    def render_rotating(self, plt, internal=5):
        result_imgs = []
        pause_num = fps_save
        pause = np.zeros(pause_num).astype(np.int32)
        change_time = 90//internal
        roates = np.ones(change_time) * internal
        go_up = np.sin(np.arange(change_time).astype(np.float32)/change_time) * 1
        go_down = np.sin(np.arange(change_time).astype(np.float32)/change_time - 1) * 1
        #top2front = np.ones(pause_num) * -((90-30)/pause_num)
        azimuth_angles = np.concatenate([pause, roates, roates, roates, roates])
        elevation_angles = np.concatenate([pause, go_up, go_down, go_up, go_down])
        #rot_angles = np.concatenate([pause, roates, pause, roates, pause, roates, pause, roates, pause])
        plt.camera.Elevation(20)
        for rid, azimuth_angle in enumerate(azimuth_angles):
            # if rid==pause_num:
            #     plt.camera.Elevation(30)
            #     plt.camera.Azimuth(0)
            plt.show(azimuth=azimuth_angle, elevation=elevation_angles[rid])
            result_img = plt.topicture(scale=2)
            rows, cols, _ = result_img._data.GetDimensions()
            vtkimage = result_img._data.GetPointData().GetScalars()
            image_result = vtk_to_numpy(vtkimage).reshape((rows, cols, 3))
            result_imgs.append(image_result[::-1])
        return result_imgs, np.arange(len(azimuth_angles))

    def render_one_time(self, plt, camera_pose):
        image_result = plt.show(camera=camera_pose) #elevation=10,azimuth=0,,bg=self.bg_path 
        result_img = plt.topicture(scale=2)
        rows, cols, _ = result_img._data.GetDimensions()
        vtkimage = result_img._data.GetPointData().GetScalars()
        image_result = vtk_to_numpy(vtkimage).reshape((rows, cols, 3))
        image_result = image_result[::-1]
        #result_img.write(save_name)
        #screenshot(save_name) #returnNumpy=True
        
        return [image_result]