import os,sys
import vedo
from vedo import *
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
import pickle
import cv2

import config
import constants
from config import args
from utils.temporal_optimization import OneEuroFilter
from utils.cam_utils import convert_cam_to_3d_trans

def convert_cam_to_stand_on_image_trans(cam, enlarge_scale=3):
    trans_3d = convert_cam_to_3d_trans(cam)
    stand_on_image_trans = np.zeros(3)
    # The x-axis is supposed to be adapted to the proper scale
    stand_on_image_trans[0] = trans_3d[0] * 0.3
    stand_on_image_trans[1] = 0.6 #0.46 #0.5 - trans_3d[1] * 0.2
    #stand_on_image_trans[1] = 0.56
    #stand_on_image_trans[2] = trans_3d[1] - trans_3d[2]/3 + 2.6
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
    def __init__(self):
        if args().character == 'smpl':
            self.faces = pickle.load(open(os.path.join(args().smpl_model_path,'SMPL_NEUTRAL.pkl'),'rb'), encoding='latin1')['f']
        elif args().character == 'nvxia':
            params_dict = np.load(os.path.join(args().nvxia_model_path, 'nvxia.npz'), allow_pickle=True)
            self.faces = np.array([np.array(face) for face in params_dict['polygons']])
            self.texture_file = cv2.imread(os.path.join(args().nvxia_model_path, 'Kachujin_diffuse.png'))[:,:,::-1]
            self.uvs = parse_nvxia_uvmap(params_dict['uvmap'],self.faces)
        self.scene_bg_color = [240,255,255]
        self.default_camera={'pos':{'far':(0,800,1000), 'close':(0,200,800)}[args().soi_camera]} 
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
                save_names = [os.path.join(args().output_dir, '3D_meshes-'+os.path.basename(meta_data['imgpath'][single_img_verts_inds[0]]+'_{:03d}.jpg'.format(ra))) for ra in rot_angles]
            else:
                result_imgs = self.render_one_time(plt, self.default_camera)
                save_names = [os.path.join(args().output_dir, '3D_meshes-'+os.path.basename(meta_data['imgpath'][single_img_verts_inds[0]]+'.jpg'))]

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
            if args().character == 'smpl':
                mesh = mesh.c([255,255,255]).smooth(niter=20)
                if mesh_colors is not None:
                    mesh.c(mesh_colors[inds].astype(np.uint8))
            elif args().character == 'nvxia':
                mesh.texture(self.texture_file,tcoords=self.uvs).smooth(niter=20)#.lighting('glossy')
            visulize_list.append(mesh)
        plt += visulize_list
        for light in self.lights:
            plt += light
        return plt

    def render_rotating(self, plt, internal=5):
        result_imgs = []
        pause_num = args().fps_save
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

# class vedo_visualizer(object):
#     def __init__(self):  
#         smpl_param_dict = pickle.load(open(os.path.join(args().smpl_model_path,'smpl','SMPL_NEUTRAL.pkl'),'rb'), encoding='latin1')
#         self.faces = smpl_param_dict['f']
#         #self.verts_mean = smpl_param_dict['v_template']
#         #self.load_smpl_tex()
#         #self.load_smpl_vtk()
#         if args().webcam_mesh_color == 'female_tex':
#             self.uv_map = np.load(args().smpl_uvmap)
#             self.texture_file = args().smpl_female_texture
#         elif args().webcam_mesh_color == 'male_tex':
#             self.uv_map = np.load(args().smpl_uvmap)
#             self.texture_file = args().smpl_male_texture
#         else:
#             self.mesh_color = np.array(constants.mesh_color_dict[args().webcam_mesh_color])/255.

#         #self.mesh = self.create_smpl_mesh(self.verts_mean)
#         self.mesh_smoother = OneEuroFilter(4.0, 0.0)
#         self.vp = Plotter(title='Predicted 3D mesh',interactive=0)#
#         self.vp_2d = Plotter(title='Input frame',interactive=0)
#         #show(self.mesh, axes=1, viewup="y", interactive=0)
    
#     def load_smpl_tex(self):
#         import scipy.io as sio
#         UV_info = sio.loadmat(os.path.join(args().smpl_model_path,'smpl','UV_Processed.mat'))
#         self.vertex_reorder = UV_info['All_vertices'][0]-1
#         self.faces = UV_info['All_Faces']-1
#         self.uv_map = np.concatenate([UV_info['All_U_norm'], UV_info['All_V_norm']],1)

#     def run(self, verts,frame):
#         verts[:,1:] = verts[:,1:]*-1
#         verts = self.mesh_smoother.process(verts)
#         #verts = verts[self.vertex_reorder]
#         #self.mesh.points(verts)
#         mesh = self.create_smpl_mesh(verts)
#         self.vp.show(mesh,viewup=np.array([0,-1,0]))
#         self.vp_2d.show(Picture(frame))
        
#         return False

#     def create_smpl_mesh(self, verts):
#         mesh = Mesh([verts, self.faces])
#         mesh.texture(self.texture_file,tcoords=self.uv_map)
#         mesh = self.collapse_triangles_with_large_gradient(mesh)
#         mesh.computeNormals()
#         return mesh

#     def collapse_triangles_with_large_gradient(self, mesh, threshold=4.0):
#         points = mesh.points()
#         new_points = np.array(points)
#         mesh_vtk = Mesh(os.path.join(args().smpl_model_path,'smpl_male.vtk'), c='w').texture(self.texture_file).lw(0.1)
#         grad = mesh_vtk.gradient("tcoords")
#         ugrad, vgrad = np.split(grad, 2, axis=1)
#         ugradm, vgradm = mag(ugrad), mag(vgrad)
#         gradm = np.log(ugradm*ugradm + vgradm*vgradm)

#         largegrad_ids = np.arange(mesh.N())[gradm>threshold]
#         for f in mesh.faces():
#             if np.isin(f, largegrad_ids).all():
#                 id1, id2, id3 = f
#                 uv1, uv2, uv3 = self.uv_map[f]
#                 d12 = mag(uv1-uv2)
#                 d23 = mag(uv2-uv3)
#                 d31 = mag(uv3-uv1)
#                 idm = np.argmin([d12, d23, d31])
#                 if idm == 0: # d12, collapse segment to pt3
#                     new_points[id1] = new_points[id3]
#                     new_points[id2] = new_points[id3]
#                 elif idm == 1: # d23
#                     new_points[id2] = new_points[id1]
#                     new_points[id3] = new_points[id1]
#         mesh.points(new_points)
#         return mesh