import cv2
import keyboard
import imageio
import torch
import numpy as np
import random
import open3d as o3d
from transforms3d.axangles import axangle2mat
import pickle
from PIL import Image
import torchvision
import time
import sys, os
root_dir = os.path.join(os.path.dirname(__file__),'..')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
import config
import constants
from config import args
from vedo import *
from multiprocessing import Process
from utils import save_obj
from utils.temporal_optimization import OneEuroFilter

version = int(o3d.__version__.split('.')[1])
# if version==9:
#     print('using open3d 0.9.0, importing functions from vis_utils_py36_o3d9.')
#     from visualization.vis_utils_py36_o3d9 import create_mesh, create_mesh_with_uvmap, get_uvs
# elif version >=11:
#     print('using open3d {}, importing functions from vis_utils_o3d13.'.format(version))
#     from visualization.vis_utils_o3d13 import create_mesh, create_mesh_with_uvmap, get_uvs
# else:
#     print('Error: the open3d version may not be supported.')
    
def get_video_bn(video_file_path):
    basename = os.path.basename(video_file_path)
    for ext in constants.video_exts:
        basename.replace(ext, '')
    return basename

def save_meshes(reorganize_idx, outputs, output_dir, smpl_faces):
    vids_org = np.unique(reorganize_idx)
    for idx, vid in enumerate(vids_org):
        verts_vids = np.where(reorganize_idx==vid)[0]
        img_path = outputs['meta_data']['imgpath'][verts_vids[0]]
        obj_name = os.path.join(output_dir, '{}'.format(os.path.basename(img_path))).replace('.mp4','').replace('.jpg','').replace('.png','')+'.obj'
        for subject_idx, batch_idx in enumerate(verts_vids):
            save_obj(outputs['verts'][batch_idx].detach().cpu().numpy().astype(np.float16), \
                smpl_faces,obj_name.replace('.obj', '_{}.obj'.format(subject_idx)))

def convert_cam_to_3d_trans(cams, weight=2.):
    trans3d = []
    (s, tx, ty) = cams
    depth, dx, dy = 1./s, tx/s, ty/s
    trans3d = np.array([dx, dy, depth])*weight
    return trans3d

class OpenCVCapture:
    def __init__(self, video_file=None, show=False):
        if video_file is None:
            self.cap = cv2.VideoCapture(int(args().cam_id))
        else:
            self.cap = cv2.VideoCapture(video_file)
            self.length = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.whether_to_show=show


    def read(self, return_rgb=True):
        flag, frame = self.cap.read()
        if not flag:
          return None
        if self.whether_to_show:
            cv2.imshow('webcam',frame)
            cv2.waitKey(1)
        if return_rgb:
            frame = np.flip(frame, -1).copy() # BGR to RGB
        return frame

class Image_Reader:
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.image_list = os.listdir(self.image_folder)
        self.current_num=0

    def read(self):
        frame = cv2.imread(os.path.join(self.image_folder,self.image_list[self.current_num]))
        self.current_num+=1
        if self.current_num==len(self.image_list):
            self.current_num=0
        return np.flip(frame, -1).copy() # BGR to RGB


class Time_counter():
    def __init__(self,thresh=0.1):
        self.thresh=thresh
        self.runtime = 0
        self.frame_num = 0

    def start(self):
        self.start_time = time.time()

    def count(self, frame_num=1):
        time_cost = time.time()-self.start_time
        if time_cost<self.thresh:
            self.runtime+=time_cost
            self.frame_num+=frame_num
        self.start()

    def fps(self):
        print('average per-frame runtime:',self.runtime/self.frame_num)
        print('FPS: {}, not including visualization time. '.format(self.frame_num/self.runtime))

    def reset(self):
        self.runtime = 0
        self.frame_num = 0

def get_uvs(uvmap_path):
    uv_map_vt_ft = np.load(uvmap_path, allow_pickle=True)
    vt, ft = uv_map_vt_ft['vt'], uv_map_vt_ft['ft']
    uvs = np.concatenate([vt[ft[:,ind]][:,None] for ind in range(3)],1).reshape(-1,2)
    uvs[:,1] = 1-uvs[:,1]
    return uvs

Vector3dVector = o3d.utility.Vector3dVector
Vector3iVector = o3d.utility.Vector3iVector
Vector2iVector = o3d.utility.Vector2iVector
TriangleMesh = o3d.geometry.TriangleMesh

def create_mesh_with_uvmap(vertices, faces, texture_path=None, uvs=None, **kwargs):
    mesh = TriangleMesh()
    mesh.vertices = Vector3dVector(vertices)
    mesh.triangles = Vector3iVector(faces)
    if texture_path is not None and uvs is not None:
        mesh.texture = o3d.io.read_image(texture_path)
        mesh.triangle_uvs = uvs
    mesh.compute_vertex_normals()
    return mesh

def create_mesh(vertices, faces, colors=None, **kwargs):
    mesh = TriangleMesh()
    mesh.vertices = Vector3dVector(vertices)
    mesh.triangles = Vector3iVector(faces)
    if colors is not None:
        mesh.vertex_colors = Vector3dVector(colors)
    else:
        mesh.paint_uniform_color([1., 0.8, 0.8])
    mesh.compute_vertex_normals()
    return mesh

class Open3d_visualizer(object):
    def __init__(self, multi_mode=False):
        self.view_mat = axangle2mat([1, 0, 0], np.pi) # align different coordinate systems
        self.window_size = 1080
        
        smpl_param_dict = pickle.load(open(os.path.join(args().smpl_model_path,'SMPL_NEUTRAL.pkl'),'rb'), encoding='latin1')
        self.faces = smpl_param_dict['f']
        self.verts_mean = smpl_param_dict['v_template']
        # self.mesh_color = np.array(constants.mesh_color_dict[args().webcam_mesh_color])/255.

        self.viewer = o3d.visualization.Visualizer()
        self.viewer.create_window(width=self.window_size+1, height=self.window_size+1, window_name='ROMP - output')
        if multi_mode:
            self.current_mesh_num = 10
            self.zero_vertices = o3d.utility.Vector3dVector(np.zeros((6890,3)))
            self.meshes = []
            for _ in range(self.current_mesh_num):
                new_mesh = self.create_single_mesh(self.verts_mean)
                self.meshes.append(new_mesh)
            self.set_meshes_zero(list(range(self.current_mesh_num)))
        else:
            self.mesh = self.create_single_mesh(self.verts_mean)
        
        view_control = self.viewer.get_view_control()
        cam_params = view_control.convert_to_pinhole_camera_parameters()
        extrinsic = cam_params.extrinsic.copy()
        extrinsic[0:3, 3] = 0
        cam_params.extrinsic = extrinsic
        self.count = 0

        view_control.convert_from_pinhole_camera_parameters(cam_params)
        view_control.set_constant_z_far(1000)

        render_option = self.viewer.get_render_option()
        render_option.load_from_json('romp/lib/visualization/vis_cfgs/render_option.json')
        self.viewer.update_renderer()

        self.mesh_smoother = OneEuroFilter(1., 0.0)

    def set_meshes_zero(self, mesh_ids):
        for ind in mesh_ids:
            self.meshes[ind].vertices = self.zero_vertices

    def run(self, verts):
        verts = self.mesh_smoother.process(verts)
        verts = np.matmul(self.view_mat, verts.T).T
        self.mesh.vertices = o3d.utility.Vector3dVector(verts)
        self.mesh.compute_triangle_normals()
        self.mesh.compute_vertex_normals()

        # for some version of open3d you may need `viewer.update_geometry(mesh)`
        self.viewer.update_geometry(self.mesh)
        self.viewer.poll_events()

    def run_multiperson(self, verts):
        print('recieving {} people'.format(len(verts)))
        geometries = []
        for v_id, vert in enumerate(verts):
            #self.mesh += self.create_single_mesh(vert)
            self.meshes[v_id].vertices = o3d.utility.Vector3dVector(vert)
            self.meshes[v_id].compute_triangle_normals()
            #self.meshes[v_id].compute_vertex_normals()
            self.viewer.update_geometry(self.meshes[v_id])
        
        self.viewer.poll_events()
        self.viewer.update_renderer()

    def create_single_mesh(self, vertices):
        if args().mesh_cloth in constants.wardrobe or args().mesh_cloth=='random':
            uvs = get_uvs(args().smpl_uvmap)
            if args().mesh_cloth=='random':
                mesh_cloth_id = random.sample(list(constants.wardrobe.keys()), 1)[0]
                print('choose mesh_cloth: ', mesh_cloth_id, constants.wardrobe[mesh_cloth_id])
                texture_file = os.path.join(args().wardrobe, constants.wardrobe[mesh_cloth_id])
            else:
                texture_file = os.path.join(args().wardrobe, constants.wardrobe[args().mesh_cloth])
            mesh = create_mesh_with_uvmap(vertices, self.faces, texture_path=texture_file, uvs=uvs)
        elif args().mesh_cloth in constants.mesh_color_dict:
            mesh_color = np.array(constants.mesh_color_dict[args().mesh_cloth])/255.
            mesh = create_mesh(vertices=vertices, faces=self.faces, colors=mesh_color)
        else:
            mesh = create_mesh(vertices=vertices, faces=self.faces)
        self.viewer.add_geometry(mesh)
        return mesh

def video2frame(video_name, frame_save_dir=None):
    cap = OpenCVCapture(video_name)
    os.makedirs(frame_save_dir, exist_ok=True)
    frame_list = []
    for frame_id in range(int(cap.length)):
        frame = cap.read(return_rgb=False)
        save_path = os.path.join(frame_save_dir, '{:06d}.jpg'.format(frame_id))
        cv2.imwrite(save_path, frame)
        frame_list.append(save_path)
    return frame_list

def frames2video(images_path, video_name,fps=30):
    writer = imageio.get_writer(video_name, format='mp4', mode='I', fps=fps)
    for path in images_path:
        image = imageio.imread(path)
        writer.append_data(image)
    writer.close()

class vedo_visualizer(object):
    def __init__(self):  
        smpl_param_dict = pickle.load(open(os.path.join(args().smpl_model_path,'smpl','SMPL_NEUTRAL.pkl'),'rb'), encoding='latin1')
        self.faces = smpl_param_dict['f']
        #self.verts_mean = smpl_param_dict['v_template']
        #self.load_smpl_tex()
        #self.load_smpl_vtk()
        if args().webcam_mesh_color == 'female_tex':
            self.uv_map = np.load(args().smpl_uvmap)
            self.texture_file = args().smpl_female_texture
        elif args().webcam_mesh_color == 'male_tex':
            self.uv_map = np.load(args().smpl_uvmap)
            self.texture_file = args().smpl_male_texture
        else:
            self.mesh_color = np.array(constants.mesh_color_dict[args().webcam_mesh_color])/255.

        #self.mesh = self.create_single_mesh(self.verts_mean)
        self.mesh_smoother = OneEuroFilter(4.0, 0.0)
        self.vp = Plotter(title='Predicted 3D mesh',interactive=0)#
        self.vp_2d = Plotter(title='Input frame',interactive=0)
        #show(self.mesh, axes=1, viewup="y", interactive=0)
    
    def load_smpl_tex(self):
        import scipy.io as sio
        UV_info = sio.loadmat(os.path.join(args().smpl_model_path,'smpl','UV_Processed.mat'))
        self.vertex_reorder = UV_info['All_vertices'][0]-1
        self.faces = UV_info['All_Faces']-1
        self.uv_map = np.concatenate([UV_info['All_U_norm'], UV_info['All_V_norm']],1)

    def run(self, verts,frame):
        verts[:,1:] = verts[:,1:]*-1
        verts = self.mesh_smoother.process(verts)
        #verts = verts[self.vertex_reorder]
        #self.mesh.points(verts)
        mesh = self.create_single_mesh(verts)
        self.vp.show(mesh,viewup=np.array([0,-1,0]))
        self.vp_2d.show(Picture(frame))
        
        return False

    def create_single_mesh(self, verts):
        mesh = Mesh([verts, self.faces])
        mesh.texture(self.texture_file,tcoords=self.uv_map)
        mesh = self.collapse_triangles_with_large_gradient(mesh)
        mesh.computeNormals()
        return mesh

    def collapse_triangles_with_large_gradient(self, mesh, threshold=4.0):
        points = mesh.points()
        new_points = np.array(points)
        mesh_vtk = Mesh(os.path.join(args().smpl_model_path,'smpl_male.vtk'), c='w').texture(self.texture_file).lw(0.1)
        grad = mesh_vtk.gradient("tcoords")
        ugrad, vgrad = np.split(grad, 2, axis=1)
        ugradm, vgradm = mag(ugrad), mag(vgrad)
        gradm = np.log(ugradm*ugradm + vgradm*vgradm)

        largegrad_ids = np.arange(mesh.N())[gradm>threshold]
        for f in mesh.faces():
            if np.isin(f, largegrad_ids).all():
                id1, id2, id3 = f
                uv1, uv2, uv3 = self.uv_map[f]
                d12 = mag(uv1-uv2)
                d23 = mag(uv2-uv3)
                d31 = mag(uv3-uv1)
                idm = np.argmin([d12, d23, d31])
                if idm == 0: # d12, collapse segment to pt3
                    new_points[id1] = new_points[id3]
                    new_points[id2] = new_points[id3]
                elif idm == 1: # d23
                    new_points[id2] = new_points[id1]
                    new_points[id3] = new_points[id1]
        mesh.points(new_points)
        return mesh
