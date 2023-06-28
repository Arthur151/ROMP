import cv2
import keyboard
import imageio
import torch
import numpy as np
import random
#import open3d as o3d
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
#from vedo import *
from multiprocessing import Process
from utils import save_obj
from utils.temporal_optimization import OneEuroFilter

#from visualization.vis_utils import create_mesh, create_mesh_with_uvmap
#from visualization.vis_server import get_uvs

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

def img_preprocess(image, imgpath, input_size=512, ds='internet', single_img_input=False):
    image = image[:,:,::-1]
    image_size = image.shape[:2][::-1]
    image_org = Image.fromarray(image)
    
    resized_image_size = (float(input_size)/max(image_size) * np.array(image_size) // 2 * 2).astype(np.int32)
    padding = tuple((input_size-resized_image_size)//2)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize([resized_image_size[1],resized_image_size[0]], interpolation=3),
        torchvision.transforms.Pad(padding, fill=0, padding_mode='constant'),
        #torchvision.transforms.ToTensor(),
        ])
    image = torch.from_numpy(np.array(transform(image_org))).float()

    padding_org = tuple((max(image_size)-np.array(image_size))//2)
    transform_org = torchvision.transforms.Compose([
        torchvision.transforms.Pad(padding_org, fill=0, padding_mode='constant'),
        torchvision.transforms.Resize((input_size*2, input_size*2), interpolation=3), #max(image_size)//2,max(image_size)//2
        #torchvision.transforms.ToTensor(),
        ])
    image_org = torch.from_numpy(np.array(transform_org(image_org)))
    padding_org = (np.array(list(padding_org))*float(input_size*2/max(image_size))).astype(np.int32)
    if padding_org[0]>0:
        image_org[:,:padding_org[0]] = 255 
        image_org[:,-padding_org[0]:] = 255
    if padding_org[1]>0:
        image_org[:padding_org[1]] = 255 
        image_org[-padding_org[1]:] = 255 

    offset = (max(image_size) - np.array(image_size))/2
    offsets = np.array([image_size[1],image_size[0],0,\
        resized_image_size[0]+padding[1],0,resized_image_size[1]+padding[0],offset[1],\
        resized_image_size[0],offset[0],resized_image_size[1],max(image_size)],dtype=np.int32)
    offsets = torch.from_numpy(offsets).float()

    name = os.path.basename(imgpath)

    if single_img_input:
        image = image.unsqueeze(0).contiguous()
        image_org = image_org.unsqueeze(0).contiguous()
        offsets = offsets.unsqueeze(0).contiguous()
        imgpath, name, ds = [imgpath], [name], [ds]
    input_data = {
        'image': image,
        'image_org': image_org,
        'imgpath': imgpath,
        'offsets': offsets,
        'name': name,
        'data_set':ds }
    return input_data



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

class Open3d_visualizer(object):
    def __init__(self, multi_mode=False):
        self.view_mat = axangle2mat([1, 0, 0], np.pi) # align different coordinate systems
        self.window_size = 1080
        
        smpl_param_dict = pickle.load(open(os.path.join(args().smpl_model_path,'smpl','SMPL_NEUTRAL.pkl'),'rb'), encoding='latin1')
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

        self.mesh_smoother = OneEuroFilter(4.0, 0.0)

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
        if args().cloth in constants.wardrobe or args().cloth=='random':
            uvs = get_uvs()
            if args().cloth=='random':
                cloth_id = random.sample(list(constants.wardrobe.keys()), 1)[0]
                print('choose cloth: ', cloth_id, constants.wardrobe[cloth_id])
                texture_file = os.path.join(args().wardrobe, constants.wardrobe[cloth_id])
            else:
                texture_file = os.path.join(args().wardrobe, constants.wardrobe[args().cloth])
            mesh = create_mesh_with_uvmap(vertices, self.faces, texture_path=texture_file, uvs=uvs)
        elif args().cloth in constants.mesh_color_dict:
            mesh_color = np.array(constants.mesh_color_dict[args().cloth])/255.
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

def frames2video(images_path, video_name, images=None, fps=30):
    writer = imageio.get_writer(video_name, format='mp4', mode='I', fps=fps)
    if images is None:
        for path in images_path:
            image = imageio.imread(path)
            writer.append_data(image)
    else:
        for image in images:
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
