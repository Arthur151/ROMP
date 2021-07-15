import cv2
import keyboard
import imageio
import torch
import numpy as np
import open3d as o3d
import pygame
from pygame.locals import *
from transforms3d.axangles import axangle2mat
import pickle
from PIL import Image
import torchvision
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

def get_video_bn(video_file_path):
    return os.path.basename(video_file_path)\
    .replace('.mp4', '').replace('.avi', '').replace('.webm', '').replace('.gif', '')

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
    
    resized_image_size = (float(input_size)/max(image_size) * np.array(image_size) // 2 * 2).astype(np.int)
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
    padding_org = (np.array(list(padding_org))*float(input_size*2/max(image_size))).astype(np.int)
    if padding_org[0]>0:
        image_org[:,:padding_org[0]] = 255 
        image_org[:,-padding_org[0]:] = 255
    if padding_org[1]>0:
        image_org[:padding_org[1]] = 255 
        image_org[-padding_org[1]:] = 255 

    offset = (max(image_size) - np.array(image_size))/2
    offsets = np.array([image_size[1],image_size[0],0,\
        resized_image_size[0]+padding[1],0,resized_image_size[1]+padding[0],offset[1],\
        resized_image_size[0],offset[0],resized_image_size[1],max(image_size)],dtype=np.int)
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


'''
learn from the minimal hand https://github.com/CalciferZh/minimal-hand
'''
class LowPassFilter:
  def __init__(self):
    self.prev_raw_value = None
    self.prev_filtered_value = None

  def process(self, value, alpha):
    if self.prev_raw_value is None:
      s = value
    else:
      s = alpha * value + (1.0 - alpha) * self.prev_filtered_value
    self.prev_raw_value = value
    self.prev_filtered_value = s
    return s

class OneEuroFilter:
  def __init__(self, mincutoff=1.0, beta=0.0, dcutoff=1.0, freq=30):
    self.freq = freq
    self.mincutoff = mincutoff
    self.beta = beta
    self.dcutoff = dcutoff
    self.x_filter = LowPassFilter()
    self.dx_filter = LowPassFilter()

  def compute_alpha(self, cutoff):
    te = 1.0 / self.freq
    tau = 1.0 / (2 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / te)

  def process(self, x):
    prev_x = self.x_filter.prev_raw_value
    dx = 0.0 if prev_x is None else (x - prev_x) * self.freq
    edx = self.dx_filter.process(dx, self.compute_alpha(self.dcutoff))
    cutoff = self.mincutoff + self.beta * np.abs(edx)
    return self.x_filter.process(x, self.compute_alpha(cutoff))

class OpenCVCapture:
    def __init__(self, video_file=None):
        if video_file is None:
            self.cap = cv2.VideoCapture(int(args().cam_id))
        else:
            self.cap = cv2.VideoCapture(video_file)

    def read(self):
        flag, frame = self.cap.read()
        if not flag:
          return None
        return np.flip(frame, -1).copy() # BGR to RGB

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

class Open3d_visualizer(object):
    def __init__(self):
        self.view_mat = axangle2mat([1, 0, 0], np.pi) # align different coordinate systems
        self.window_size = 1080
        
        smpl_param_dict = pickle.load(open(os.path.join(args().smpl_model_path,'smpl','SMPL_NEUTRAL.pkl'),'rb'), encoding='latin1')
        self.faces = smpl_param_dict['f']
        self.verts_mean = smpl_param_dict['v_template']
        self.mesh_color = np.array(constants.mesh_color_dict[args().webcam_mesh_color])/255.

        self.viewer = o3d.visualization.Visualizer()
        self.viewer.create_window(width=self.window_size+1, height=self.window_size+1, window_name='ROMP - output')
        self.mesh = self.create_single_mesh(self.verts_mean)
        self.viewer.add_geometry(self.mesh)

        view_control = self.viewer.get_view_control()
        cam_params = view_control.convert_to_pinhole_camera_parameters()
        extrinsic = cam_params.extrinsic.copy()
        extrinsic[0:3, 3] = 0
        cam_params.extrinsic = extrinsic
        self.count = 0

        #cam_params.intrinsic.set_intrinsics(
        #  self.window_size, self.window_size, 620.744, 621.151,
        #  self.window_size//2, self.window_size//2
        #)
        view_control.convert_from_pinhole_camera_parameters(cam_params)
        view_control.set_constant_z_far(1000)

        render_option = self.viewer.get_render_option()
        render_option.load_from_json('lib/utils/render_option.json')
        self.viewer.update_renderer()

        self.mesh_smoother = OneEuroFilter(4.0, 0.0)

        ############ input visualization ############
        pygame.init()
        self.display = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption('ROMP - input')

    def run(self, verts,frame):
        verts = self.mesh_smoother.process(verts)
        verts = np.matmul(self.view_mat, verts.T).T
        self.mesh.vertices = o3d.utility.Vector3dVector(verts)
        self.mesh.compute_triangle_normals()
        self.mesh.compute_vertex_normals()

        # for some version of open3d you may need `viewer.update_geometry(mesh)`
        self.viewer.update_geometry(self.mesh)

        self.viewer.poll_events()
        self.display.blit(
          pygame.surfarray.make_surface(
            np.transpose(cv2.resize(frame, (self.window_size, self.window_size), cv2.INTER_LINEAR), (1, 0, 2))),(0, 0))
        pygame.display.update()
        for event in pygame.event.get():
            if (event.type == KEYUP) or (event.type == KEYDOWN):
                print('key pressed')
                return True
        return False

    def run_multiperson(self, verts,frame):
        geometries = []
        #self.reset_mesh()
        self.viewer.destroy_window()
        self.viewer = o3d.visualization.Visualizer()
        self.viewer.create_window()
        for v_id, vert in enumerate(verts):
            #self.mesh += self.create_single_mesh(vert)
            #geometries.append(self.create_single_mesh(vert))
            self.viewer.add_geometry(self.create_single_mesh(vert))
        self.viewer.poll_events()
        self.viewer.update_renderer()
        #o3d.visualization.draw_geometries(geometries)
        #self.viewer.update_geometry(self.mesh)

        self.display.blit(
          pygame.surfarray.make_surface(
            np.transpose(cv2.resize(frame, (self.window_size, self.window_size), cv2.INTER_LINEAR), (1, 0, 2))),(0, 0))
        pygame.display.update()
        for event in pygame.event.get():
            if (event.type == KEYUP) or (event.type == KEYDOWN):
                print('key pressed')
                return True
        return False

    def create_single_mesh(self, verts):
        mesh = o3d.geometry.TriangleMesh()
        mesh.triangles = o3d.utility.Vector3iVector(self.faces)
        mesh.vertices = o3d.utility.Vector3dVector(np.matmul(self.view_mat, verts.T).T)
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()
        return mesh

    def reset_mesh(self):
        self.mesh.triangles = o3d.utility.Vector3iVector(self.faces)
        self.mesh.vertices = o3d.utility.Vector3dVector(self.verts_mean)
        #self.mesh = self.set_texture(self.mesh)
        self.mesh.compute_vertex_normals()

    def set_texture(self, mesh):
        if args().webcam_mesh_color=='female_tex' or args().webcam_mesh_color=='male_tex':
            print('setting texture')
            mesh.triangle_uvs = o3d.utility.Vector2dVector(self.smpl_uvmap) 
            mesh.textures = [o3d.geometry.Image(self.smpl_uv_texture)]
        else:
            mesh.paint_uniform_color(self.mesh_color)
        return mesh
        

def frames2video(images, video_name,fps=30):
    writer = imageio.get_writer(video_name, format='mp4', mode='I', fps=fps)

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
        mesh_vtk = Mesh(os.path.join(args().smpl_model_path,'smpl','smpl_male.vtk'), c='w').texture(self.texture_file).lw(0.1)
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
