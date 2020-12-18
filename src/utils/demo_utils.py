import cv2
import keyboard
import imageio
import numpy as np
import open3d as o3d
import pygame
from pygame.locals import *
from transforms3d.axangles import axangle2mat
import pickle
import sys,os
sys.path.append(os.path.abspath(__file__).replace('core/demo.py',''))
import config
import constants
from config import args

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
            self.cap = cv2.VideoCapture(int(args.cam_id))
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
        
        self.mesh_color = np.array(constants.mesh_color_dict[args.webcam_mesh_color])/255.
        smpl_param_dict = pickle.load(open(os.path.join(args.smpl_model_path,'smpl','SMPL_NEUTRAL.pkl'),'rb'), encoding='latin1')
        self.faces = smpl_param_dict['f']
        self.verts_mean = smpl_param_dict['v_template']

        self.viewer = o3d.visualization.Visualizer()
        self.viewer.create_window(width=self.window_size+1, height=self.window_size+1, window_name='CenterHMR - output')
        self.mesh = self.create_single_mesh(self.verts_mean)
        self.viewer.add_geometry(self.mesh)

        view_control = self.viewer.get_view_control()
        cam_params = view_control.convert_to_pinhole_camera_parameters()
        extrinsic = cam_params.extrinsic.copy()
        extrinsic[0:3, 3] = 0
        cam_params.extrinsic = extrinsic

        #cam_params.intrinsic.set_intrinsics(
        #  self.window_size, self.window_size, 620.744, 621.151,
        #  self.window_size//2, self.window_size//2
        #)
        view_control.convert_from_pinhole_camera_parameters(cam_params)
        view_control.set_constant_z_far(1000)

        render_option = self.viewer.get_render_option()
        render_option.load_from_json('utils/render_option.json')
        self.viewer.update_renderer()

        self.mesh_smoother = OneEuroFilter(4.0, 0.0)

        ############ input visualization ############
        pygame.init()
        self.display = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption('CenterHMR - input')

    def run(self, verts,frame):
        verts = self.mesh_smoother.process(verts)
        self.mesh.vertices = o3d.utility.Vector3dVector(np.matmul(self.view_mat, verts.T).T)
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
        mesh.paint_uniform_color(self.mesh_color)
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()
        return mesh

    def reset_mesh(self):
        self.mesh.triangles = o3d.utility.Vector3iVector(self.faces)
        self.mesh.vertices = o3d.utility.Vector3dVector(self.verts_mean)
        self.mesh.paint_uniform_color(self.mesh_color)
        self.mesh.compute_vertex_normals()
        

def frames2video(images, video_name,fps=30):
    writer = imageio.get_writer(video_name, format='mp4', mode='I', fps=fps)

    for image in images:
        writer.append_data(image)
    writer.close()