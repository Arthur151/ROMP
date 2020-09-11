import pickle
import time
import cv2
import keyboard
import numpy as np
import open3d as o3d
import pygame
from pygame.locals import *
from transforms3d.axangles import axangle2mat
import pickle
import sys,os
from multiprocessing.connection import Listener
from multiprocessing.connection import Client


smpl_model_path = os.path.join('../models/smpl/SMPL_NEUTRAL.pkl','SMPL_NEUTRAL.pkl')
render_option_path = 'utils/render_option.json'

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
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

    def read(self):
        flag, frame = self.cap.read()
        if not flag:
          return None
        return frame

class Open3d_visualizer(object):
    def __init__(self):
        self.view_mat = axangle2mat([1, 0, 0], np.pi) # align different coordinate systems
        self.window_size = 1080
        self.mesh_COLOR = [225/255, 250/255, 250/255] # light purple 230,230,250; 
        smpl_param_dict = pickle.load(open(smpl_model_path,'rb'), encoding='latin1')
        self.faces = smpl_param_dict['f']
        verts_mean = smpl_param_dict['v_template']
        self.mesh = o3d.geometry.TriangleMesh()
        self.mesh.triangles = o3d.utility.Vector3iVector(self.faces)
        self.mesh.vertices = o3d.utility.Vector3dVector(np.matmul(self.view_mat, verts_mean.T).T * 1000)
        self.mesh.compute_vertex_normals()

        self.viewer = o3d.visualization.Visualizer()
        self.viewer.create_window(
          width=self.window_size+1, height=self.window_size+1,
          window_name='CenterHMR - output')
        self.viewer.add_geometry(self.mesh)

        view_control = self.viewer.get_view_control()
        cam_params = view_control.convert_to_pinhole_camera_parameters()
        extrinsic = cam_params.extrinsic.copy()
        extrinsic[0:3, 3] = 0
        cam_params.extrinsic = extrinsic
        #cam_params.intrinsic.set_intrinsics(
        #  self.window_size, self.window_size, fx, fy,
        #  self.window_size//2, self.window_size//2
        #)
        view_control.convert_from_pinhole_camera_parameters(cam_params)
        view_control.set_constant_z_far(1000)

        render_option = self.viewer.get_render_option()
        render_option.load_from_json(render_option_path)
        self.viewer.update_renderer()
        self.mesh_smoother = OneEuroFilter(4.0, 0.0)

        ############ input visualization ############
        pygame.init()
        self.display = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption('CenterHMR - input')

    def run(self, v,frame):
        v = self.mesh_smoother.process(v)
        self.mesh.triangles = o3d.utility.Vector3iVector(self.faces)
        self.mesh.vertices = o3d.utility.Vector3dVector(np.matmul(self.view_mat, v.T).T)
        self.mesh.paint_uniform_color(self.mesh_COLOR)
        self.mesh.compute_triangle_normals()
        self.mesh.compute_vertex_normals()
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

def run_client(host, port):
    capture = OpenCVCapture()
    visualizer = Open3d_visualizer()
    client = Client((host, port))
    data_cacher = np.zeros((6890,3))

    while True:
        frame = capture.read()
        if frame is not None:
            data = frame
        else:
            data = ['waiting for the cam']
        data_string = pickle.dumps(data)
        client.send(data_string)

        if frame is not None:
            data_bytes = client.recv()
            data_recieve = pickle.loads(data_bytes)
            if not isinstance(data_recieve, list):
                break_flag = visualizer.run(data_recieve, frame[:,:,::-1])
                data_cacher = data_recieve
            else:
                break_flag = visualizer.run(data_cacher, frame[:,:,::-1])
            if break_flag:
                break

if __name__ == '__main__':
    server_host = '10.207.174.18'  # host = 'localhost'
    server_port = 10086  # if [Address already in use], use another port
    run_client(server_host, server_port)  # then, run this function only in client
    pass