'''
  Brought from https://github.com/zju3dv/EasyMocap/blob/master/easymocap/socket/o3d.py
'''

import sys, os
root_dir = os.path.join(os.path.dirname(__file__),'..')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
import open3d as o3d
from visualization.vis_utils_py36_o3d9 import Timer, CritRange, Config, \
                                                convert_verts_to_cam_space
from visualization.socket_utils import BaseSocket, log
import json
import numpy as np
from os.path import join
import copy
import importlib
from config import args
import constants
import torch
import random
from utils.temporal_optimization import OneEuroFilter
from visualization.create_meshes import create_body_mesh, create_body_model

def load_object(module_name, module_args):
    module_path = '.'.join(module_name.split('.')[:-1])
    module = importlib.import_module(module_path)
    name = module_name.split('.')[-1]
    obj = getattr(module, name)(**module_args)
    return obj

def merge_params(params):
    params_batch = {}
    for key in params[0]:
        if isinstance(params[0][key], np.ndarray):
            if len(params[0][key].shape)>0:
                params_batch[key] = torch.from_numpy(np.concatenate([param[key] for param in params], 0)).float()
    return params_batch

rotate = False
def o3d_callback_rotate(vis=None):
    global rotate
    rotate = not rotate
    return False

class VisOpen3DSocket(BaseSocket):
    def __init__(self, host, port, cfg) -> None:
        # output
        self.write = cfg.write
        self.out = cfg.out
        self.cfg = cfg
        if self.write:
            print('[Info] capture the screen to {}'.format(self.out))
            os.makedirs(self.out, exist_ok=True)
        # scene
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name='Visualizer', width=cfg.width, height=cfg.height)
        self.load_scene_objects(cfg)
        # create vis => update => super() init
        super().__init__(host, port, debug=cfg.debug)
        self.block = cfg.block

        self.body_model = create_body_model()
        self.max_human = cfg.max_human
        self.track = cfg.track
        self.filter = cfg.filter
        self.camera_pose = cfg.camera.camera_pose
        self.init_camera(cfg.camera.camera_pose)
        self.zero_vertices = o3d.utility.Vector3dVector(np.zeros((self.body_model.get_num_verts(), 3)))

        self.filter_dict, self.meshes = {}, []
        self.verts_available_ids, self.current_mesh_num = [], 0
        self.pid_verts_dict = {}
        for i in range(self.max_human):
            self.add_human()
        self.verts_available_ids = sorted(self.verts_available_ids,reverse = True)
        self.set_meshes_zero()
        self.verts_change_cacher = []
        
        self.count = 0
        self.previous = {}
        self.critrange = CritRange(**cfg.range)
        self.new_frames  = cfg.new_frames

    def load_scene_objects(self, cfg):
        # load the scene
        for key, mesh_args in cfg.scene.items():
            mesh = load_object(key, mesh_args)
            self.vis.add_geometry(mesh)
    
    def set_meshes_zero(self):
        for mesh in self.meshes:
            mesh.vertices = self.zero_vertices

    def add_human(self):
        self.verts_available_ids.append(self.current_mesh_num)
        self.current_mesh_num+=1
        mesh = create_body_mesh()

        self.meshes.append(mesh)
        self.vis.add_geometry(mesh)
        self.init_camera(self.camera_pose)
        self.meshes[-1].vertices = self.zero_vertices

    @staticmethod
    def set_camera(cfg, camera_pose):
        theta, phi = np.deg2rad(-(cfg.camera.theta + 90)), np.deg2rad(cfg.camera.phi)
        theta = theta + np.pi
        st, ct = np.sin(theta), np.cos(theta)
        sp, cp = np.sin(phi), np.cos(phi)
        rot_x = np.array([
            [1., 0., 0.],
            [0., ct, -st],
            [0, st, ct]
        ])
        rot_z = np.array([
            [cp, -sp, 0],
            [sp, cp, 0.],
            [0., 0., 1.]
        ])
        camera_pose[:3, :3] = rot_x @ rot_z
        return camera_pose

    def init_camera(self, camera_pose):
        ctr = self.vis.get_view_control()
        init_param = ctr.convert_to_pinhole_camera_parameters()
        # init_param.intrinsic.set_intrinsics(init_param.intrinsic.width, init_param.intrinsic.height, fx, fy, cx, cy)
        init_param.extrinsic = np.array(camera_pose)
        ctr.convert_from_pinhole_camera_parameters(init_param) 

    def get_camera(self):
        ctr = self.vis.get_view_control()
        init_param = ctr.convert_to_pinhole_camera_parameters()
        return np.array(init_param.extrinsic)

    def main(self, datas):
        if self.debug:log('[Info] Load data {}'.format(self.count))
        if isinstance(datas, str):
            datas = json.loads(datas)
        for data in datas:
            for key in data.keys():
                if key == 'id':
                    continue
                data[key] = np.array(data[key])

        with Timer('forward'):
            params = []
            pid_showup = [data['id'] for data in datas]
            verts_id_used = []
            vertices_dict = {}
            print('recieving pid_showup: {}'.format(pid_showup))
            print('verts_change_cacher: ',self.verts_change_cacher)
            for i, data in enumerate(datas):
                pid = data['id']
                if pid not in self.pid_verts_dict:
                    if len(self.verts_available_ids)==0:
                        self.add_human()
                    self.pid_verts_dict[pid] = self.verts_available_ids.pop()
                if pid not in self.filter_dict:
                    self.filter_dict[pid] = OneEuroFilter(3.0,0.0)

                params.append(data)
            if len(params) > 0:
                params = merge_params(params)
                vertices = self.body_model(**params)['verts'].cpu().numpy()
                for ind, pid in enumerate(pid_showup):
                    verts_processed = convert_verts_to_cam_space(vertices[ind])
                    verts_processed = self.filter_dict[pid].process(verts_processed)
                    vertices_dict[self.pid_verts_dict[pid]] = o3d.utility.Vector3dVector(verts_processed)
                    verts_id_used.append(self.pid_verts_dict[pid])
            
            pid_used_all = np.unique(np.array(verts_id_used+self.verts_change_cacher))
            for ind in pid_used_all:
                if ind not in verts_id_used:
                    print('set mesh {} to zeros'.format(ind))
                    self.meshes[ind].vertices = self.zero_vertices
                    self.verts_available_ids.append(ind)
        
        print('pid_used_all', pid_used_all)
        # Open3D will lock the thread here
        with Timer('set vertices'):
            for mesh_id, verts in vertices_dict.items():
                self.meshes[mesh_id].vertices = verts

        self.verts_change_cacher = verts_id_used

    def o3dcallback(self):
        if rotate:
            self.cfg.camera.phi += np.pi/10
            camera_pose = self.set_camera(self.cfg, self.get_camera())
            self.init_camera(camera_pose)

    def update(self):
        if self.disconnect and not self.block:
            self.previous.clear()
        if not self.queue.empty():
            if self.debug:log('Update' + str(self.queue.qsize()))
            datas = self.queue.get()
            if not self.block:
                while self.queue.qsize() > 0:
                    datas = self.queue.get()
            self.main(datas)
            with Timer('update geometry'):
                for mesh in self.meshes:
                    mesh.compute_triangle_normals()
                    self.vis.update_geometry(mesh)
                self.o3dcallback()
                self.vis.poll_events()
                self.vis.update_renderer()
            if self.write:
                outname = join(self.out, '{:06d}.jpg'.format(self.count))
                with Timer('capture'):
                    self.vis.capture_screen_image(outname)
            self.count += 1
        else:
            with Timer('update renderer', True):
                self.o3dcallback()
                self.vis.poll_events()
                self.vis.update_renderer()

if __name__ == '__main__':
    cfg = Config().load('romp/lib/visualization/vis_cfgs/o3d_scene_smpl.yml')
    server = VisOpen3DSocket(cfg.host, cfg.port, cfg)
    while True:
        server.update()


'''
                if i >= len(self.meshes):
                    print('[Error] the number of human exceeds!')
                    self.add_human(data)


            #if 'keypoints3d' not in data.keys() and self.filter:
            #    outputs = self.body_model(**data)
            #    data['keypoints3d'] = outputs['j3d'].cpu().numpy()[0]
        #if self.filter:
        #   datas = self.filter_human(datas)
    def filter_human(self, datas):
        datas_new = []
        for data in datas:
            kpts3d = np.array(data['keypoints3d'])
            data['keypoints3d'] = kpts3d
            pid = data['id']
            if pid not in self.previous.keys():
                if not self.critrange(kpts3d):
                    continue
                self.previous[pid] = 0
            self.previous[pid] += 1
            if self.previous[pid] > self.new_frames:
                datas_new.append(data)
        return datas_new

'''