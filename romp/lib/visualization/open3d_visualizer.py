import open3d as o3d
import numpy as np

import config
import constants
from config import args
from utils.temporal_optimization import OneEuroFilter
from visualization.create_meshes import create_body_mesh, create_body_model


def convert_trans_scale(trans):
    trans *= np.array([0.4, 0.6, 0.7])
    return trans

class Open3d_visualizer(object):
    def __init__(self, multi_mode=False):
        self.window_size = np.array([1280,1080])
        #self.window_size = np.array([720,720])
        self._init_viewer_()
        if not multi_mode:
            self.prepare_single_person_scene()
        else:
            self.prepare_multi_person_scene()

    def _init_viewer_(self):
        self.viewer = o3d.visualization.Visualizer()
        self.viewer.create_window(width=self.window_size[0], height=self.window_size[1], window_name='ROMP - output')

    def _set_view_configs_(self, cam_location, focal_length=1000):
        view_control = self.viewer.get_view_control()
        cam_params = view_control.convert_to_pinhole_camera_parameters()

        intrinsics = cam_params.intrinsic.intrinsic_matrix.copy()
        focal_length = max(self.window_size)/2. / np.tan(np.radians(args().FOV/2.))
        intrinsics[0,0], intrinsics[1,1] = focal_length, focal_length
        cam_params.intrinsic.intrinsic_matrix = intrinsics
        #print('Open3d_visualizer Camera intrinsic matrix: ', intrinsics)

        extrinsics = np.eye(4)
        extrinsics[0:3, 3] = cam_location
        #print('Open3d_visualizer Camera extrinsic matrix: ', extrinsics)
        cam_params.extrinsic = extrinsics

        view_control.convert_from_pinhole_camera_parameters(cam_params)
        view_control.set_constant_z_far(100)

        render_option = self.viewer.get_render_option()
        render_option.load_from_json('romp/lib/visualization/vis_cfgs/render_option.json')
        self.viewer.update_renderer()

    def update_viewer(self):
        self.viewer.poll_events()
        self.viewer.update_renderer()

    def prepare_single_person_scene(self):
        self.mesh = create_body_mesh()
        self.viewer.add_geometry(self.mesh)
        cam_location = np.array([0,0,0])
        if not args().add_trans:
            cam_location[2] = 2.6/np.tan(np.radians(args().FOV/2.))
        self._set_view_configs_(cam_location=cam_location)
        self.smoother = self.create_filter()
        self.update_viewer()

    def create_filter(self):
        return {'verts':OneEuroFilter(4.,0), 'trans':OneEuroFilter(3.,0.)}

    def process_single_mesh(self, verts, trans, smoother, mesh_ob):
        verts = smoother['verts'].process(verts)
        #verts = np.matmul(self.view_mat, verts.T).T
        if trans is not None:
            trans_converted = convert_trans_scale(trans)
            verts += smoother['trans'].process(trans_converted)[None]
        
        mesh_ob.vertices = o3d.utility.Vector3dVector(verts)
        mesh_ob.compute_triangle_normals()
        mesh_ob.compute_vertex_normals()
        self.viewer.update_geometry(mesh_ob)

    def run(self, verts, trans=None):
        self.process_single_mesh(verts, trans, self.smoother, self.mesh)
        self.update_viewer()

    def prepare_multi_person_scene(self, start_person_num=6):
        self.pid2mid_dict = {}
        self.mesh_usage_change_cacher = []
        self.mesh_num = start_person_num
        self.mesh_ids_available = list(range(start_person_num))
        self.meshes = {mid:create_body_mesh() for mid in self.mesh_ids_available}
        self.zero_mesh = o3d.utility.Vector3dVector(np.zeros((len(self.meshes[0].vertices),3)))
        self.filter_dict = {}
        for mid in self.mesh_ids_available:
            self.viewer.add_geometry(self.meshes[mid])
            self.update_viewer()
        for mid in self.mesh_ids_available:
            self.reset_mesh(mid)
            self.update_viewer()
        cam_location = np.array([0,0,0])
        self._set_view_configs_(cam_location=cam_location)
        
    def add_mesh(self):
        print('Adding new Mesh {}'.format(self.mesh_num))
        self.mesh_ids_available.append(self.mesh_num)
        new_mesh = create_body_mesh()
        self.filter_dict[self.mesh_num] = self.create_filter()
        new_mesh.vertices=self.zero_mesh
        new_mesh.compute_triangle_normals()
        new_mesh.compute_vertex_normals()
        self.meshes[self.mesh_num] = new_mesh
        self.viewer.add_geometry(self.meshes[self.mesh_num])
        #self.update_viewer()
        self.mesh_num += 1

    def reset_mesh(self, mesh_id):
        print('Reseting Mesh {}'.format(mesh_id))
        self.meshes[mesh_id].vertices = self.zero_mesh
        self.meshes[mesh_id].compute_triangle_normals()
        self.meshes[mesh_id].compute_vertex_normals()
        self.filter_dict[mesh_id] = self.create_filter()
        if mesh_id not in self.mesh_ids_available:
            self.mesh_ids_available.append(mesh_id)
        self.viewer.update_geometry(self.meshes[mesh_id])
        #self.update_viewer()

    def run_multiperson(self, verts, trans=None, tracked_ids=None):
        #print('recieving {} people'.format(len(verts)))
        assert len(verts)==len(trans)==len(tracked_ids), print('length is not equal~')
        for vert, tran, pid in zip(verts, trans, tracked_ids):
            if pid not in self.pid2mid_dict:
                if len(self.mesh_ids_available)==0:
                    self.add_mesh()
                self.pid2mid_dict[pid] = self.mesh_ids_available.pop()
            mesh_id = self.pid2mid_dict[pid]
            self.process_single_mesh(vert, tran, self.filter_dict[mesh_id], self.meshes[mesh_id])

        # reset the disappeared people
        for pid in self.mesh_usage_change_cacher:
            if pid in self.pid2mid_dict and pid not in tracked_ids:
                self.reset_mesh(self.pid2mid_dict[pid])
                self.pid2mid_dict.pop(pid, None)
        self.mesh_usage_change_cacher = tracked_ids
        self.update_viewer()