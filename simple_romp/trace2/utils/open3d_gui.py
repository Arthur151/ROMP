# This script is modified from https://github.com/isl-org/Open3D/blob/master/examples/python/gui/vis-gui.py
# This script also refers to https://github.com/mkocabas/body-model-visualizer
# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
import os
import cv2
import glob
import torch
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import scipy.spatial.transform.rotation as R
import open3d.visualization.rendering as rendering
import time
import threading

default_settings = {
    'image_window_width':20, # 20 times of ems wide
    'image_bg': True,
    'show_camera': 'camera_motion', #'camera_view',#
}

def prepare_o3d_mesh(mesh, face=None, color=[0.5, 0.75, 1.0]):
    if isinstance(mesh, str):
        mesh = o3d.io.read_triangle_mesh(mesh)
    elif isinstance(mesh, np.ndarray) or isinstance(mesh, torch.Tensor):
        mesh = o3d.utility.Vector3dVector(mesh)
        mesh = o3d.geometry.TriangleMesh(mesh, face)
    
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return mesh

def prepare_o3d_traj(traj_seqs, frame_ind, color=[0.5, 0.75, 1.0]):
    o3d.geometry.OrientedBoundingBox.create_from_points
    points = []
    line_indices = []
    colors = []
    frame_inds = list(traj_seqs.keys())
    for idx, ind in enumerate(frame_inds):
        if idx > frame_ind:
            break
        points.append(traj_seqs[ind])
        if idx != 0:
            line_indices.append([idx-1, idx])
            colors.append(color)
    traj = o3d.geometry.LineSet()
    traj.points = o3d.utility.Vector3dVector(points)
    traj.lines = o3d.utility.Vector2iVector(line_indices)
    traj.colors = o3d.utility.Vector3dVector(colors)
    return traj

def prepare_mesh_vertices(vertices):
    if isinstance(vertices, str):
        vertices = o3d.io.read_triangle_mesh(vertices).vertices
    elif isinstance(vertices, np.ndarray) or isinstance(vertices, torch.Tensor):
        vertices = o3d.utility.Vector3dVector(vertices)
    return vertices

def get_mesh_name(mesh_id):
    return f"mesh_{mesh_id}"
def get_traj_name(mesh_id):
    return f"traj_{mesh_id}"
def get_smpl_face():
    faces = np.load(os.path.join(os.path.dirname(__file__), 'data', 'smpl_faces.npy'))
    return o3d.utility.Vector3iVector(faces)

class Video3DVisualizer:
    def __init__(self, title, smpl_faces=None, settings=default_settings):
        self.settings = settings

        self.window = gui.Application.instance.create_window(title, 800, 600)
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_close)
        self.smpl_face =  o3d.utility.Vector3iVector(smpl_faces)

        self.setup_widgets() # add widgets to the window
        self.setup_3Dscene()

        self.is_done = False
        self.start_running()
    
    def start_running(self):
        threading.Thread(target=self._update_thread).start()
    
    def setup_widgets(self):
        # add 3D scene to visualize 3D objects
        self.widget3d = gui.SceneWidget()
        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.widget3d)

        # add 2D panel to visualize 2D images
        em = self.window.theme.font_size
        margin = 0.5 * em
        
        self.panel = gui.Vert(0.5 * em, gui.Margins(margin))
        self.image_label = gui.Label("Input image")
        self.panel.add_child(self.image_label)
        self.rgb_widget = gui.ImageWidget()
        self.panel.add_child(self.rgb_widget)

        spacing = int(np.round(em))
        # add a grid to stack the slider and button vertically
        # (the number of columns, the spacing between items (both vertically and horizontally), the margins)
        # gui.Margins(left, top, right, bottom)
        self.fixed_grid = gui.VGrid(1, spacing, gui.Margins(em, em, em, em))

        # add slider to control the frame index
        self.frame_slider_label = gui.Label("Frame index")
        self.fixed_grid.add_child(self.frame_slider_label)
        self.frame_slider = gui.Slider(gui.Slider.INT)
        self.frame_slider.int_value = 0
        self.frame_slider.set_on_value_changed(self._on_slider_changed)
        self.fixed_grid.add_child(self.frame_slider)
        self.panel.add_child(self.fixed_grid)

        # add button to control the frame index of the video (forward/backward) one by one
        self.fixed_grid2 = gui.Horiz(2, gui.Margins(3*em, 0, 3*em, 0))
        self.backward_button = gui.Button("-")
        self.backward_button.set_on_clicked(self._on_backward_frame_button_clicked)
        self.backward_button.horizontal_padding_em = 3
        self.fixed_grid2.add_child(self.backward_button)
        self.forward_button = gui.Button("+")
        self.forward_button.set_on_clicked(self._on_forward_frame_button_clicked)
        self.forward_button.horizontal_padding_em = 3
        self.fixed_grid2.add_child(self.forward_button)
        self.panel.add_child(self.fixed_grid2)

        # add button to control the play/pause of the video
        left_margin = 5.8 * em
        self.fixed_grid3 = gui.VGrid(1, spacing, gui.Margins(left_margin, 0, 0, 0))
        self.stop_button = gui.Button("Stop")
        self.stop_button.set_on_clicked(self._on_stop_button_clicked)
        self.stop_button.horizontal_padding_em = 3
        self.fixed_grid3.add_child(self.stop_button)
        self.panel.add_child(self.fixed_grid3)
        
        self.window.add_child(self.panel)
    
    def setup_3Dscene(self):
        self.material = rendering.MaterialRecord()
        self.material.shader = "defaultLit"
        self.lineset_material = rendering.MaterialRecord()
        self.lineset_material.base_color = [0.5, 0.75, 1.0, 1.0]
        self.lineset_material.line_width = 5.0
        self.lineset_material.shader = "unlitLine"
        self.traj_material = rendering.MaterialRecord()
        self.traj_material.line_width = 6.0
        self.traj_material.shader = "unlitLine"

        self.mesh_id = 0
        self.mesh_seqs = {}
        self.traj_seqs = {}

        self.camera_intrinsics = []
        self.camera_extrinsics = []

        self.widget3d.scene.show_axes(True)
        self.widget3d.scene.show_skybox(True)
        # add a ground plane, XY, XZ, YZ axis
        ground_plane = o3d.visualization.rendering.Scene.GroundPlane(0)
        self.widget3d.scene.show_ground_plane(True, ground_plane)

    def add_images(self, images, image_names=None):
        self.rgb_images = []
        self.rgb_image_names = []
        for ind, image in enumerate(images):
            if isinstance(image, str):
                if image_names is None:
                    self.rgb_image_names.append(os.path.basename(image))
                image = np.array(cv2.imread(image)[:,:,::-1])
            img = o3d.geometry.Image(image)
            self.rgb_images.append(img)

            if image_names is not None:
                self.rgb_image_names.append(image_names[ind])
        self.frame_slider.set_limits(0, self._seq_length_()-1)
        print('Total number of frames: {} loaded'.format(self._seq_length_()))
        
    def add_meshes(self, meshes, trans, frame_inds=None):
        frame_inds = np.arange(len(meshes)) if frame_inds is None else frame_inds
        print(meshes.shape, trans.shape)
        self.mesh_seqs[self.mesh_id] = {frame_ind:mesh+tran[None] for frame_ind, mesh, tran in zip(frame_inds, meshes, trans)}
        self.traj_seqs[self.mesh_id] = {frame_ind:tran for frame_ind, tran in zip(frame_inds, trans)}
        self.mesh_id += 1
    
    def add_camera_poses(self, camera_intrinsics, camera_extrinsics):
        # in open3D, camera coordinate is defined as x right, y down, z forward(pointing into image plane), left figure below
        # in pyrender, camera coordinate is defined as x right, y up, z backward(pointing from image plane), right figure below
        #     ^                    ^
        #    /                     |
        #   z Open3D               y  Pyrender        camera coordinates
        # /                        |
        # --- x --->                --- x ---> 
        # |                       /  
        # y                      z
        # |                     /
        # v                    v
        self.camera_intrinsics = camera_intrinsics
        self.camera_extrinsics = camera_extrinsics
        
        if self.settings['show_camera'] == 'camera_motion':
            #look_at(center, eye, up): sets the camera view so that the camera is located at ‘eye’, 
            # pointing towards ‘center’, and oriented so that the up vector is ‘up’
            scene_camera_lookat = [0, 0, 0]
            scene_camera_position = [0, 2, -5] # [5, 2, 0] # [0, 2, -5]
            scene_camera_up = [0, -1, 0]
            self.widget3d.look_at(scene_camera_lookat, scene_camera_position, scene_camera_up)
    
    def set_camera_pose(self, intrinsic, extrinsic, width_px=512, height_px=512):
        # setup_camera(intrinsic_matrix, extrinsic_matrix, intrinsic_width_px, intrinsic_height_px, model_bounds)
        bounds = self.widget3d.scene.bounding_box
        self.widget3d.setup_camera(intrinsic, extrinsic, width_px, height_px, bounds)
    
    def show_camera_motion(self, intrinsic_matrix, extrinsic_matrix, width=512, height=512, camera_name="camera_mesh"):
        if self.widget3d.scene.has_geometry(camera_name):
            self.widget3d.scene.remove_geometry(camera_name)
        camera_mesh = o3d.geometry.LineSet.create_camera_visualization(width, height, intrinsic_matrix, extrinsic_matrix, scale=1)
        self.widget3d.scene.add_geometry(camera_name, camera_mesh, self.lineset_material)
    
    def set_camera2see_all(self):
        bounds = self.widget3d.scene.bounding_box
        self.widget3d.setup_camera(60.0, bounds, bounds.get_center())

    def _on_layout(self, layout_context):
        contentRect = self.window.content_rect
        panel_width = self.settings['image_window_width'] * layout_context.theme.font_size  
        # define the position of the widgets:
        # 3D widget is on the left; while panel, slider, button are stacked on the right side.
        # gui.Rect(x, y, width, height): x, y are the position of the left-top corner; width, height are the size of the widget.
        self.widget3d.frame = gui.Rect(contentRect.x, contentRect.y,
                                       contentRect.width - panel_width, contentRect.height)
        self.panel.frame = gui.Rect(self.widget3d.frame.get_right(), contentRect.y, 
                                    panel_width, contentRect.height)
    
    def update_func(self, go_next=True):
        frame_ind = self.frame_slider.int_value
        # Get the next frame, for instance, reading a frame from the camera.
        rgb_frame = self.rgb_images[frame_ind]

        # Update the images. This must be done on the UI thread.
        def update():
            self.rgb_widget.update_image(rgb_frame)
            if len(self.rgb_image_names)>frame_ind:
                self.image_label.text = f'Img: {self.rgb_image_names[frame_ind]}'
            if self.settings['image_bg']:
                self.widget3d.scene.set_background([1, 1, 1, 1], rgb_frame)
            # update mesh
            #self.widget3d.scene.clear_geometry() # clear all meshes
            for mesh_id in range(self.mesh_id):
                if self.widget3d.scene.has_geometry(get_mesh_name(mesh_id)):
                    self.widget3d.scene.remove_geometry(get_mesh_name(mesh_id))
                if self.widget3d.scene.has_geometry(get_traj_name(mesh_id)):
                    self.widget3d.scene.remove_geometry(get_traj_name(mesh_id))
                if frame_ind in self.mesh_seqs[mesh_id]:
                    #print(f'update mesh {mesh_id}', frame_ind)
                    new_mesh = prepare_o3d_mesh(self.mesh_seqs[mesh_id][frame_ind], face=self.smpl_face)
                    self.widget3d.scene.add_geometry(get_mesh_name(mesh_id), new_mesh, self.material) 
                if frame_ind in self.traj_seqs[mesh_id] and frame_ind>min(list(self.traj_seqs[mesh_id].keys())):
                    new_traj = prepare_o3d_traj(self.traj_seqs[mesh_id], frame_ind)
                    self.widget3d.scene.add_geometry(get_traj_name(mesh_id), new_traj, self.traj_material)
            
            if len(self.camera_intrinsics)>frame_ind:
                if self.settings['show_camera'] == 'camera_view':
                    self.set_camera_pose(self.camera_intrinsics[frame_ind], self.camera_extrinsics[frame_ind])
                elif self.settings['show_camera'] == 'camera_motion':
                    self.show_camera_motion(self.camera_intrinsics[frame_ind], self.camera_extrinsics[frame_ind])
            #self.set_camera2see_all()
        
        if go_next:
            self.frame_slider.int_value += 1
            if self.frame_slider.int_value >= self._seq_length_()-1:
                self.frame_slider.int_value = 0
        return update

    def _seq_length_(self):
        return len(self.rgb_images)

    def _update_thread(self):
        # This is NOT the UI thread, need to call post_to_main_thread() to update
        # the scene or any part of the UI.
        while not self.is_done:
            time.sleep(0.030)
            update = self.update_func(go_next=True)
            if not self.is_done:
                gui.Application.instance.post_to_main_thread(
                    self.window, update)
    
    def test(self, test_image_dir):
        image_paths = sorted(glob.glob(os.path.join(test_image_dir, '*')))
        self.add_images(image_paths)
        tet = o3d.geometry.TriangleMesh.create_tetrahedron()
        self.add_meshes([tet])
        self.set_camera2see_all()
    
    def _on_slider_changed(self, value):
        if value >= self._seq_length_():
            value = self._seq_length_()-1
        #print('Slider changed to', value)
        self.frame_slider.int_value = int(value)

        update = self.update_func(go_next=False)
        gui.Application.instance.post_to_main_thread(
                    self.window, update)
    
    def _on_stop_button_clicked(self):
        if self.stop_button.text == 'Stop':
            self.is_done = True
            self.stop_button.text = 'Play'
        else:
            self.is_done = False
            self.stop_button.text = 'Stop'
            self.start_running()
    
    def _on_forward_frame_button_clicked(self):
        if self.stop_button.text == 'Stop':
            self.is_done = True
            self.stop_button.text = 'Play'

        self.frame_slider.int_value += 1
        if self.frame_slider.int_value >= self._seq_length_()-1:
            self.frame_slider.int_value = 0
        update = self.update_func(go_next=False)
        gui.Application.instance.post_to_main_thread(
                    self.window, update)

    def _on_backward_frame_button_clicked(self):
        if self.stop_button.text == 'Stop':
            self.is_done = True
            self.stop_button.text = 'Play'
            
        self.frame_slider.int_value -= 1
        if self.frame_slider.int_value < 0:
            self.frame_slider.int_value = self._seq_length_()-1
        update = self.update_func(go_next=False)
        gui.Application.instance.post_to_main_thread(
                    self.window, update)

    def _on_close(self):
        self.is_done = True
        return True  # False would cancel the close

def visualize_world_annots(title, verts, world_trans, camera_intrinsics, camera_extrinsics, img_paths, smpl_face):
    app = o3d.visualization.gui.Application.instance
    app.initialize()

    win = Video3DVisualizer(title, smpl_faces=smpl_face)
    win.add_images(img_paths)
    for vert, tran in zip(verts, world_trans):
        win.add_meshes(vert, tran)
    win.add_camera_poses(camera_intrinsics, camera_extrinsics)

    app.run()

def main():
    test_image_dir = '/Users/mac/Desktop/final_clip_frames/residence9_jumping-0'

    app = o3d.visualization.gui.Application.instance
    app.initialize()

    win = Video3DVisualizer()
    win.test(test_image_dir)

    app.run()


if __name__ == "__main__":
    main()
