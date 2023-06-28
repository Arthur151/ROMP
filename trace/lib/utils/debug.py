import cv2
import torch
import numpy as np
from vedo import *
import logging

def show_keypoints(kp3ds_list, bones_list, colors_list, invalid_value=-2., point_radius=6):
    show_items = []

    for kp3ds, bones, colors in zip(kp3ds_list, bones_list, colors_list):
        points3D = []
        for point in kp3ds:
            if invalid_value in point:
                continue
            points3D.append(point)
        for p1_ind, p2_ind in bones:
            if invalid_value in kp3ds[p1_ind] or invalid_value in kp3ds[p2_ind]:
                continue
            show_items.append(Line(kp3ds[p1_ind], p1=kp3ds[p2_ind], closed=False, res=2, lw=3, c='k1', alpha=1))
        cpts = Points(points3D, c=colors, r=point_radius)
        show_items.append(cpts)
    
    show(*show_items)#.close()


def show_verts(verts_list, face_list, colors_list):
    
    face = face_list[0]
    for verts1, verts2 in zip(verts_list[0], verts_list[1]):
        show_items = []
        #logging.info(verts.shape)
        mesh1 = Mesh([verts1, face]) 
        mesh2 = Mesh([verts2, face]) 
        show_items.append(mesh1)
        show_items.append(mesh2)
    
        show(*show_items)#.close()

class Plotter3dPoses:

    def __init__(self, canvas_size=(512,512), origin=(0.5, 0.5), scale=200):
        self.canvas_size = canvas_size
        self.origin = np.array([origin[1] * canvas_size[1], origin[0] * canvas_size[0]], dtype=np.float32)  # x, y
        self.scale = np.float32(scale)
        self.theta, self.phi = 0, np.pi/2 #np.pi/4, -np.pi/6
        axis_length = 200
        axes = [
            np.array([[-axis_length/2, -axis_length/2, 0], [axis_length/2, -axis_length/2, 0]], dtype=np.float32),
            np.array([[-axis_length/2, -axis_length/2, 0], [-axis_length/2, axis_length/2, 0]], dtype=np.float32),
            np.array([[-axis_length/2, -axis_length/2, 0], [-axis_length/2, -axis_length/2, axis_length]], dtype=np.float32)]
        step = 20
        for step_id in range(axis_length // step + 1):  # add grid
            axes.append(np.array([[-axis_length / 2, -axis_length / 2 + step_id * step, 0],
                                  [axis_length / 2, -axis_length / 2 + step_id * step, 0]], dtype=np.float32))
            axes.append(np.array([[-axis_length / 2 + step_id * step, -axis_length / 2, 0],
                                  [-axis_length / 2 + step_id * step, axis_length / 2, 0]], dtype=np.float32))
        self.axes = np.array(axes)

    def plot(self, pose_3ds, bones=None, colors=[(255, 0, 0)], img=None):
        img = np.ones((self.canvas_size[0],self.canvas_size[1],3), dtype=np.uint8) * 0 if img is None else img
        R = self._get_rotation(self.theta, self.phi)
        #self._draw_axes(img, R)
        for vertices, color in zip(pose_3ds,colors):
            self._plot_edges(img, vertices, bones, R, color)
        return img

    def encircle_plot(self, pose_3ds, bones, colors=[(255, 255, 255)], img=None):
        img = np.ones((self.canvas_size[0],self.canvas_size[1],3), dtype=np.uint8) * 255 if img is None else img
        #encircle_theta, encircle_phi = [0, np.pi/4, np.pi/2, 3*np.pi/4], [np.pi/2,np.pi/2,np.pi/2,np.pi/2]
        encircle_theta, encircle_phi = [0,0,0, np.pi/4,np.pi/4,np.pi/4, np.pi/2,np.pi/2,np.pi/2], [np.pi/2, 5*np.pi/7, -2*np.pi/7, np.pi/2, 5*np.pi/7, -2*np.pi/7, np.pi/2, 5*np.pi/7, -2*np.pi/7,]
        encircle_origin = np.array([[0.165, 0.165], [0.165, 0.495], [0.165, 0.825],\
                                    [0.495, 0.165], [0.495, 0.495], [0.495, 0.825],\
                                    [0.825, 0.165], [0.825, 0.495], [0.825, 0.825]], dtype=np.float32) * np.array(self.canvas_size)[None]
        for self.theta, self.phi, self.origin in zip(encircle_theta, encircle_phi, encircle_origin):
            R = self._get_rotation(self.theta, self.phi)
            #self._draw_axes(img, R)
            for vertices, color in zip(pose_3ds,colors):
                self._plot_edges(img, vertices*0.6, bones, R, color)
        return img

    def _draw_axes(self, img, R):
        axes_2d = np.dot(self.axes, R)
        axes_2d = axes_2d + self.origin
        for axe in axes_2d:
            axe = axe.astype(int)
            cv2.line(img, tuple(axe[0]), tuple(axe[1]), (128, 128, 128), 1, cv2.LINE_AA)

    def _plot_edges(self, img, vertices, edges, R, color):
        vertices_2d = np.dot(vertices, R)
        edges_vertices = vertices_2d.reshape((-1, 2))[edges] * self.scale + self.origin
        org_verts = vertices.reshape((-1, 3))[edges]
        for inds, edge_vertices in enumerate(edges_vertices):
            if 0 in org_verts[inds]:
                continue
            edge_vertices = edge_vertices.astype(int)
            cv2.line(img, tuple(edge_vertices[0]), tuple(edge_vertices[1]), color, 10, cv2.LINE_AA)

    def _get_rotation(self, theta, phi):
        sin, cos = math.sin, math.cos
        return np.array([
            [ cos(theta),  sin(theta) * sin(phi)],
            [-sin(theta),  cos(theta) * sin(phi)],
            [ 0,                       -cos(phi)]
        ], dtype=np.float32)  # transposed

# plot3d = Plotter3dPoses()
# pose3d_plot = plot3d.plot([kp3d_preds[pids[0]], kp3d_gts[gids[0]]], bones=lsp14_connMat, colors=[(255, 0, 0), (0,0,255)])
# cv2.imshow('3D skeleton', pose3d_plot)
# cv2.waitKey(10)