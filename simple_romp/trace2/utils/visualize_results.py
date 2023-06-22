import cv2
import os
import numpy as np
import torch
import time
import copy
import math
from vis_human import setup_renderer

color_table_default = np.array([
    [0.4, 0.6, 1], # blue
    [0.8, 0.7, 1], # pink
    [0.1, 0.9, 1], # cyan
    [0.8, 0.9, 1], # gray
    [1, 0.6, 0.4], # orange
    [1, 0.7, 0.8], # rose
    [1, 0.9, 0.1], # Yellow
    [1, 0.9, 0.8], # skin
    [0.9, 1, 1],   # light blue
    [0.9, 0.7, 0.4], # brown
    [0.8, 0.7, 1], # purple
    [0.8, 0.9, 1], # light blue 2
    [0.9, 0.3, 0.1], # red
    [0.7, 1, 0.6],   # green
    [0.7, 0.4, 0.6], # dark purple
    [0.3, 0.5, 1], # deep blue
])[:,::-1]

def mesh_color_left2right(trans, color_table=None):
    left2right_order = torch.sort(trans[:,0].cpu()).indices.numpy()
    color_inds = np.arange(len(trans))
    color_inds[left2right_order] = np.arange(len(trans))
    if color_table is None:
        color_table = color_table_default
    return np.array([color_table[ind % len(color_table)] for ind in color_inds])

tracking_color_list = np.array([
            0.000, 0.447, 0.741,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000]).astype(np.float32).reshape((-1,3))

def mesh_color_trackID(track_ids, color_table=None):
    if color_table is None:
        color_table = tracking_color_list
    return np.array([color_table[tid % len(color_table)] for tid in track_ids])

def get_rotate_x_mat(angle):
    angle = np.radians(angle)
    rot_mat = torch.Tensor([
        [1, 0, 0], 
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle), np.cos(angle)]])
    return rot_mat

def get_rotate_y_mat(angle):
    angle = np.radians(angle)
    rot_mat = torch.Tensor([
        [np.cos(angle), 0, np.sin(angle)], 
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]])
    return rot_mat

def rotate_view_weak_perspective(verts, rx=30, ry=0, img_shape=[512,512], expand_ratio=1.2, bbox3D_center=None, scale=None):
    device, dtype = verts.device, verts.dtype
    h, w = img_shape
    
    # front2birdview: rx=90, ry=0 ; front2sideview: rx=0, ry=90
    Rx_mat = get_rotate_x_mat(rx).type(dtype).to(device)
    Ry_mat = get_rotate_y_mat(ry).type(dtype).to(device)
    verts_rot = torch.einsum('bij,kj->bik', verts, Rx_mat)
    verts_rot = torch.einsum('bij,kj->bik', verts_rot, Ry_mat)
    
    if bbox3D_center is None:
        flatten_verts = verts_rot.view(-1, 3)
        # To move the vertices to the center of view, we get the bounding box of vertices and its center location 
        bbox3D_center = 0.5 * (flatten_verts.min(0).values + flatten_verts.max(0).values)[None, None]
    verts_aligned = verts_rot - bbox3D_center
    
    rendered_image_center = torch.Tensor([[[w / 2, h / 2]]]).to(device).type(verts_aligned.dtype)
    
    if scale is None:
        # To ensure all vertices are visible, we need to rescale the vertices
        scale = 1 / (expand_ratio * torch.abs(torch.div(verts_aligned[:,:,:2], rendered_image_center)).max()) 
    # move to the center of rendered image 
    verts_aligned *=  scale
    verts_aligned[:,:,:2] += rendered_image_center

    return verts_aligned, bbox3D_center, scale

def rotate_view_perspective(verts, rx=30, ry=0, FOV=60, bbox3D_center=None, depth=None):
    device, dtype = verts.device, verts.dtype

    # front2birdview: rx=90, ry=0 ; front2sideview: rx=0, ry=90
    Rx_mat = get_rotate_x_mat(rx).type(dtype).to(device)
    Ry_mat = get_rotate_y_mat(ry).type(dtype).to(device)
    verts_rot = torch.einsum('bij,kj->bik', verts, Rx_mat)
    verts_rot = torch.einsum('bij,kj->bik', verts_rot, Ry_mat)
    
    if bbox3D_center is None:
        flatten_verts = verts_rot.view(-1, 3)
        # To move the vertices to the center of view, we get the bounding box of vertices and its center location 
        bbox3D_center = 0.5 * (flatten_verts.min(0).values + flatten_verts.max(0).values)[None, None]
    verts_aligned = verts_rot - bbox3D_center
    
    if depth is None:
        # To ensure all vertices are visible, we need to move them further.
        # get the least / the greatest distance between the center of 3D bbox and all vertices
        dist_min = torch.abs(verts_aligned.view(-1, 3).min(0).values)
        dist_max = torch.abs(verts_aligned.view(-1, 3).max(0).values)
        z = dist_max[:2].max() / np.tan(np.radians(FOV/2)) + dist_min[2]
        depth = torch.tensor([[[0, 0, z]]], device=device)    
    verts_aligned = verts_aligned + depth

    return verts_aligned, bbox3D_center, depth

def rendering_mesh_rotating_view(vert_trans, renderer, triangles, image, background, internal=5):
    result_imgs = []
    pause_num = 24
    pause = np.zeros(pause_num).astype(np.int32)
    change_time = 90//internal
    roates = np.ones(change_time) * internal
    go_up = np.sin(np.arange(change_time).astype(np.float32)/change_time) * 1
    go_down = np.sin(np.arange(change_time).astype(np.float32)/change_time - 1) * 1
    azimuth_angles = np.concatenate([pause, roates, roates, roates, roates])
    elevation_angles = np.concatenate([pause, go_up, go_down, go_up, go_down])

    camera_pose = np.eye(4)
    elevation_start = 20
    camera_pose[:3,:3] = get_rotate_x_mat(-elevation_start)
    cam_height = 1.4*vert_trans[:,:,2].mean().item()*np.tan(np.radians(elevation_start))
    camera_pose[:3,3] = np.array([0,cam_height,0]) # translation

    verts_rotated = vert_trans.clone()
    bbox3D_center, move_depth = None, None
    for azimuth_angle, elevation_angle in zip(azimuth_angles, elevation_angles):
        verts_rotated, bbox3D_center, move_depth = rotate_view_perspective(verts_rotated, rx=0, ry=azimuth_angle, depth=move_depth)
        rendered_image, rend_depth = renderer(verts_rotated.cpu().numpy(), triangles, background, mesh_colors=np.array([[0.9, 0.9, 0.8]]), camera_pose=camera_pose)
        result_imgs.append(rendered_image)
    
    return result_imgs

smpl24_connMat = np.array([0,1, 0,2, 0,3, 1,4,4,7,7,10, 2,5,5,8,8,11, 3,6,6,9,9,12,12,15, 12,13,13,16,16,18,18,20,20,22, 12,14,14,17,17,19,19,21,21,23]).reshape(-1, 2)
def draw_skeleton(image, pts, bones=smpl24_connMat, cm=None, label_kp_order=False,r=8):
    for i,pt in enumerate(pts):
        if len(pt)>1:
            if pt[0]>0 and pt[1]>0:
                image = cv2.circle(image,(int(pt[0]), int(pt[1])),r,cm[i%len(cm)],-1)
                if label_kp_order and i in bones:
                    img=cv2.putText(image,str(i),(int(pt[0]), int(pt[1])),cv2.FONT_HERSHEY_COMPLEX,1,(255,215,0),1)
    if bones is not None:
        set_colors = np.array([cm for i in range(len(bones))]).astype(np.int32)
        bones = np.concatenate([bones,set_colors],1).tolist()
        for line in bones:
            pa = pts[line[0]]
            pb = pts[line[1]]
            if (pa>0).all() and (pb>0).all():
                xa,ya,xb,yb = int(pa[0]),int(pa[1]),int(pb[0]),int(pb[1])
                image = cv2.line(image,(xa,ya),(xb,yb),(int(line[2]), int(line[3]), int(line[4])), r)
    return image

def draw_skeleton_multiperson(image, pts_group, colors):
    for ind, pts in enumerate(pts_group):
        image = draw_skeleton(image, pts, cm=colors[ind])
    return image

def process_idx(reorganize_idx, vids=None):
    reorganize_idx = reorganize_idx.cpu().numpy()
    used_org_inds = np.unique(reorganize_idx)
    per_img_inds = [np.where(reorganize_idx==org_idx)[0] for org_idx in used_org_inds]

    return used_org_inds, per_img_inds

def visulize_result(renderer, outputs, imgpath, rendering_cfgs, save_dir, smpl_model_path):
    used_org_inds, per_img_inds = process_idx(outputs['reorganize_idx'])
    render_images_path = []
    for org_ind, img_inds in zip(used_org_inds, per_img_inds):
        image_path = imgpath[org_ind]
        image = cv2.imread(image_path)
        if image.shape[1]>1024:
            cv2.resize(image, (image.shape[1]//2,image.shape[0]//2))
        render_image = render_image_results(renderer, outputs, img_inds, image, rendering_cfgs, smpl_model_path)
        save_path = os.path.join(save_dir, os.path.basename(image_path))
        cv2.imwrite(save_path, render_image)
        render_images_path.append(save_path)
    return render_images_path

def save_video(frame_save_paths, save_path, frame_rate=24):
    if len(frame_save_paths)== 0:
        return 
    height, width = cv2.imread(frame_save_paths[0]).shape[:2]
    writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width, height))
    for frame_path in frame_save_paths:
        writer.write(cv2.imread(frame_path))
    writer.release()

def render_image_results(renderer, outputs, img_inds, image, rendering_cfgs, smpl_model_path):
    triangles = torch.load(os.path.join(smpl_model_path))['f'].numpy().astype(np.int32)
    h, w = image.shape[:2]
    background = np.ones([h, h, 3], dtype=np.uint8) * 255
    result_image = [image]

    cam_trans = outputs['cam_trans'][img_inds]
    if rendering_cfgs['mesh_color'] == 'identity':
        if 'track_ids' in outputs:
            mesh_colors = mesh_color_trackID(outputs['track_ids'][img_inds])
        else:
            mesh_colors = mesh_color_left2right(cam_trans)
    elif rendering_cfgs['mesh_color'] == 'same':
        mesh_colors = np.array([[.9, .9, .8] for _ in range(len(cam_trans))])

    if rendering_cfgs['renderer'] == 'sim3dr':
        depth_order = torch.sort(cam_trans[:,2].cpu(),descending=True).indices.numpy()
        mesh_colors = mesh_colors[depth_order]
        verts_tran = (outputs['verts'][img_inds] + cam_trans.unsqueeze(1))[depth_order]
        verts_tran[:,:,2] = verts_tran[:,:,2]*-1
        if 'mesh' in rendering_cfgs['items']:
            vertices = outputs['verts_camed_org'][img_inds][depth_order].cpu().numpy()
            vertices[:,:,2] = vertices[:,:,2]*-1 
            rendered_image = renderer(vertices, triangles, image, mesh_colors=mesh_colors)
            #rendered_image = cv2.addWeighted(image, 1 - alpha, rendered_image, alpha, 0)
            result_image.append(rendered_image)
        
        if 'world_mesh' in rendering_cfgs['items']:
            vertices = outputs['world_verts_camed_org'][img_inds][depth_order].cpu().numpy()
            vertices[:,:,2] = vertices[:,:,2]*-1 
            rendered_image = renderer(vertices, triangles, image, mesh_colors=mesh_colors)
            #rendered_image = cv2.addWeighted(image, 1 - alpha, rendered_image, alpha, 0)
            result_image.append(rendered_image)

        if 'mesh_bird_view' in rendering_cfgs['items']:
            verts_bird_view, bbox3D_center, scale = rotate_view_weak_perspective(verts_tran, rx=-90, ry=0, img_shape=background.shape[:2], expand_ratio=1.2)
            rendered_bv_image = renderer(verts_bird_view.cpu().numpy(), triangles, background, mesh_colors=mesh_colors)
            result_image.append(rendered_bv_image)
        
        if 'mesh_side_view' in rendering_cfgs['items']:
            verts_side_view, bbox3D_center, scale = rotate_view_weak_perspective(verts_tran, rx=0, ry=-90, img_shape=image.shape[:2], expand_ratio=1.2)
            rendered_sv_image = renderer(verts_side_view.cpu().numpy(), triangles, background, mesh_colors=mesh_colors)
            result_image.append(rendered_sv_image)

    if rendering_cfgs['renderer'] == 'pyrender':
        verts_tran = outputs['verts'][img_inds] + cam_trans.unsqueeze(1)
        
        if 'mesh' in rendering_cfgs['items']:
            rendered_image, rend_depth = renderer(verts_tran.cpu().numpy(), triangles, image, mesh_colors=mesh_colors)
            #rendered_image = cv2.addWeighted(image, 1 - alpha, rendered_image, alpha, 0)
            result_image.append(rendered_image)
        
        if 'world_mesh' in rendering_cfgs['items']:
            verts_tran = outputs['world_verts'][img_inds] + outputs['world_trans'][img_inds].unsqueeze(1)
            rendered_image, rend_depth = renderer(verts_tran.cpu().numpy(), triangles, image, mesh_colors=mesh_colors)
            #rendered_image = cv2.addWeighted(image, 1 - alpha, rendered_image, alpha, 0)
            result_image.append(rendered_image)

        if 'mesh_bird_view' in rendering_cfgs['items']:
            verts_bird_view, bbox3D_center, move_depth = rotate_view_perspective(verts_tran, rx=90, ry=0)
            rendered_bv_image, rend_depth = renderer(verts_bird_view.cpu().numpy(), triangles, background, persp=False, mesh_colors=mesh_colors)
            result_image.append(cv2.resize(rendered_bv_image, (h, h)))

        if 'mesh_side_view' in rendering_cfgs['items']:
            verts_side_view, bbox3D_center, move_depth = rotate_view_perspective(verts_tran, rx=0, ry=90)
            rendered_sv_image, rend_depth = renderer(verts_side_view.cpu().numpy(), triangles, background, mesh_colors=mesh_colors)
            result_image.append(cv2.resize(rendered_sv_image, (h, h)))
    
        if 'rotate_mesh' in rendering_cfgs['items']:
            rot_trans = cam_trans.unsqueeze(1)
            rot_trans[:,:,2] /= 1.5
            verts_tran_rot = outputs['verts'][img_inds] + rot_trans
            rotate_renderings = rendering_mesh_rotating_view(verts_tran_rot, renderer, triangles, image, background)
            time_stamp = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(int(round(time.time()*1000))/1000))
            save_path = os.path.join(os.path.expanduser("~"),'rotate-{}.mp4'.format(time_stamp))
            frame_rate = 24
            writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, rotate_renderings[0].shape[:2])
            for frame in rotate_renderings:
                writer.write(frame)
            writer.release()
            print('-'*6,'Rotate_mesh has been saved to ', save_path)
        
    if 'pj2d' in rendering_cfgs['items']:
        img_skeleton2d = draw_skeleton_multiperson(copy.deepcopy(image), outputs['pj2d_org'][img_inds].cpu().numpy()[:,:24], mesh_colors*255)
        result_image.append(img_skeleton2d)
    
    if 'j3d' in rendering_cfgs['items']:
        plot_3dpose = Plotter3dPoses(canvas_size=(h,h))
        joint_trans = (outputs['j3d'][img_inds] + cam_trans.unsqueeze(1)).cpu().numpy()[:,:24]*3
        img_skeleton3d = plot_3dpose.plot(joint_trans,colors=mesh_colors*255)
        result_image.append(img_skeleton3d)
    
    if 'center_conf' in rendering_cfgs['items']:
        for ind, kp in enumerate(outputs['pj2d_org'][img_inds].cpu().numpy()[:,0]):
            cv2.putText(result_image[0], '{:.3f}'.format(outputs['center_confs'][img_inds][ind].item()), tuple(kp.astype(int)+10), cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),1)  

    if 'tracking' in rendering_cfgs['items'] and 'track_ids' in outputs:
        for ind, kp in enumerate(outputs['pj2d_org'][img_inds].cpu().numpy()[:,0]):
            cv2.putText(result_image[1], '{:d}'.format(outputs['track_ids'][img_inds][ind]), tuple(kp.astype(int)), cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)  
    
    return np.concatenate(result_image, 0)

def visualize_predictions(outputs, imgpath, FOV, seq_save_dir, smpl_model_path):
    rendering_cfgs = {'mesh_color':'identity', 'items':'mesh,tracking', 'renderer': 'sim3dr'} # 'world_mesh' 
    #rendering_cfgs = {'mesh_color':'identity', 'items':'mesh,tracking', 'renderer': 'pyrender'} # 'world_mesh'
    renderer = setup_renderer(name=rendering_cfgs['renderer'], FOV=FOV)
    os.makedirs(seq_save_dir, exist_ok=True)
    render_images_path = visulize_result(renderer, outputs, imgpath, rendering_cfgs, seq_save_dir, smpl_model_path)
    save_video(render_images_path, seq_save_dir+'.mp4', frame_rate=25)


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

    def plot(self, pose_3ds, bones=smpl24_connMat, colors=[(255, 0, 0)], img=None):
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