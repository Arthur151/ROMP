import numpy as np
import torch
import cv2
import torch.nn.functional as F
import trimesh
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import copy

import os, sys
import pytorch3d.renderer as pyr

import constants
import config
from config import args
import utils.projection as proj
from utils.train_utils import process_idx, determine_rendering_order
from .renderer_pt3d import get_renderer
from utils import denormalize_cam_params_to_trans
from pytorch3d.renderer import look_at_view_transform, get_world_to_view_transform
from .web_vis import write_to_html, convert_3dpose_to_line_figs, convert_image_list
from collections import OrderedDict

import pandas
import pickle

default_cfg = {'save_dir':None, 'vids':None, 'settings':[]} # 'put_org'

class Visualizer(object):
    def __init__(self, resolution=(512,512), result_img_dir = None, with_renderer=False):
        self.resolution = resolution
        self.FOV = np.radians(args().FOV)
        self.smpl_face = torch.load(args().smpl_model_path)['f']
        if with_renderer:
            self.perps_proj = args().perspective_proj
            T= None if self.perps_proj else torch.Tensor([[0.,0.,100]])                
            self.renderer = get_renderer(resolution=self.resolution, perps=self.perps_proj, T=T)
            if args().model_version>=4:
                # dervice the camera pose/position via setting the object location (at) and the view dist/evelevation/azim in a sphere coodinate
                # The position is just a sphere coodinate which takes "at" as the origin with the corresponding angles.
                # World coordinates (right-hand coord _|/ ) +Y up, +X left and +Z in.
                R,T = look_at_view_transform(dist=5,elev=80,azim=180,at=torch.Tensor([[0.,0.,3]]))
                self.bv_renderer = get_renderer(resolution=self.resolution, perps=False, R=R, T=T)
        self.result_img_dir = result_img_dir
        self.heatmap_kpnum = 17
        self.vis_size = resolution
        self.mesh_color = (torch.Tensor([[[0.65098039, 0.74117647, 0.85882353]]])*255).long()
        self.color_table = np.array([[255,0,0], [0,255,0], [0,0,255], [0,255,255], [255,0,255], [255,255,0], [128,128,0], [0,128,128], [128,0,128]])
        self.skeleton_3D_ploter = Plotter3dPoses()
        self.color_class_dict = {0:{0:[0.94,1.,1.],1:[0.49,1.,0],2:[0,1.,1.],3:[1.,0.98,0.804], -1:[.9,.9,.8]},\
                                1:{0:[1.,0.753,0.796],1:[1,0.647,0],2:[1,0.431,0.706],3:[1.,0.98,0.804], -1:[.9,.9,.8]},\
                                2:{0:[.9,.9,.8],1:[.9,.9,.8],2:[.9,.9,.8],3:[.9,.9,.8], -1:[.9,.9,.8]}}
        # adult lightcyan　浅蓝；　teen Chartreuse 草绿色;  kid Cyan 青色; baby RosyBrown 玫瑰褐;  -1 silver 银色
        self.age_color_dict = {0:[0.94,1.,1.], 1:[0.49,1.,0], 2:[0, 1., 1.], 3:[1., 0.41, 0.41], -1:[.9,.9,.8]}
        self.age_name_dict = {0:'adult', 1:'teen',2:'kid', 3:'baby', -1:'NotSure'}
        self.gender_name_dict = {0:'male', 1:'female', 2:'neutral'}


    def visualize_renderer_verts_list(self, verts_list, faces_list=None, images=None, bird_view=False, auto_cam=False, cam_params=None,\
                                            colors=torch.Tensor([.9, .9, .8]), trans=None, thresh=0.):
        verts_list = [verts.contiguous() for verts in verts_list]
        if faces_list is None:
            faces_list = [self.smpl_face.repeat(len(verts), 1, 1).to(verts.device) for verts in verts_list]
        
        if bird_view:
            renderer = self.bv_renderer
        else:
            renderer = self.renderer
        rendered_imgs = []
        for ind, (verts, faces) in enumerate(zip(verts_list, faces_list)):
            if trans is not None:
                verts += trans[ind].unsqueeze(1)
            if auto_cam:
                cam_params = calc_auto_cam_params(verts)
            
            color = colors[ind] if isinstance(colors, list) else colors
            #if color is None:
            #    print('None color,', color, colors)

            if self.perps_proj:
                rendered_img = renderer(verts, faces, colors=color, merge_meshes=True, cam_params=cam_params)
            else:
                verts[:,:,2] -= 1.
                rendered_img = renderer(verts, faces, colors=color, merge_meshes=False, cam_params=cam_params)
                rendered_img = determine_rendering_order(rendered_img)
            rendered_imgs.append(rendered_img)
        rendered_imgs = torch.cat(rendered_imgs, 0).cpu().numpy()
        if rendered_imgs.shape[-1]==4:
            transparent = rendered_imgs[:,:, :, -1]
            rendered_imgs = rendered_imgs[:,:,:,:-1] * 255
        
        visible_weight = 0.9
        if images is not None:
            valid_mask = (transparent > thresh)[:,:, :,np.newaxis]
            rendered_imgs = rendered_imgs * valid_mask * visible_weight + images * valid_mask * (1-visible_weight) + (1 - valid_mask) * images
        return rendered_imgs.astype(np.uint8)

    def mark_classify_results_on_img(self, images, class_preds, class_probs, center_coords):
        for img_id in range(len(images)):
            for class_pred, class_prob, center_coord in zip(class_preds[img_id], class_probs[img_id], center_coords[img_id]):
                age_cls, age_prob = class_pred, int(class_prob*100)
                center_loc = (center_coord * (args().input_size/args().centermap_size)).astype(np.int)
                text = '{} {}%'.format(self.age_name_dict[age_cls], age_prob)
                cv2.putText(images[img_id], text, (center_loc[0], center_loc[1]), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 1)
                #text2 = '{} {}%'.format(self.gender_name_dict[gender_cls], gender_prob)#[reorganize_idx[vids]==used_idx_org[idx]]
                #cv2.putText(images[img_id], text2, (center_loc[0], center_loc[1]+14), cv2.FONT_HERSHEY_PLAIN, 1.2, (255,0,0), 1)
        return images

    def show_verts_on_imgs(self, outputs, meta_data, all_inds, org_imgs, mesh_colors=None,img_names=None, put_org=True, plot_dict=None, save2html=True, drop_texts=False):
        used_org_inds, per_img_inds, img_inds_org = all_inds
        if self.perps_proj:
            per_img_verts_list = [outputs['verts'][inds].detach() for inds in per_img_inds]
            trans = [outputs['cam_trans'][inds].detach() for inds in per_img_inds]

            if mesh_colors is None:
                mesh_colors = torch.Tensor([.9, .9, .8])
                if 'Age_preds' in outputs and 'kid_offsets_pred' in outputs:
                    class_preds = [outputs['Age_preds'][inds].detach().cpu().numpy() for inds in per_img_inds]
                    class_probs = [outputs['kid_offsets_pred'][inds].detach().cpu().numpy() for inds in per_img_inds]
                    mesh_colors = [torch.Tensor([self.age_color_dict[age] for age in age_preds]) for age_preds in class_preds]
            rendered_imgs = self.visualize_renderer_verts_list(per_img_verts_list, images=org_imgs.copy(), trans=trans, colors=mesh_colors)
            
            if 'Age_preds' in outputs and 'kid_offsets_pred' in outputs:
                if not drop_texts:
                    center_preds = [outputs['center_preds'][inds].detach().cpu().numpy() for inds in per_img_inds]
                    rendered_imgs = self.mark_classify_results_on_img(rendered_imgs, class_preds, class_probs, center_preds)
                    rendered_imgs_bv = self.visualize_renderer_verts_list(per_img_verts_list, trans=trans, colors=mesh_colors, bird_view=True, auto_cam=True)
                    if plot_dict is not None:
                        plot_dict['mesh_rendering_imgs_bv'] = {'figs':convert_image_list(rendered_imgs_bv), 'type':'image'} if save2html else {'figs':rendered_imgs_bv, 'type':'image'}

        elif 'verts_camed' in outputs:
            per_img_verts_list = [outputs['verts_camed'][inds].detach() for inds in per_img_inds]
            rendered_imgs = self.visualize_renderer_verts_list(per_img_verts_list, images=org_imgs.copy(), colors=mesh_colors)
        
        if put_org:
            offsets = meta_data['offsets'].cpu().numpy().astype(np.int)[img_inds_org]
            img_pad_size, crop_trbl, pad_trbl = offsets[:,:2], offsets[:,2:6], offsets[:,6:10]
            rendering_onorg_images = []
            for inds, j in enumerate(used_org_inds):
                org_imge = cv2.imread(img_names[inds])
                (ih, iw), (ph,pw) = org_imge.shape[:2], img_pad_size[inds]
                resized_images = cv2.resize(rendered_imgs[inds], (ph+1, pw+1), interpolation = cv2.INTER_CUBIC)
                (ct, cr, cb, cl), (pt, pr, pb, pl) = crop_trbl[inds], pad_trbl[inds]
                org_imge[ct:ih-cb, cl:iw-cr] = resized_images[pt:ph-pb, pl:pw-pr]
                rendering_onorg_images.append(org_imge)
            if plot_dict is not None:
                plot_dict['mesh_rendering_orgimgs'] = {'figs':convert_image_list(rendering_onorg_images), 'type':'image'} if save2html else {'figs':rendering_onorg_images, 'type':'image'}
            rendered_imgs = rendering_onorg_images

        if plot_dict is not None:
            plot_dict['mesh_rendering_imgs'] = {'figs':convert_image_list(rendered_imgs), 'type':'image'} if save2html else {'figs':rendered_imgs, 'type':'image'}
        return rendered_imgs


    def visulize_result(self, outputs, meta_data, show_items=['org_img', 'mesh'], vis_cfg=default_cfg, save2html=True, **kwargs):
        vis_cfg = dict(default_cfg, **vis_cfg)
        if vis_cfg['save_dir'] is None:
            vis_cfg['save_dir'] = self.result_img_dir
        os.makedirs(vis_cfg['save_dir'], exist_ok=True)

        used_org_inds, per_img_inds = process_idx(outputs['reorganize_idx'], vids=vis_cfg['vids'])
        img_inds_org = [inds[0] for inds in per_img_inds]
        img_names = np.array(meta_data['imgpath'])[img_inds_org]
        org_imgs = meta_data['image'].cpu().numpy().astype(np.uint8)[img_inds_org]
        detection_flag = outputs['detection_flag'].sum()>0

        plot_dict = OrderedDict()
        for vis_name in show_items:
            if vis_name == 'org_img':
                if save2html:
                    plot_dict['org_img'] = {'figs':convert_image_list(org_imgs), 'type':'image'}
                else:
                    plot_dict['org_img'] = {'figs':org_imgs, 'type':'image'}

            if vis_name == 'mesh' and detection_flag:
                rendered_imgs = self.show_verts_on_imgs(outputs, meta_data, (used_org_inds, per_img_inds, img_inds_org), \
                    org_imgs, img_names=img_names, put_org='put_org' in vis_cfg['settings'], plot_dict=plot_dict, save2html=save2html)

            if vis_name == 'j3d' and detection_flag:
                real_aligned, pred_aligned, pos3d_vis_mask, joint3d_bones = kwargs['kp3ds']
                real_3ds = (real_aligned*pos3d_vis_mask.unsqueeze(-1)).cpu().numpy()
                predicts = (pred_aligned*pos3d_vis_mask.unsqueeze(-1)).detach().cpu().numpy()
                if save2html:
                    plot_dict['j3d'] = {'figs':convert_3dpose_to_line_figs([predicts, real_3ds], joint3d_bones), 'type':'skeleton'}
                else:
                    skeleton_3ds = []
                    for inds in per_img_inds:
                        for real_pose_3d, pred_pose_3d in zip(real_3ds[inds], predicts[inds]):
                            skeleton_3d = self.skeleton_3D_ploter.encircle_plot([real_pose_3d, pred_pose_3d], \
                                joint3d_bones, colors=[(255, 0, 0), (0, 255, 255)])
                            skeleton_3ds.append(skeleton_3d)
                    plot_dict['j3d'] = {'figs':np.array(skeleton_3ds), 'type':'skeleton'}

            if vis_name == 'pj2d' and detection_flag and outputs['pj2d'].shape[1]==54:
                kp_imgs = []
                for img_id, inds_list in enumerate(per_img_inds):
                    org_img = copy.deepcopy(org_imgs[img_id])
                    try:
                        for kp2d_vis in outputs['pj2d'][inds_list]:
                            if len(kp2d_vis)>0:
                                kp2d_vis = ((kp2d_vis+1)/2 * org_imgs.shape[1])
                                #org_img = draw_skeleton(org_img, kp2d_vis, bones=constants.body17_connMat, cm=constants.cm_body17)
                                org_img = draw_skeleton(org_img, kp2d_vis, bones=constants.All54_connMat, cm=constants.cm_All54)
                    except Exception as error:
                        print(error, ' reported while drawing 2D pose')
                    kp_imgs.append(org_img)
                if save2html:
                    kp_imgs = convert_image_list(kp_imgs)
                plot_dict['pj2d'] = {'figs':kp_imgs, 'type':'image'}

            if vis_name == 'hp_aes' and 'kp_ae_maps' in outputs and detection_flag:
                heatmaps_AEmaps = []
                #hp_aes = torch.nn.functional.interpolate(hp_aes[vids],size=(img_size,img_size),mode='bilinear',align_corners=True)
                for img_id, hp_ae in enumerate(outputs['kp_ae_maps'][used_org_inds]):
                    img_bk = cv2.resize(org_imgs[img_id].copy(),(hp_ae.shape[1],hp_ae.shape[2]))
                    heatmaps_AEmaps.append(np.vstack([make_heatmaps(img_bk, hp_ae[:self.heatmap_kpnum]),make_tagmaps(img_bk, hp_ae[self.heatmap_kpnum:])]))
            
            if vis_name == 'centermap' and 'center_map' in outputs and detection_flag:
                centermaps_list = []
                for img_id, centermap in enumerate(outputs['center_map'][used_org_inds]):
                    img_bk = cv2.resize(org_imgs[img_id].copy(),org_imgs.shape[1:3])
                    centermaps_list.append(make_heatmaps(img_bk, centermap))
                if save2html:
                    centermaps_list = convert_image_list(centermaps_list)
                plot_dict['centermap'] = {'figs':centermaps_list, 'type':'image'}
            
        if save2html:
            write_to_html(img_names, plot_dict, vis_cfg)

        return plot_dict, img_names

    def draw_skeleton(self, image, pts, **kwargs):
        return draw_skeleton(image, pts, **kwargs)

    def draw_skeleton_multiperson(self, image, pts, **kwargs):
        return draw_skeleton_multiperson(image, pts, **kwargs)


def calc_auto_cam_params(verts):
    x_max, x_min = verts[:,:,0].max().item(), verts[:,:,0].min().item()
    y_max, y_min = verts[:,:,1].max().item(), verts[:,:,1].min().item()
    z_max, z_min = verts[:,:,2].max().item(), verts[:,:,2].min().item()
    cx, cy, cz = np.mean([x_max,x_min]), np.mean([y_max,y_min]), np.mean([z_max,z_min])
    span = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2. + 1
    #xyz_ranges = dict(znear=max(cz-span, 0.5), zfar=cz+span, max_y=cy+span, min_y=cy-span, max_x=cx+span, min_x=cx-span)
    xyz_ranges = dict(znear=0.5, zfar=100, max_y=span, min_y=-span, max_x=span, min_x=-span)
    #height =  2* span / np.tan(np.radians(fov/2.))
    height = 20
    R,T = look_at_view_transform(dist=height,elev=280,azim=0,at=torch.Tensor([[cy, cx, cz]]))
    #return (R, T, fov)
    return (R, T, xyz_ranges)


def make_heatmaps(image, heatmaps):
    heatmaps = torch.nn.functional.interpolate(heatmaps[None],size=image.shape[:2],mode='bilinear')[0]
    heatmaps = heatmaps.mul(255)\
                       .clamp(0, 255)\
                       .byte()\
                       .detach().cpu().numpy()

    num_joints, height, width = heatmaps.shape
    image_grid = np.zeros((height, (num_joints+1)*width, 3), dtype=np.uint8)

    for j in range(num_joints):
        heatmap = heatmaps[j, :, :]
        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        image_fused = colored_heatmap*0.7 + image*0.3

        width_begin = width * (j+1)
        width_end = width * (j+2)
        image_grid[:, width_begin:width_end, :] = image_fused

    image_grid[:, 0:width, :] = image

    return image_grid


def make_tagmaps(image, tagmaps):
    num_joints, height, width = tagmaps.shape
    image_resized = cv2.resize(image, (int(width), int(height)))

    image_grid = np.zeros((height, (num_joints+1)*width, 3), dtype=np.uint8)

    for j in range(num_joints):
        tagmap = tagmaps[j, :, :]
        min = float(tagmap.min())
        max = float(tagmap.max())
        tagmap = tagmap.add(-min)\
                       .div(max - min + 1e-5)\
                       .mul(255)\
                       .clamp(0, 255)\
                       .byte()\
                       .detach().cpu().numpy()

        colored_tagmap = cv2.applyColorMap(tagmap, cv2.COLORMAP_JET)
        image_fused = colored_tagmap*0.9 + image_resized*0.1

        width_begin = width * (j+1)
        width_end = width * (j+2)
        image_grid[:, width_begin:width_end, :] = image_fused

    image_grid[:, 0:width, :] = image_resized

    return image_grid


def draw_skeleton(image, pts, bones=None, cm=None, label_kp_order=False,r=3):
    for i,pt in enumerate(pts):
        if len(pt)>1:
            if pt[0]>0 and pt[1]>0:
                image = cv2.circle(image,(int(pt[0]), int(pt[1])),r,(255,0,0),-1)
                if label_kp_order and i in bones:
                    img=cv2.putText(image,str(i),(int(pt[0]), int(pt[1])),cv2.FONT_HERSHEY_COMPLEX,1,(255,215,0),1)
    
    if bones is not None:
        if cm is None:
            set_colors = np.array([[255,0,0] for i in range(len(bones))]).astype(np.int)
        else:
            if len(bones)>len(cm):
                cm = np.concatenate([cm for _ in range(len(bones)//len(cm)+1)],0)
            set_colors = cm[:len(bones)].astype(np.int)
        bones = np.concatenate([bones,set_colors],1).tolist()
        for line in bones:
            pa = pts[line[0]]
            pb = pts[line[1]]
            if (pa>0).all() and (pb>0).all():
                xa,ya,xb,yb = int(pa[0]),int(pa[1]),int(pb[0]),int(pb[1])
                image = cv2.line(image,(xa,ya),(xb,yb),(int(line[2]), int(line[3]), int(line[4])),r)
    return image

def draw_skeleton_multiperson(image, pts_group,**kwargs):
    for pts in pts_group:
        image = draw_skeleton(image, pts, **kwargs)
    return image


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

    def plot(self, pose_3ds, bones, colors=[(255, 255, 255)], img=None):
        img = np.ones((self.canvas_size[0],self.canvas_size[1],3), dtype=np.uint8) * 255 if img is None else img
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
            cv2.line(img, tuple(edge_vertices[0]), tuple(edge_vertices[1]), color, 2, cv2.LINE_AA)

    def _get_rotation(self, theta, phi):
        sin, cos = math.sin, math.cos
        return np.array([
            [ cos(theta),  sin(theta) * sin(phi)],
            [-sin(theta),  cos(theta) * sin(phi)],
            [ 0,                       -cos(phi)]
        ], dtype=np.float32)  # transposed

def test_visualizer():
    visualizer = Visualizer(resolution=(512,512), input_size=args().input_size, result_img_dir=args().result_img_dir, with_renderer=True)

if __name__ == '__main__':
    test_visualizer()