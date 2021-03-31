import numpy as np
import torch
import cv2

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys, os
root_dir = os.path.join(os.path.dirname(__file__),'..')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
import constants
import config
from .renderer import get_renderer


class Visualizer(object):
    def __init__(self,resolution = (512,512,3),input_size=512,result_img_dir = None,with_renderer=False):
        self.input_size = input_size
        if with_renderer:
            self.renderer = get_renderer(resolution=resolution)
        self.result_img_dir = result_img_dir
        self.heatmap_kpnum = 17
        self.vis_size = resolution[:2]
        self.mesh_color = (torch.Tensor([[[0.65098039, 0.74117647, 0.85882353]]])*255).long()

    def add_mesh_to_writer(self,writer, verts, name):
        writer.add_mesh(name, vertices=verts.detach().cpu(), \
            colors=self.mesh_color.repeat(verts.shape[0],verts.shape[1],1), \
            faces=self.renderer.faces.cpu(),global_step=self.global_count)

    def visualize_renderer(self, verts, images=None, reorganize_idx=None,thresh=0.2,visible_weight=0.9, scale_thresh=100):
        verts = verts.detach().cpu().numpy()
        renders = []
        for vert in verts:
            render_result = self.renderer(vert, color=[.9, .9, .8])
            renders.append(render_result)
        renders = np.array(renders)
        
        # sorting the render via the rendering area size
        if reorganize_idx is not None:
            renders_summoned = []
            for idxs in reorganize_idx:
                main_renders = renders[idxs[0]]
                main_render_mask = main_renders[:, :, -1] > thresh
                render_scale_map = np.zeros(self.vis_size)
                render_scale_map[main_render_mask] = main_render_mask.sum()
                for jdx in range(1,len(idxs)):
                    other_idx = idxs[jdx]
                    other_renders = renders[other_idx]
                    other_render_mask = other_renders[:, :, -1] > thresh
                    render_scale_map_other = np.zeros(self.vis_size)
                    render_scale_map_other[other_render_mask] = other_render_mask.sum()
                    other_render_mask = render_scale_map_other>(render_scale_map+scale_thresh)
                    render_scale_map[other_render_mask] = other_render_mask.sum()
                    main_renders[other_render_mask] = other_renders[other_render_mask]
                renders_summoned.append(main_renders)
            renders = np.array(renders_summoned)

        visible_weight = 0.9
        if images is not None:
            valid_mask = (renders[:,:, :, -1] > thresh)[:,:, :,np.newaxis]
            #renders[valid_mask, :-1] = images[valid_mask]
            if renders.shape[-1]==4:
                renders = renders[:,:,:,:-1]
            
            renders = renders * valid_mask * visible_weight + images * valid_mask * (1-visible_weight) + (1 - valid_mask) * images
        return renders.astype(np.uint8)

    def visulize_result_onorg(self, vertices, verts_camed, data, reorganize_idx=None, save_img=False,hp_aes=None, centermaps=None,**kwargs): #pkps, kps, 
        img_size=1024
        if reorganize_idx is not None:
            vids_org = np.unique(reorganize_idx)
            verts_vids, single_vids, new_idxs = [],[],[]
            count = 0
            for idx, vid in enumerate(vids_org):
                verts_vids.append(np.where(reorganize_idx==vid)[0])
                single_vids.append(np.where(reorganize_idx==vid)[0][0])
                new_idx = []
                for j in range((reorganize_idx==vid).sum()):
                    new_idx.append(count)
                    count+=1
                new_idxs.append(new_idx)
            verts_vids = np.concatenate(verts_vids)
            assert count==len(verts_vids)
        else:
            new_idxs = None
            vids_org, verts_vids, single_vids = [np.arange(data['image_org'].shape[0]) for _ in range(3)]

        images = data['image_org'].cpu().contiguous().numpy().astype(np.uint8)[single_vids]
        if images.shape[1] != self.vis_size[0]:
            images_new = []
            for image in images:
                images_new.append(cv2.resize(image, tuple(self.vis_size)))
            images = np.array(images_new)
        rendered_imgs = self.visualize_renderer(verts_camed[verts_vids], images=images, reorganize_idx=new_idxs)
        show_list = [images, rendered_imgs]

        if centermaps is not None:
            centermaps_list = []
            centermaps = torch.nn.functional.interpolate(centermaps[vids_org],size=(img_size,img_size),mode='bilinear')
            for idx,centermap in enumerate(centermaps):
                img_bk = cv2.resize(images[idx].copy(),(img_size,img_size))[:,:,::-1]
                centermaps_list.append(make_heatmaps(img_bk, centermap))

        out_list = []
        for idx in range(len(vids_org)):
            result_img = np.hstack([item[idx] for item in show_list])
            out_list.append(result_img[:,:,::-1])
            
        if save_img:
            img_names = np.array(data['imgpath'])[single_vids]
            os.makedirs(self.result_img_dir, exist_ok=True)
            for idx,result_img in enumerate(out_list):
                name = img_names[idx].split('/')[-2]+'-'+img_names[idx].split('/')[-1]
                name_save = os.path.join(self.result_img_dir,name)
                cv2.imwrite(name_save,result_img)
                if centermaps is not None:
                    cv2.imwrite(name_save.replace('.jpg','_centermap.jpg'),centermaps_list[idx])
        return np.array(out_list)

    def draw_skeleton(self, image, pts, **kwargs):
        return draw_skeleton(image, pts, **kwargs)

    def draw_skeleton_multiperson(self, image, pts, **kwargs):
        return draw_skeleton_multiperson(image, pts, **kwargs)

    def make_mp4(self,images,name):
        num = images.shape[0]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_movie = cv2.VideoWriter(name+'.mp4', fourcc, 50, (images.shape[2], images.shape[1]))
        for i in range(num):
            if i%100==0:
                print('Writing frame: ',i,'/',num)
            output_movie.write(images[i])


def make_heatmaps(image, heatmaps):
    heatmaps = heatmaps.mul(255)\
                       .clamp(0, 255)\
                       .byte()\
                       .detach().cpu().numpy()

    num_joints, height, width = heatmaps.shape
    image_resized = cv2.resize(image, (int(width), int(height)))

    image_grid = np.zeros((height, (num_joints+1)*width, 3), dtype=np.uint8)

    for j in range(num_joints):
        # add_joints(image_resized, joints[:, j, :])
        heatmap = heatmaps[j, :, :]
        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        image_fused = colored_heatmap*0.7 + image_resized*0.3

        width_begin = width * (j+1)
        width_end = width * (j+2)
        image_grid[:, width_begin:width_end, :] = image_fused

    image_grid[:, 0:width, :] = image_resized

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


def draw_skeleton(image, pts, bones=None, cm=None,put_text=False,r=3):
    for i,pt in enumerate(pts):
        if len(pt)>1:
            if pt[0]>0 and pt[1]>0:
                image = cv2.circle(image,(int(pt[0]), int(pt[1])),r,(255,0,0),-1)
                if put_text:
                    img=cv2.putText(image,str(i),(int(pt[0]), int(pt[1])),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),1)

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