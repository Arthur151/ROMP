import imp
import cv2
import torch
import numpy as np
from .vis_utils import mesh_color_left2right, mesh_color_trackID, rotate_view_perspective, rendering_mesh_rotating_view, \
    rotate_view_weak_perspective, draw_skeleton_multiperson, Plotter3dPoses
import copy
import time
import os

def setup_renderer(name='sim3dr', **kwargs):
    if name == 'sim3dr':
        from vis_human.sim3drender import Sim3DR
        renderer = Sim3DR(**kwargs)
    elif name == 'pyrender':
        from vis_human.pyrenderer import Py3DR
        renderer = Py3DR(**kwargs)
    elif name == 'open3d':
        from vis_human.o3drenderer import O3DDR
        renderer = O3DDR(multi_mode=True,**kwargs) #not self.settings.show_largest
    return renderer

def rendering_romp_bev_results(renderer, outputs, image, rendering_cfgs, alpha=1):
    triangles = outputs['smpl_face'].cpu().numpy().astype(np.int32)
    h, w = image.shape[:2]
    #length = max(h, w)
    background = np.ones([h, h, 3], dtype=np.uint8) * 255
    result_image = [image]

    cam_trans = outputs['cam_trans']
    if rendering_cfgs['mesh_color'] == 'identity':
        if 'track_ids' in outputs:
            mesh_colors = mesh_color_trackID(outputs['track_ids'])
        else:
            mesh_colors = mesh_color_left2right(cam_trans)
    elif rendering_cfgs['mesh_color'] == 'same':
        mesh_colors = np.array([[.9, .9, .8] for _ in range(len(cam_trans))])

    if rendering_cfgs['renderer'] == 'sim3dr':
        depth_order = torch.sort(cam_trans[:,2].cpu(),descending=True).indices.numpy()
        vertices = outputs['verts_camed_org'][depth_order].cpu().numpy()
        mesh_colors = mesh_colors[depth_order]
        verts_tran = (outputs['verts'] + cam_trans.unsqueeze(1))[depth_order]
        vertices[:,:,2] = vertices[:,:,2]*-1 
        verts_tran[:,:,2] = verts_tran[:,:,2]*-1
        if 'mesh' in rendering_cfgs['items']:
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
        verts_tran = outputs['verts'] + cam_trans.unsqueeze(1)
        
        if 'mesh' in rendering_cfgs['items']:
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
            verts_tran_rot = outputs['verts'] + rot_trans
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
        img_skeleton2d = draw_skeleton_multiperson(copy.deepcopy(image), outputs['pj2d_org'].cpu().numpy()[:,:24], mesh_colors*255)
        result_image.append(img_skeleton2d)
    
    if 'j3d' in rendering_cfgs['items']:
        plot_3dpose = Plotter3dPoses(canvas_size=(h,h))
        joint_trans = (outputs['joints'] + cam_trans.unsqueeze(1)).cpu().numpy()[:,:24]*3
        img_skeleton3d = plot_3dpose.plot(joint_trans,colors=mesh_colors*255)
        result_image.append(img_skeleton3d)
    
    if 'center_conf' in rendering_cfgs['items']:
        for ind, kp in enumerate(outputs['pj2d_org'].cpu().numpy()[:,0]):
            cv2.putText(result_image[1], '{:.3f}'.format(outputs['center_confs'][ind]), tuple(kp.astype(int)), cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),1)  

    if 'tracking' in rendering_cfgs['items'] and 'track_ids' in outputs:
        for ind, kp in enumerate(outputs['pj2d_org'].cpu().numpy()[:,0]):
            cv2.putText(result_image[1], '{:d}'.format(outputs['track_ids'][ind]), tuple(kp.astype(int)), cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)  
    
    outputs['rendered_image'] = np.concatenate(result_image, 1)
    return outputs
