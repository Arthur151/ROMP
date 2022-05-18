
import cv2
import numpy as np
import os, sys
import os.path as osp
import torch
from torch import nn
import argparse
import copy

from .model import BEVv1
from .post_parser import SMPLA_parser, body_mesh_projection2image, pack_params_dict,\
    suppressing_redundant_prediction_via_projection, remove_outlier, denormalize_cam_params_to_trans
from romp.utils import img_preprocess, create_OneEuroFilter, check_filter_state, \
    time_cost, download_model, determine_device, ResultSaver, WebcamVideoStream, \
    wait_func, collect_frame_path, progress_bar, smooth_results, convert_tensor2numpy, save_video_results
from vis_human import setup_renderer, rendering_romp_bev_results

model_dict = {
    1: 'BEV_ft_agora.pth',
    2: 'BEV.pth',
}
model_id = 2
conf_dict = {1:[0.25, 20, 2], 2:[0.1, 20, 1.6]}
long_conf_dict = {1:[0.12, 20, 1.5, 0.46], 2:[0.08, 20, 1.6, 0.8]}

def bev_settings(input_args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description = 'ROMP: Monocular, One-stage, Regression of Multiple 3D People')
    parser.add_argument('-m', '--mode', type=str, default='image', help = 'Inferece mode, including image, video, webcam')
    parser.add_argument('--model_id', type=int, default=2, help = 'Whether to process the input as a long image, sliding window way')
    parser.add_argument('-i', '--input', type=str, default='/home/yusun/CenterMesh/simple_romp/test/ages.png', help = 'Path to the input image / video')
    parser.add_argument('-o', '--save_path', type=str, default=osp.join(osp.expanduser("~"),'BEV_results'), help = 'Path to save the results')
    parser.add_argument('--crowd', action='store_true', help = 'Whether to process the input as a long image, sliding window way')
    parser.add_argument('--GPU', type=int, default=0, help = 'The gpu device number to run the inference on. If GPU=-1, then running in cpu mode')

    parser.add_argument('--overlap_ratio', type=float, default=long_conf_dict[model_id][3], help = 'The frame_rate of saved video results')
    parser.add_argument('--center_thresh', type=float, default=conf_dict[model_id][0], help = 'The confidence threshold of positive detection in 2D human body center heatmap.')
    parser.add_argument('--nms_thresh', type=float, default=conf_dict[model_id][1], help = 'The 2D-pose-projection similarity threshold of suppressing overlapping predictions.')
    parser.add_argument('--relative_scale_thresh', type=float, default=conf_dict[model_id][2], help = 'The confidence threshold of positive detection in 2D human body center heatmap.')
    parser.add_argument('--show_largest', action='store_true', help = 'Whether to show the largest person only')
    parser.add_argument('--show_patch_results', action='store_true', help = 'During processing long image, whether to show the results of intermediate results of each patch.')
    parser.add_argument('--calc_smpl', action='store_false', help = 'Whether to calculate the smpl mesh from estimated SMPL parameters')
    parser.add_argument('--renderer', type=str, default='sim3dr', help = 'Choose the renderer for visualizaiton: pyrender (great but slow), sim3dr (fine but fast), open3d (webcam)')
    parser.add_argument('--render_mesh', action='store_false', help = 'Whether to render the estimated 3D mesh mesh to image')
    parser.add_argument('--show', action='store_true', help = 'Whether to show the rendered results')
    parser.add_argument('--show_items', type=str, default='mesh,mesh_bird_view', help = 'The items to visualized, including mesh,pj2d,j3d,mesh_bird_view,mesh_side_view,center_conf,rotate_mesh. splited with ,')
    parser.add_argument('--save_video', action='store_true', help = 'Whether to save the video results')
    parser.add_argument('--frame_rate', type=int, default=24, help = 'The frame_rate of saved video results')
    parser.add_argument('--smpl_path', type=str, default=osp.join(osp.expanduser("~"),'.romp','smpla_packed_info.pth'), help = 'The path of SMPL-A model file')
    parser.add_argument('--smil_path', type=str, default=osp.join(osp.expanduser("~"),'.romp','smil_packed_info.pth'), help = 'The path of SMIL model file')
    parser.add_argument('--model_path', type=str, default=osp.join(osp.expanduser("~"),'.romp',model_dict[model_id]), help = 'The path of BEV checkpoint')

    # not support temporal processing now
    parser.add_argument('-t', '--temporal_optimize', action='store_true', help = 'Whether to use OneEuro filter to smooth the results')
    parser.add_argument('-sc','--smooth_coeff', type=float, default=3., help = 'The smoothness coeff of OneEuro filter, the smaller, the smoother.')
    args = parser.parse_args(input_args)

    if args.model_id != 2:
        args.model_path = osp.join(osp.expanduser("~"),'.romp',model_dict[args.model_id])
        args.center_thresh = conf_dict[args.model_id][0]
        args.nms_thresh = conf_dict[args.model_id][1]
        args.relative_scale_thresh = conf_dict[model_id][2]
    if not torch.cuda.is_available():
        args.GPU = -1
    if args.show:
        args.render_mesh = True
    if args.render_mesh or args.show_largest:
        args.calc_smpl = True
    if not os.path.exists(args.smpl_path):
        smpl_url = 'https://github.com/Arthur151/ROMP/releases/download/V2.0/smpla_packed_info.pth'
        download_model(smpl_url, args.smpl_path, 'SMPL-A')
    if not os.path.exists(args.smil_path):
        smil_url = 'https://github.com/Arthur151/ROMP/releases/download/V2.0/smil_packed_info.pth'
        download_model(smil_url, args.smil_path, 'SMIL')
    if not os.path.exists(args.model_path):
        romp_url = 'https://github.com/Arthur151/ROMP/releases/download/V2.0/'+model_dict[model_id]
        download_model(romp_url, args.model_path, 'BEV')
    if args.crowd:
        args.center_thresh = long_conf_dict[args.model_id][0]
        args.nms_thresh = long_conf_dict[args.model_id][1]
        args.relative_scale_thresh = long_conf_dict[model_id][2]
        args.overlap_ratio = long_conf_dict[args.model_id][3]
    
    return args

default_settings = bev_settings(input_args=[])

class BEV(nn.Module):
    def __init__(self, romp_settings):
        super(BEV, self).__init__()
        self.settings = romp_settings
        self.tdevice = determine_device(self.settings.GPU)
        self._build_model_()
        self._initilization_()
    
    def _build_model_(self):
        model = BEVv1(center_thresh=self.settings.center_thresh).eval()
        model.load_state_dict(torch.load(self.settings.model_path, map_location=self.tdevice), strict=False)
        model = model.to(self.tdevice)
        self.model = nn.DataParallel(model)

    def _initilization_(self):        
        if self.settings.calc_smpl:
            self.smpl_parser = SMPLA_parser(self.settings.smpl_path, self.settings.smil_path).to(self.tdevice)
        
        if self.settings.temporal_optimize:
            self._initialize_optimization_tools_(self.settings.smooth_coeff)

        if self.settings.render_mesh or self.settings.mode == 'webcam':
            self.renderer = setup_renderer(name=self.settings.renderer)
        self.visualize_items = self.settings.show_items.split(',')
        self.result_keys = ['smpl_thetas', 'smpl_betas', 'cam','cam_trans', 'params_pred', 'center_confs', 'pred_batch_ids']
    
    def _initialize_optimization_tools_(self, smooth_coeff):
        self.OE_filters = {}
        if not self.settings.show_largest:
            from tracker.byte_tracker_3dcenter import Tracker
            self.tracker = Tracker(det_thresh=0.12, low_conf_det_thresh=0.05, track_buffer=60, match_thresh=300, frame_rate=30)

    def single_image_forward(self, image):
        input_image, image_pad_info = img_preprocess(image)
        parsed_results = self.model(input_image.to(self.tdevice))
        if parsed_results is None:
            return None, image_pad_info
        parsed_results.update(pack_params_dict(parsed_results['params_pred']))
        parsed_results.update({'cam_trans':denormalize_cam_params_to_trans(parsed_results['cam'])})

        all_result_keys = list(parsed_results.keys())
        for key in all_result_keys:
            if key not in self.result_keys:
                del parsed_results[key]
        return parsed_results, image_pad_info
        
    @time_cost('BEV')
    @torch.no_grad()
    def forward(self, image, signal_ID=0, **kwargs):
        if image.shape[1] / image.shape[0] >= 2:
            outputs = self.process_long_image(image, show_patch_results=self.settings.show_patch_results)
        else:
            outputs = self.process_normal_image(image, signal_ID)
        if outputs is None:
            return None

        if self.settings.render_mesh:
            mesh_color_type = 'identity' if self.settings.mode!='webcam' and not self.settings.save_video else 'same'
            rendering_cfgs = {'mesh_color':mesh_color_type, 'items': self.visualize_items, 'renderer': self.settings.renderer}
            outputs = rendering_romp_bev_results(self.renderer, outputs, image, rendering_cfgs)
        if self.settings.show:
            h, w = outputs['rendered_image'].shape[:2]
            show_image = outputs['rendered_image'] if h<=1080 else cv2.resize(outputs['rendered_image'], (int(w*(1080/h)), 1080))
            cv2.imshow('rendered', show_image)
            wait_func(self.settings.mode)
        return convert_tensor2numpy(outputs)
        
    def process_normal_image(self, image, signal_ID):
        outputs, image_pad_info = self.single_image_forward(image)
        meta_data = {'input2org_offsets': image_pad_info}
        
        if outputs is None:
            return None
        
        if self.settings.temporal_optimize:
            outputs = self.temporal_optimization(outputs, signal_ID)
            if outputs is None:
                return None
            outputs.update({'cam_trans':denormalize_cam_params_to_trans(outputs['cam'])})
        
        if self.settings.calc_smpl:
            verts, joints, face = self.smpl_parser(outputs['smpl_betas'], outputs['smpl_thetas']) 
            outputs.update({'verts': verts, 'joints': joints, 'smpl_face':face})
            if self.settings.render_mesh:
                meta_data['vertices'] = outputs['verts']
            projection = body_mesh_projection2image(outputs['joints'], outputs['cam'], **meta_data)
            outputs.update(projection)
            
            outputs = suppressing_redundant_prediction_via_projection(outputs,image.shape, thresh=self.settings.nms_thresh)
            outputs = remove_outlier(outputs,relative_scale_thresh=self.settings.relative_scale_thresh)
        return outputs
    
    #@time_cost('BEV')
    def process_long_image(self, full_image, show_patch_results=False):
        print('processing in crowd mode')
        from .split2process import get_image_split_plan, convert_crop_cam_params2full_image,\
            collect_outputs, exclude_boudary_subjects, padding_image_overlap
        full_image_pad, image_pad_info, pad_length = padding_image_overlap(full_image, overlap_ratio=self.settings.overlap_ratio)
        meta_data = {'input2org_offsets': image_pad_info}
        
        fh, fw = full_image_pad.shape[:2]
        # please crop the human area out from the huge/long image to facilitate better predictions.
        crop_boxes = get_image_split_plan(full_image_pad,overlap_ratio=self.settings.overlap_ratio)

        croped_images, outputs_list = [], []
        for cid, crop_box in enumerate(crop_boxes):
            l,r,t,b = crop_box
            croped_image = full_image_pad[t:b, l:r]
            crop_outputs, image_pad_info = self.single_image_forward(croped_image)
            if crop_outputs is None:
                outputs_list.append(crop_outputs)
                continue
            verts, joints, face = self.smpl_parser(crop_outputs['smpl_betas'], crop_outputs['smpl_thetas']) 
            crop_outputs.update({'verts': verts, 'joints': joints, 'smpl_face':face})
            outputs_list.append(crop_outputs)
            croped_images.append(croped_image)
        
        # exclude the subjects in the overlapping area, the right of this crop
        for cid in range(len(crop_boxes)):
            this_outs = outputs_list[cid]
            if this_outs is not None:
                if cid != len(crop_boxes) - 1:
                    this_right, next_left = crop_boxes[cid, 1], crop_boxes[cid+1, 0]
                    drop_boundary_ratio = (this_right - next_left) / fh / 2
                    exclude_boudary_subjects(this_outs, drop_boundary_ratio, ptype='left', torlerance=0)
                ch, cw = croped_images[cid].shape[:2]
                projection = body_mesh_projection2image(this_outs['joints'], this_outs['cam'], vertices=this_outs['verts'], input2org_offsets=torch.Tensor([0, ch, 0, cw, ch, cw]))
                this_outs.update(projection)
                
        # exclude the subjects in the overlapping area, the left of next crop
        for cid in range(1,len(crop_boxes)-1):
            this_outs, next_outs = outputs_list[cid], outputs_list[cid+1]
            this_right, next_left = crop_boxes[cid, 1], crop_boxes[cid+1, 0]
            drop_boundary_ratio = (this_right - next_left) / fh / 2 
            if next_outs is not None:
                exclude_boudary_subjects(next_outs, drop_boundary_ratio, ptype='right', torlerance=0) 
        
        for cid, crop_image in enumerate(croped_images):
            this_outs = outputs_list[cid]
            ch, cw = croped_images[cid].shape[:2]
            this_outs = suppressing_redundant_prediction_via_projection(this_outs, [ch, cw], thresh=self.settings.nms_thresh,conf_based=True)
            this_outs = remove_outlier(this_outs, scale_thresh=1, relative_scale_thresh=self.settings.relative_scale_thresh)
        
        if show_patch_results:
            rendering_cfgs = {'mesh_color':'identity', 'items':['mesh','center_conf','pj2d'], 'renderer':self.settings.renderer}
            for cid, crop_image in enumerate(croped_images):
                this_outs = outputs_list[cid]
                this_outs = rendering_romp_bev_results(self.renderer, this_outs, crop_image, rendering_cfgs)
                saver = ResultSaver(self.settings.mode, self.settings.save_path)
                saver(this_outs, 'crop.jpg', prefix=f'{self.settings.center_thresh}_{cid}')     
        
        outputs = {}
        for cid, crop_box in enumerate(crop_boxes):
            crop_outputs = outputs_list[cid]
            if crop_outputs is None:
                continue
            crop_box[:2] -= pad_length
            crop_outputs['cam'] = convert_crop_cam_params2full_image(crop_outputs['cam'], crop_box, full_image.shape[:2])
            collect_outputs(crop_outputs, outputs)
        
        if self.settings.render_mesh:
            meta_data['vertices'] = outputs['verts']
        projection = body_mesh_projection2image(outputs['joints'], outputs['cam'], **meta_data)
        outputs.update(projection)
        outputs = suppressing_redundant_prediction_via_projection(outputs, full_image.shape, thresh=self.settings.nms_thresh,conf_based=True)
        outputs = remove_outlier(outputs, scale_thresh=0.5, relative_scale_thresh=self.settings.relative_scale_thresh)

        return outputs
    
    def temporal_optimization(self, outputs, signal_ID, image_scale=128, depth_scale=30):
        check_filter_state(self.OE_filters, signal_ID, self.settings.show_largest, self.settings.smooth_coeff)
        if self.settings.show_largest:
            max_id = torch.argmax(outputs['cam'][:,0])
            outputs['smpl_thetas'], outputs['smpl_betas'], outputs['cam'] = \
                smooth_results(self.OE_filters[signal_ID], \
                    outputs['smpl_thetas'][max_id], outputs['smpl_betas'][max_id], outputs['cam'][max_id])
        else:
            cam_trans = outputs['cam_trans'].cpu().numpy()
            cams = outputs['cam'].cpu().numpy()
            det_confs = outputs['center_confs'].cpu().numpy()
            tracking_points = np.concatenate([(cams[:,[2,1]]+1)*image_scale, cam_trans[:,[2]]*depth_scale, cams[:,[0]]*image_scale/2],1)
            tracked_ids, results_inds = self.tracker.update(tracking_points, det_confs)
            if len(tracked_ids) == 0:
                return None

            for key in self.result_keys:
                outputs[key] = outputs[key][results_inds]

            for ind, tid in enumerate(tracked_ids):
                if tid not in self.OE_filters[signal_ID]:
                    self.OE_filters[signal_ID][tid] = create_OneEuroFilter(self.settings.smooth_coeff)
                outputs['smpl_thetas'][ind], outputs['smpl_betas'][ind], outputs['cam'][ind] = \
                    smooth_results(self.OE_filters[signal_ID][tid], \
                    outputs['smpl_thetas'][ind], outputs['smpl_betas'][ind], outputs['cam'][ind])
            outputs['track_ids'] = np.array(tracked_ids).astype(np.int32)
        return outputs

def main():
    args = bev_settings()
    bev = BEV(args)
    if args.mode == 'image':
        saver = ResultSaver(args.mode, args.save_path)
        image = cv2.imread(args.input)
        outputs = bev(image)
        saver(outputs, args.input, prefix=f'{args.center_thresh}')
    
    if args.mode == 'video':
        frame_paths, video_save_path = collect_frame_path(args.input, args.save_path)
        saver = ResultSaver(args.mode, args.save_path)
        for frame_path in progress_bar(frame_paths):
            image = cv2.imread(frame_path)
            outputs = bev(image)
            saver(outputs, frame_path, prefix=f'_{model_id}_{args.center_thresh}')
        save_video_results(saver.frame_save_paths)
        if args.save_video:
            saver.save_video(video_save_path, frame_rate=args.frame_rate)

    if args.mode == 'webcam':
        cap = WebcamVideoStream(0)
        cap.start()
        while True:
            frame = cap.read()
            outputs = bev(frame)
            if cv2.waitKey(1) == 27:
                break 
        cap.stop()

if __name__ == '__main__':
    main()
    
    
