from .model import ROMPv1
import cv2
import numpy as np
import os, sys
import os.path as osp
import torch
from torch import nn
import argparse

from .post_parser import SMPL_parser, body_mesh_projection2image, parsing_outputs
from .utils import img_preprocess, create_OneEuroFilter, euclidean_distance, \
    time_cost, download_model, determine_device, ResultSaver, WebcamVideoStream, \
    wait_func, collect_frame_path, progress_bar, get_tracked_ids
from romp_visualizer import Sim3DR
from .post_parser import CenterMap

def romp_settings():
    parser = argparse.ArgumentParser(description = 'ROMP: Monocular, One-stage, Regression of Multiple 3D People')
    parser.add_argument('-m', '--mode', type=str, default='image', help = 'Inferece mode, including image, video, webcam')
    parser.add_argument('-i', '--input', type=str, default=None, help = 'Path to the input image / video')
    parser.add_argument('-o', '--save_path', type=str, default='romp_results', help = 'Path to save the results')
    parser.add_argument('--GPU', type=int, default=0, help = 'The gpu device number to run the inference on. If GPU=-1, then running in cpu mode')

    parser.add_argument('-t', '--temporal_optimize', action='store_true', help = 'Whether to use OneEuro filter to smooth the results')
    parser.add_argument('--center_thresh', type=float, default=0.25, help = 'The confidence threshold of positive detection in 2D human body center heatmap.')
    parser.add_argument('--show_largest', action='store_true', help = 'Whether to show the largest person only')
    parser.add_argument('-sc','--smooth_coeff', type=float, default=3., help = 'The smoothness coeff of OneEuro filter, the smaller, the smoother.')
    parser.add_argument('--calc_smpl', action='store_true', help = 'Whether to calculate the smpl mesh from estimated SMPL parameters')
    parser.add_argument('--render_mesh', action='store_true', help = 'Whether to render the estimated 3D mesh mesh to image')
    parser.add_argument('--show', action='store_true', help = 'Whether to show the rendered results')
    parser.add_argument('--save_video', action='store_true', help = 'Whether to save the video results')
    parser.add_argument('--frame_rate', type=int, default=24, help = 'The frame_rate of saved video results')
    parser.add_argument('--smpl_path', type=str, default=osp.join(osp.expanduser("~"),'.romp','smpl_packed_info.pth'), help = 'The path of smpl model file')
    parser.add_argument('--model_path', type=str, default=osp.join(osp.expanduser("~"),'.romp','ROMP.pkl'), help = 'The path of ROMP checkpoint')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        args.GPU = -1
        args.temporal_optimize = False
    if args.show:
        args.render_mesh = True
    if args.render_mesh or args.show_largest:
        args.calc_smpl = True
    if not os.path.exists(args.smpl_path):
        smpl_url = 'https://github.com/Arthur151/ROMP/releases/download/S1/smpl_packed_info.pth'
        download_model(smpl_url, args.smpl_path, 'SMPL')
    if not os.path.exists(args.model_path):
        romp_url = 'https://github.com/Arthur151/ROMP/releases/download/S1/ROMP.pkl'
        download_model(romp_url, args.model_path, 'ROMP')
    return args

class ROMP(nn.Module):
    def __init__(self, romp_settings):
        super(ROMP, self).__init__()
        self.settings = romp_settings
        self.tdevice = determine_device(self.settings.GPU)
        self._build_model_()
        self._initilization_()
    
    def _build_model_(self):
        model = ROMPv1().eval()
        model.load_state_dict(torch.load(self.settings.model_path, map_location=self.tdevice))
        model = model.to(self.tdevice)
            
        self.model = nn.DataParallel(model)
    
    def _initilization_(self):
        self.centermap_parser = CenterMap(conf_thresh=self.settings.center_thresh)
        
        if self.settings.calc_smpl:
            self.smpl_parser = SMPL_parser(self.settings.smpl_path).to(self.tdevice)
        
        if self.settings.temporal_optimize:
            self._initialize_optimization_tools_(self.settings.smooth_coeff)

        if self.settings.render_mesh:
            self.renderer = Sim3DR(intensity_ambient=0.9)

    def single_image_forward(self, image):
        input_image, image_pad_info = img_preprocess(image)
        center_maps, params_maps  = self.model(input_image.to(self.tdevice))
        parsed_results = parsing_outputs(center_maps, params_maps, self.centermap_parser)
        return parsed_results, image_pad_info
    
    def _initialize_optimization_tools_(self, smooth_coeff):
        self.OE_filters = {}
        if not self.settings.show_largest:
            try:
                from norfair import Tracker
            except:
                print('To perform temporal optimization, installing norfair for tracking.')
                os.system('pip install norfair')
                from norfair import Tracker
            self.tracker = Tracker(distance_function=euclidean_distance, distance_threshold=80)
            self.tracker_initialized = False
        else:
            self.OE_filters = create_OneEuroFilter(smooth_coeff)
    
    def temporal_optimization(self, outputs):
        if self.settings.show_largest:
            max_id = torch.argmax(outputs['params']['cam'][:,0])
            pred_pose = outputs['params']['poses'][max_id]
            pred_beta = outputs['params']['betas'][max_id]
            pred_cam = outputs['params']['cam'][max_id]
            pred_pose[3:] = self.OE_filters['pose'].process(pred_pose[3:])
            outputs['params']['poses'] = pred_pose[None]
            outputs['params']['betas'] = self.OE_filters['betas'].process(pred_beta)[None]
            outputs['params']['cam'] = self.OE_filters['cam'].process(pred_cam)[None]
        else:
            pred_cams = outputs['params']['cam']
            from norfair import Detection
            detections = [Detection(points=cam[[2,1]]*512) for cam in pred_cams.cpu().numpy()]
            if not self.tracker_initialized:
                for _ in range(8):
                    tracked_objects = self.tracker.update(detections=detections)
            tracked_objects = self.tracker.update(detections=detections)
            if len(tracked_objects)==0:
                return outputs
            tracked_ids = get_tracked_ids(detections, tracked_objects)
            for sid, tid in enumerate(tracked_ids):
                if tid not in self.OE_filters:
                    self.OE_filters[tid] = create_OneEuroFilter(self.settings.smooth_coeff)
                pred_pose = outputs['params']['poses'][sid]
                pred_beta = outputs['params']['betas'][sid]
                pred_cam = outputs['params']['cam'][sid]
                pred_pose[3:] = self.OE_filters[tid]['pose'].process(pred_pose[3:])
                outputs['params']['poses'][sid] = pred_pose
                outputs['params']['betas'][sid] = self.OE_filters[tid]['betas'].process(pred_beta)
                outputs['params']['cam'][sid] = self.OE_filters[tid]['cam'].process(pred_cam)
            outputs['tracked_ids'] = tracked_ids
        return outputs

    def rendering_results(self, outputs, image, image_pad_info, alpha=0.8):
        projection = body_mesh_projection2image(outputs['joints'], outputs['params']['cam'], vertices=outputs['verts'], input2org_offsets=image_pad_info)
        vertices = projection['verts_camed_org'].cpu().numpy()
        triangles = outputs['smpl_face'].cpu().numpy().astype(np.int32)
        rendered_image = self.renderer.render_verts_list(vertices, triangles, image, color=np.array([[0.65, 0.74, 0.86]]))
        outputs['rendered_image'] = cv2.addWeighted(image, 1 - alpha, rendered_image, alpha, 0)
        if self.settings.show:
            cv2.imshow('rendered', outputs['rendered_image'])
            wait_func(self.settings.mode)

    @time_cost('ROMP')
    def forward(self, image, **kwargs):
        outputs, image_pad_info = self.single_image_forward(image)
        if outputs is None:
            return None
        if self.settings.temporal_optimize:
            outputs = self.temporal_optimization(outputs)
        if self.settings.calc_smpl:
            outputs = self.smpl_parser(outputs) 
        if self.settings.render_mesh:
            self.rendering_results(outputs, image, image_pad_info)
        return outputs

def main():
    args = romp_settings()
    romp = ROMP(args)
    if args.mode == 'image':
        saver = ResultSaver(args.mode, args.save_path)
        image = cv2.imread(args.input)
        outputs = romp(image)
        saver(outputs, args.input)
    
    if args.mode == 'video':
        frame_paths, video_save_path = collect_frame_path(args.input, args.save_path)
        saver = ResultSaver(args.mode, args.save_path)
        for frame_path in progress_bar(frame_paths):
            image = cv2.imread(frame_path)
            outputs = romp(image)
            saver(outputs, frame_path)
        if args.save_video:
            saver.save_video(video_save_path, frame_rate=args.frame_rate)

    if args.mode == 'webcam':
        cap = WebcamVideoStream(0)
        cap.start()
        try:
            while True:
                frame = cap.read()
                outputs = romp(frame)
        except:
            cap.stop()

if __name__ == '__main__':
    main()
    
    
