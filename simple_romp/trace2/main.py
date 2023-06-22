import cv2
import torch
from torch import nn
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import time

from .track import collect_sequence_tracking_results

from .utils.infer_settings import update_seq_cfgs, get_seq_cfgs, trace_settings
from .models.raft.process import FlowExtract, show_seq_flow
from .models.hrnet_32 import HigherResolutionNet
from .models.model import TRACE_head
from .utils.utils import reorganize_items, load_model, load_config_dict, preds_save_paths, print_dict, download_model
from .utils.load_data import prepare_data_loader, prepare_video_frame_dict, extract_seq_data
from .utils.visualize_results import visualize_predictions

from .utils.infer_utils import remove_large_keys, collect_kp_results, insert_last_human_state, save_last_human_state, merge_output
from .results_parser.temp_result_parser import TempResultParser

# default_settings = trace_settings(input_args=[])

class TRACE(nn.Module):
    def __init__(self, args):
        super(TRACE, self).__init__()
        load_config_dict(self, vars(args)) # loading settings to self.
        self.default_seq_cfgs = get_seq_cfgs(args)
        self.device = torch.device(f'cuda:{self.GPU}') if self.GPU>-1 else torch.device('cpu')
        print('putting data onto:', self.device)
        self.__load_models__()  
      
        self.video_eval_cfg = {'mode':'parsing', 'sequence_input':True, 'is_training':False, 'update_data': True, 'calc_loss':False, \
                            'input_type': 'sequence', 'with_nms': True, 'with_2d_matching': True, 'new_training': False, \
                            'regress_params': True, 'traj_conf_threshold': 0.12, 'temp_clip_length_eval':8, 'xs':2, 'ys':2}
        self.continuous_state_cacher = {'image':{}, 'image_feats':{}, 'temp_state':{}}
        self.track_id_start = 0
    
    def __load_models__(self):
        image_backbone = HigherResolutionNet()
        image_backbone = load_model(self.image_backbone_model_path, image_backbone, prefix='module.backbone.', drop_prefix='', fix_loaded=True)
        self.image_backbone = nn.DataParallel(image_backbone.cuda()).eval()
        self.motion_backbone = FlowExtract(self.raft_model_path, self.device)
        
        self._result_parser = TempResultParser(self.smpl_path, self.center_thresh)
        temporal_head_model = TRACE_head(self._result_parser, temp_clip_length=self.temp_clip_length, smpl_model_path=self.smpl_path)
        temporal_head_model = load_model(self.trace_head_model_path, temporal_head_model, prefix='', drop_prefix='', fix_loaded=False) #module.
        self.temporal_head_model = nn.DataParallel(temporal_head_model.to(self.device))
    
    def temp_head_forward(self, feat_inputs, meta_data, seq_name, **cfg):
        temp_states = self.continuous_state_cacher['temp_state'][seq_name] if seq_name in self.continuous_state_cacher['temp_state'] else [None] * 5 
        # TODO: how to associate the track ids to connect the init_world_cams and init_world_grots
        outputs, temp_states = self.temporal_head_model({'image_feature_maps': feat_inputs['image_feature_maps'], 'optical_flows':feat_inputs['optical_flows']}, temp_states=temp_states, \
            temp_clip_length = cfg['temp_clip_length_eval'], track_id_start=self.track_id_start, seq_cfgs = cfg['seq_cfgs'], xs = cfg['xs'], ys = cfg['ys'])
        self.continuous_state_cacher['temp_state'][seq_name] = temp_states
        if outputs is not None:
            outputs, meta_data = self._result_parser.parsing_forward(outputs, meta_data, cfg)
        #BaseTrack._count = 0 # to refresh the ID back to start from 1
        return outputs
        
    @torch.no_grad()
    def sequence_inference(self, meta_data, seq_name, cfg_dict):
        input_images = meta_data['image']
        sequence_length = len(input_images)

        image_feature_maps = self.image_backbone(input_images)
        image_feature_maps = insert_last_human_state(image_feature_maps, self.continuous_state_cacher['image_feats'], seq_name)
        padded_input_images = insert_last_human_state(input_images, self.continuous_state_cacher['image'], seq_name)

        target_img_inds = torch.arange(1, len(padded_input_images))
        source_img_inds = target_img_inds - 1
        temp_inputs = {'image_feature_maps': image_feature_maps}
        temp_inputs['optical_flows'] = self.motion_backbone(padded_input_images, source_img_inds, target_img_inds)
        temp_meta_data = {'batch_ids': torch.arange(sequence_length).cuda(), 'offsets':meta_data['offsets'].cuda()}

        outputs = self.temp_head_forward(temp_inputs, temp_meta_data, seq_name, **cfg_dict)
        self.continuous_state_cacher['image'] = save_last_human_state(self.continuous_state_cacher['image'], input_images[[-1]], seq_name)
        self.continuous_state_cacher['image_feats'] = save_last_human_state(self.continuous_state_cacher['image_feats'], image_feature_maps[[-1]], seq_name)
        
        if outputs is None:
            return None, meta_data, None, None

        # aligning image path to each prediction. 
        used_imgpath = reorganize_items([meta_data['imgpath']], outputs['reorganize_idx'].cpu().numpy())[0]
        tracking_results = collect_sequence_tracking_results(outputs, used_imgpath, outputs['reorganize_idx'].cpu().numpy(), show=self.show_tracking)
        kp3d_results = collect_kp_results(outputs, used_imgpath)

        return outputs, meta_data, tracking_results, kp3d_results

    def update_sequence_cfs(self, seq_name):
        return update_seq_cfgs(seq_name, self.default_seq_cfgs)

    @torch.no_grad()
    def forward(self, sequence_dict):
        """
            Please input one sequence per time
        """
        data_loader = prepare_data_loader(sequence_dict, self.val_batch_size)
        seq_outputs, tracking_results, kp3d_results, imgpaths = {}, {}, {}, {}
        start_frame_id = 0
        start_time = time.time()
        print(f'Processing {np.unique(list(sequence_dict.keys()))}')
        for meta_data in data_loader:
            seq_data, seq_name = extract_seq_data(meta_data)
            start_frame_id += len(seq_data['image'])
            if seq_name not in imgpaths:
                imgpaths[seq_name] = []
            imgpaths[seq_name] += seq_data['imgpath']
            sfi = start_frame_id - len(seq_data['image'])

            self.video_eval_cfg['seq_cfgs'] = self.update_sequence_cfs(seq_name)
            outputs, meta_data, seq_tracking_results, seq_kp3d_results = self.sequence_inference(seq_data, seq_name, self.video_eval_cfg)
            if outputs is None:
                print('sequence', seq_name, 'has no detections at frame', sfi)
                continue
            outputs['reorganize_idx'] += sfi
            if seq_name not in seq_outputs:
                seq_outputs[seq_name], tracking_results[seq_name], kp3d_results[seq_name] = {}, {}, {}
            
            seq_outputs[seq_name] = merge_output(outputs, seq_outputs[seq_name])
            tracking_results[seq_name].update(seq_tracking_results)
            kp3d_results[seq_name].update(seq_kp3d_results)
            
        print('FPS:', start_frame_id/(time.time()-start_time))
        
        return seq_outputs, tracking_results, kp3d_results, imgpaths

    def save_results(self, outputs, tracking_results, kp3d_results, imgpaths):
        for seq_name in outputs:
            save_paths = preds_save_paths(self.results_save_dir, prefix=seq_name)
            np.savez(save_paths.seq_results_save_path, outputs=remove_large_keys(outputs[seq_name]), imgpaths=imgpaths[seq_name])
            np.savez(save_paths.seq_tracking_results_save_path, tracking=tracking_results[seq_name], kp3ds=kp3d_results[seq_name])
            if self.save_video:
                visualize_predictions(outputs[seq_name], imgpaths[seq_name], self.FOV, save_paths.seq_save_dir, self.smpl_model_path)

def main():
    args = trace_settings()
    trace = TRACE(args)
    if args.mode == 'video':
        sequence_dict = prepare_video_frame_dict([args.input], img_ext='jpg', frame_dir=args.save_path)
        outputs, tracking_results, kp3d_results, imgpaths = trace(sequence_dict)
        trace.save_results(outputs, tracking_results, kp3d_results, imgpaths)

if __name__ == '__main__':
    main()
    