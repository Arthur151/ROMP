import yaml
import sys, os, cv2
import numpy as np
import time, datetime
import copy, random, itertools
from prettytable import PrettyTable

import torch
import torch.nn as nn
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset, DataLoader, ConcatDataset

sys.path.append(os.path.abspath(__file__).replace('core/base.py',''))
import config
import constants
from config import args
from models import *
from utils import *
from maps_utils import CenterMap
from dataset.mixed_dataset import SingleDataset
from visualization.visualization import Visualizer

if args.model_precision=='fp16':
    from torch.cuda.amp import autocast, GradScaler

import platform
if 'Ubuntu' not in platform.version():
    os.environ['DISPLAY'] = ':0.0'

class Base(object):
    def __init__(self):
        hparams_dict = self.load_config_dict(vars(args))
        self.model_type = 'smpl'
        print('model_type', self.model_type)     
        self.project_dir = config.project_dir
        self._init_params()

    def _build_model(self):
        print('start building model.')
        generator = get_pose_net(params_num = self.params_num)
        if '-1' not in self.gpu:
            generator = generator.cuda()
        generator = self.load_model(self.gmodel_path,generator)
        self.generator = nn.DataParallel(generator)
        self.centermap_parser = CenterMap()
        print('finished build model.')

    def set_up_smplx(self):
        rot_dim = 6 if self.Rot_type=='6D' else 3
        cam_dim = 3
        joint_mapper = JointMapper(smpl_to_openpose(model_type=self.model_type, use_hands=True, use_face=True, use_foot=True, \
                     use_face_contour=False, openpose_format='coco25'))
        self.joint_mapper_SPIN24 = constants.joint_mapping({**constants.SMPL_24,**constants.SMPL_EXTRA_30}, constants.SPIN_24)
        self.smplx = smpl_model.create(args.smpl_model_path, J_reg_extra_path=args.smpl_J_reg_extra_path, batch_size=self.batch_size,model_type=self.model_type, gender='neutral', \
            use_face_contour=False, ext='npz', joint_mapper=joint_mapper,flat_hand_mean=True, use_pca=False)
        if '-1' not in self.gpu:
            self.smplx = self.smplx.cuda()
        self.part_name = ['cam', 'global_orient', 'body_pose', 'betas']
        self.part_idx = [cam_dim, rot_dim,  21*rot_dim,       10]
        
        self.kps_num = 25 # + 21*2
        self.params_num = np.array(self.part_idx).sum()
        self.global_orient_nocam = torch.from_numpy(constants.global_orient_nocam).unsqueeze(0)

    def _calc_smplx_params(self, param):
        idx_list = [0]
        params_dict = {}
        # cam:4; poses: 87=3+63+6+6+3+3+3; expres: 10; shape: 10 = 111
        for i,  (idx, name) in enumerate(zip(self.part_idx,self.part_name)):
            idx_list.append(idx_list[i] + idx)
            params_dict[name] = param[:, idx_list[i]: idx_list[i+1]].contiguous()

        if self.Rot_type=='6D':
            params_dict['body_pose'] = rot6D_to_angular(params_dict['body_pose'])
            params_dict['global_orient'] = rot6D_to_angular(params_dict['global_orient'])
        N = params_dict['body_pose'].shape[0]
        params_dict['body_pose'] = torch.cat([params_dict['body_pose'], torch.zeros(N,6).to(params_dict['body_pose'].device)],1)        
        
        output = self.smplx(**params_dict, return_verts=True, return_full_pose=True)
        vertices, full_pose = output.vertices, output.full_pose #10475

        j3d_smpl24 = output.joints_org[:,:24].clone()
        j3d_spin24 = output.joints_org.clone()[:,self.joint_mapper_SPIN24]
        j3d_op25 = output.joints.clone()
        j3d_op25[:,constants.OpenPose_25['Pelvis']] = j3d_op25[:,self.lr_hip_idx_op25].mean(1)
        if self.kp3d_format=='smpl24':
            j3d = j3d_smpl24.clone()
            j3d[:,constants.SMPL_24['Pelvis']] = j3d[:,self.lr_hip_idx_smpl24].mean(1)
        elif self.kp3d_format =='coco25':
            j3d = j3d_op25.clone()

        pj3d = proj.batch_orth_proj(j3d_op25, params_dict['cam'], mode='2d')
        verts_camed = proj.batch_orth_proj(vertices, params_dict['cam'], mode='3d',keep_dim=True)

        outputs = {'params': params_dict, 'verts': vertices, 'pj2d':pj3d[:,:,:2], 'j3d':j3d, 'j3d_smpl24':j3d_smpl24, 'j3d_spin24':j3d_spin24, 'j3d_op25':j3d_op25, 'verts_camed': verts_camed, 'poses':full_pose}
        return outputs

    def net_forward(self, data_3d, model, imgs=None,match_to_gt=False,mode='test'):
        if imgs is None:
            imgs = data_3d['image']
        params, center_maps, heatmap_AEs = model(imgs.contiguous())
        
        params, kps, data_3d, reorganize_idx, success_flag = self.parse_maps(params, center_maps, heatmap_AEs, data_3d)
        if params is not None and success_flag:
            outputs = self._calc_smplx_params(params.contiguous())
            outputs['success_flag'] = True
        else:
            outputs = {'success_flag':False}
        return outputs, center_maps, kps, data_3d, reorganize_idx

    def parse_maps(self,param_maps, center_maps, heatmap_AEs, data_3d=None):
        kps = heatmap_AEs
        centers_pred= []
        for batch_id in range(len(param_maps)):
            center_ids, center_conf = self.centermap_parser.parse_centermap(center_maps[batch_id])
            if len(center_ids)>0:
                center_whs_pred = center_ids.cpu().float()
                center_conf = center_conf.detach().cpu().float().numpy()
                center_filtered = center_whs_pred#self.kp2d_filter(center_whs_pred[center_conf>center_thresh], kps[batch_id] )
                centers_pred.append(center_filtered)
            else:
                centers_pred.append([])

        params_pred, reorganize_idx = [[] for i in range(2)]
        if data_3d is not None:
            info_vis = ['imgpath', 'image_org', 'offsets']
            matched_data = {}
            for key in info_vis:
                matched_data[key] = []

        # while training, use gt center to extract the parameters from the estimated map
        # while evaluation, match the estimated center with the clostest gt center for parameter sampling.
        for batch_id, param_map in enumerate(param_maps):
            centers = centers_pred[batch_id]
            for person_id, center in enumerate(centers):
                center_w, center_h = center.long()
                params_pred.append(param_map[:,center_w,center_h])
                reorganize_idx.append(batch_id)

                if data_3d is not None:
                    for key in matched_data:
                        data_gt = data_3d[key]
                        if isinstance(data_gt, torch.Tensor):
                            matched_data[key].append(data_gt[batch_id])
                        elif isinstance(data_gt, list):
                            matched_data[key].append(data_gt[batch_id])
                
        if len(params_pred)>0:
            params = torch.stack(params_pred)
        else:
            params = None
        
        success_flag=True
        if data_3d is not None:
            for key in matched_data:
                data_gt = data_3d[key]
                if isinstance(data_gt, torch.Tensor):
                    if len(matched_data[key])<1:
                        success_flag=False
                        continue
                    data_3d[key] = torch.stack(matched_data[key])
                elif isinstance(data_gt, list):
                    data_3d[key] = np.array(matched_data[key])
        
        return params, kps, data_3d, np.array(reorganize_idx), success_flag

    def _init_params(self):
        self.global_count = 0
        self.lr_hip_idx_op25 = np.array([constants.OpenPose_25['L_Hip'], constants.OpenPose_25['R_Hip']])
        self.lr_hip_idx_smpl24 = np.array([constants.SMPL_24['L_Hip'], constants.SMPL_24['R_Hip']])
        self.lr_hip_idx = self.lr_hip_idx_smpl24 if self.kp3d_format =='smpl24' else self.lr_hip_idx_op25
        self.kintree_parents = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16,17, 18, 19, 20, 21],dtype=np.int)

    def _create_single_data_loader(self, **kwargs):
        print('gathering datasets')
        datasets = SingleDataset(**kwargs)
        return DataLoader(dataset = datasets,\
            batch_size = self.batch_size if kwargs['train_flag'] else self.val_batch_size,\
            shuffle = True if kwargs['train_flag'] else False,drop_last = False, pin_memory = True,num_workers = self.nw)

    def load_config_dict(self, config_dict):
        hparams_dict = {}
        for i, j in config_dict.items():
            setattr(self,i,j)
            hparams_dict[i] = j
        return hparams_dict

    def load_model(self, path, model,prefix = 'module.',optimizer=None):
        print('*'*20)
        print('using fine_tune model: ', path)
        if os.path.exists(path):
            if '-1' in self.gpu:
                pretrained_model = torch.load(path,map_location=torch.device('cpu'))
            else:
                pretrained_model = torch.load(path)
            current_model = model.state_dict()
            if isinstance(pretrained_model, dict):
                if 'model_state_dict' in pretrained_model:
                    pretrained_model = pretrained_model['model_state_dict']
                if 'optimizer_state_dict' in pretrained_model and optimizer is not None:
                    optimizer.load_state_dict(pretrained_model['optimizer_state_dict'])
                #self.copy_state_dict_fp16(current_model, pretrained_model, prefix = prefix)
            copy_state_dict(current_model, pretrained_model, prefix = prefix)
        else:
            print('model {} not exist!'.format(path))
        print('*'*20)
        return model