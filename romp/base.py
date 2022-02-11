import sys, os, cv2
import numpy as np
import time, datetime
import logging
import copy, random, itertools
from prettytable import PrettyTable
import pickle
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import config
import constants
from config import args, parse_args, ConfigContext
from models import build_model,build_teacher_model
from models.balanced_dataparallel import DataParallel
from utils import *
from utils.projection import vertices_kp3d_projection
from utils.train_utils import justify_detection_state
from evaluation import compute_error_verts, compute_similarity_transform, compute_similarity_transform_torch, \
                    batch_compute_similarity_transform_torch, compute_mpjpe, \
                    determ_worst_best, reorganize_vis_info
from dataset.mixed_dataset import MixedDataset,SingleDataset
from visualization.visualization import Visualizer
if args().model_precision=='fp16':
    from torch.cuda.amp import autocast, GradScaler

class Base(object):
    def __init__(self, args_set=None):
        self.project_dir = config.project_dir
        hparams_dict = self.load_config_dict(vars(args() if args_set is None else args_set))
        self._init_log(hparams_dict)
        self._init_params()
        if self.save_visualization_on_img:
            logging.info('Loading {} renderer as visualizer'.format(self.renderer))
            self.visualizer = Visualizer(resolution=(512,512), result_img_dir=self.result_img_dir, renderer_type=self.renderer)

    def _build_model_(self):
        logging.info('start building model.')
        model = build_model()
        if self.fine_tune or self.eval:
            drop_prefix = ''
            if self.model_version==6:
                model = load_model(self.model_path, model, prefix = 'module.', drop_prefix=drop_prefix, fix_loaded=True)
            else:
                model = load_model(self.model_path, model, prefix = 'module.', drop_prefix=drop_prefix, fix_loaded=False) 
                train_entire_model(model)
        if self.distributed_training:
            print('local_rank', self.local_rank)
            device = torch.device('cuda', self.local_rank)
            torch.cuda.set_device(self.local_rank)
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            torch.distributed.init_process_group(backend='nccl')
            assert torch.distributed.is_initialized()
            self.model = nn.parallel.DistributedDataParallel(model.to(device), device_ids=[self.local_rank],output_device=self.local_rank,find_unused_parameters=True).cuda()
        else:
            if self.master_batch_size!=-1:
                # balance the multi-GPU memory via adjusting the batch size of each GPU.
                self.model = DataParallel(model.cuda(), device_ids=self.GPUS, chunk_sizes=self.chunk_sizes)
            else:
                self.model = nn.DataParallel(model.cuda())

    def _build_optimizer(self):
        if self.optimizer_type=='Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr)
        elif self.optimizer_type=='SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.lr, momentum=0.9, weight_decay=self.weight_decay)
        if self.model_precision=='fp16':
            self.scaler = GradScaler()
        self.e_sche = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[60,80], gamma = self.adjust_lr_factor)

        logging.info('finished build model.')

    def _init_log(self, hparams_dict):

        self.log_path = os.path.join(self.log_path,'{}'.format(self.tab))
        os.makedirs(self.log_path,exist_ok=True)
        self.log_file = os.path.join(self.log_path,'{}.log'.format(self.tab))
        write2log(self.log_file,'================ Training Loss (%s) ================\n' % time.strftime("%c"))
        self.summary_writer = SummaryWriter(self.log_path)
        save_yaml(hparams_dict, self.log_file.replace('.log', '.yml'))

        self.result_img_dir = os.path.join(config.root_dir, 'result_images', '{}_on_gpu{}_val'.format(self.tab, self.GPUS))
        os.makedirs(self.result_img_dir,exist_ok=True)
        self.train_img_dir = os.path.join(config.root_dir, 'result_image_train', '{}_on_gpu{}_val'.format(self.tab, self.GPUS))
        os.makedirs(self.train_img_dir,exist_ok=True)
        self.model_save_dir = os.path.join(config.root_dir, 'checkpoints', '{}_on_gpu{}_val'.format(self.tab, self.GPUS))
        os.makedirs(self.model_save_dir,exist_ok=True)

    def _init_params(self):
        self.global_count = 0
        self.eval_cfg = {'mode':'matching_gts', 'is_training':False, 'calc_loss': False}
        self.val_cfg = {'mode':'parsing', 'calc_loss': False}
        self.GPUS = str(self.GPUS).split(',')

        if not self.distributed_training and self.master_batch_size != -1:
            rest_batch_size = (self.batch_size - self.master_batch_size)
            self.chunk_sizes = [self.master_batch_size]
            for i in range(len(self.GPUS) - 1):
              slave_chunk_size = rest_batch_size // (len(self.GPUS) - 1)
              if i < rest_batch_size % (len(self.GPUS) - 1):
                slave_chunk_size += 1
              self.chunk_sizes.append(slave_chunk_size)
            logging.info('training chunk_sizes:{}'.format(self.chunk_sizes))

        self.lr_hip_idx = np.array([constants.SMPL_ALL_54['L_Hip'], constants.SMPL_ALL_54['R_Hip']])
        self.lr_hip_idx_lsp = np.array([constants.LSP_14['L_Hip'], constants.LSP_14['R_Hip']])
        self.kintree_parents = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16,17, 18, 19, 20, 21],dtype=np.int)
        self.All54_to_LSP14_mapper = constants.joint_mapping(constants.SMPL_ALL_54, constants.LSP_14)

    def network_forward(self, model, meta_data, cfg_dict):
        ds_org, imgpath_org = get_remove_keys(meta_data,keys=['data_set','imgpath'])
        meta_data['batch_ids'] = torch.arange(len(meta_data['params']))
        if self.model_precision=='fp16':
            with autocast():
                outputs = model(meta_data, **cfg_dict)
        else:
            outputs = model(meta_data, **cfg_dict)
        outputs['detection_flag'], outputs['reorganize_idx'] = justify_detection_state(outputs['detection_flag'], outputs['reorganize_idx'])
        meta_data.update({'imgpath':imgpath_org, 'data_set':ds_org})
        outputs['meta_data']['data_set'], outputs['meta_data']['imgpath'] = reorganize_items([ds_org, imgpath_org], outputs['reorganize_idx'].cpu().numpy())
        return outputs

    def _create_data_loader(self,train_flag=True):
        logging.info('gathering datasets')
        datasets = MixedDataset(train_flag=train_flag)
        if self.distributed_training:
            data_sampler = torch.utils.data.distributed.DistributedSampler(datasets)
            return DataLoader(dataset = datasets,\
                batch_size = self.batch_size if train_flag else self.val_batch_size, sampler=data_sampler, \
                drop_last = True if train_flag else False, pin_memory = True,num_workers = self.nw)#shuffle = True
        else:
            return DataLoader(dataset = datasets,\
                batch_size = self.batch_size if train_flag else self.val_batch_size, shuffle = True, \
                drop_last = True if train_flag else False, pin_memory = True,num_workers = self.nw)

    def _create_single_data_loader(self, shuffle=False, **kwargs):
        logging.info('gathering datasets')
        datasets = SingleDataset(**kwargs)
        return DataLoader(dataset = datasets, shuffle=shuffle,batch_size = self.val_batch_size,\
                drop_last = False, pin_memory = True, num_workers = 2)

    def set_up_val_loader(self):
        eval_datasets = self.eval_datasets.split(',')
        self.evaluation_results_dict = {}
        for ds in eval_datasets:
            self.evaluation_results_dict[ds] = {'MPJPE':[], 'PAMPJPE':[]}
        self.dataset_val_list, self.dataset_test_list = {}, {}
        if 'mpiinf' in eval_datasets:
            self.dataset_val_list['mpiinf'] = self._create_single_data_loader(dataset='mpiinf_val', train_flag = False)
            self.dataset_test_list['mpiinf'] = self._create_single_data_loader(dataset='mpiinf_test', train_flag = False)
        if 'pw3d' in eval_datasets:
            self.dataset_val_list['pw3d'] = self._create_single_data_loader(dataset='pw3d', train_flag=False, mode='vibe', split='val')
            self.dataset_test_list['pw3d'] = self._create_single_data_loader(dataset='pw3d', train_flag=False, mode='vibe', split='test')
        logging.info('dataset_val_list:{}'.format(list(self.dataset_val_list.keys())))
        logging.info('evaluation_results_dict:{}'.format(list(self.evaluation_results_dict.keys())))

    def load_config_dict(self, config_dict):
        hparams_dict = {}
        for i, j in config_dict.items():
            setattr(self,i,j)
            hparams_dict[i] = j

        logging.basicConfig(level=logging.INFO)
        logging.info(config_dict)
        logging.info('-'*66)
        return hparams_dict
