# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2020 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: Vassilis Choutas, vassilis.choutas@tuebingen.mpg.de

import os
import os.path as osp
import sys
import pickle

import numpy as np
import open3d as o3d
import torch
from loguru import logger
from tqdm import tqdm

from smplx import build_layer

from smpl_family.transfer_model.config import parse_args, update_args
from smpl_family.transfer_model.data import build_dataloader
from smpl_family.transfer_model.transfer_model import run_fitting
from smpl_family.transfer_model.utils import read_deformation_transfer, np_mesh_to_o3d


class SMPLParamConverter(object):
    def __init__(self, relation='smpl2smplx'):
        #exp_cfg = parse_args()
        if relation == 'smpl2smplx':
            exp_cfg = update_args('romp/lib/smpl_family/smpl_transfer_config_files/smpl2smplx.yaml')
        elif relation == 'smpl2smpl':
            exp_cfg = update_args('romp/lib/smpl_family/smpl_transfer_config_files/smpl2smpl.yaml')
        if torch.cuda.is_available() and exp_cfg["use_cuda"]:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
            if exp_cfg["use_cuda"]:
                if input("use_cuda=True and GPU is not available, using CPU instead,"
                        " would you like to continue? (y/n)") != "y":
                    sys.exit(3)

        logger.remove()
        logger.add(
            lambda x: tqdm.write(x, end=''), level=exp_cfg.logger_level.upper(),
            colorize=True)

        output_folder = osp.expanduser(osp.expandvars(exp_cfg.output_folder))
        logger.info(f'Saving output to: {output_folder}')
        os.makedirs(output_folder, exist_ok=True)

        model_path = exp_cfg.body_model.folder
        body_model = build_layer(model_path, **exp_cfg.body_model)
        logger.info(body_model)
        self.body_model = body_model.to(device=device)

        if relation == 'smpl2smpl':
            self.def_matrix = None
        else:
            deformation_transfer_path = exp_cfg.get('deformation_transfer_path', '')
            self.def_matrix = read_deformation_transfer(
                deformation_transfer_path, device=device)

        # Read mask for valid vertex ids
        mask_ids_fname = osp.expandvars(exp_cfg.mask_ids_fname)
        mask_ids = None
        if osp.exists(mask_ids_fname):
            logger.info(f'Loading mask ids from: {mask_ids_fname}')
            mask_ids = np.load(mask_ids_fname)
            mask_ids = torch.from_numpy(mask_ids).to(device=device)
        else:
            logger.warning(f'Mask ids fname not found: {mask_ids_fname}')
        self.mask_ids = mask_ids
        self.exp_cfg = exp_cfg
        self.device = device

    def convert_params(self, verts_face, init_var=None):
        """
        return {'vertices': np.asarray(mesh.vertices, dtype=np.float32),
                'faces': np.asarray(mesh.faces, dtype=np.int32)} 
        """
        
        var_dict = run_fitting(
            self.exp_cfg, verts_face, self.body_model, self.def_matrix, self.mask_ids, init_var=init_var)
        
        return var_dict


if __name__ == '__main__':
    converter = SMPLParamConverter()
