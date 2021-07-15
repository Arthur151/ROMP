import os
import argparse
import math
import numpy as np
import torch
import yaml
import logging


code_dir = os.path.abspath(__file__).replace('config.py','')
project_dir = os.path.abspath(__file__).replace('/src/lib/config.py','')
root_dir = project_dir.replace(project_dir.split('/')[-1],'')#os.path.abspath(__file__).replace('/CenterMesh/src/config.py','')
model_dir = os.path.join(project_dir,'models')
trained_model_dir = os.path.join(project_dir,'trained_models')

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description = 'ROMP: Monocular, One-stage, Regression of Multiple 3D People')
    parser.add_argument('--tab',type = str,default = 'ROMP_v1',help = 'additional tabs')
    parser.add_argument('--configs_yml',type = str,default = 'configs/single_image.yml',help = 'setting for training') #'configs/basic_training_v6_ld.yml' 
    parser.add_argument('--demo_image_folder',type = str,default = 'None',help = 'absolute path to the image folder containing the input images for evaluation')

    mode_group = parser.add_argument_group(title='mode options')
    #mode setting
    mode_group.add_argument('--local_rank',type = int,default=0,help = 'local rank for distributed training')
    mode_group.add_argument('--model_version',type = int,default = 1,help = 'model version')
    mode_group.add_argument('--multi_person',type = bool,default = True,help = 'whether to make Multi-person Recovery')

    mode_group.add_argument('--collision_aware_centermap',type = bool,default = False,help = 'whether to use collision_aware_centermap')
    mode_group.add_argument('--collision_factor',type = float,default = 0.2,help = 'whether to use collision_aware_centermap')
    mode_group.add_argument('--kp3d_format', type=str, default='smpl24', help='the joint defination of KP 3D joints: smpl24 or coco25')
    mode_group.add_argument('--eval',type = bool,default = False,help = 'whether to evaluation')
    mode_group.add_argument('--max_person',default=64,type=int,help = 'max person number')
    mode_group.add_argument('--input_size',default=512,type=int,help = 'size of input image')
    mode_group.add_argument('--Rot_type', type=str, default='6D', help='rotation representation type: angular, 6D')
    mode_group.add_argument('--rot_dim', type=int, default=6, help='rotation representation type: 3D angular, 6D')
    mode_group.add_argument('--centermap_conf_thresh',type = float,default = 0.25,help = 'whether to use centermap_conf_thresh')

    model_group = parser.add_argument_group(title='model settings')
    model_group.add_argument('--centermap_size', type=int, default=64, help='the size of center map')
    model_group.add_argument('--deconv_num', type=int, default=0, help='the size of center map')
    model_group.add_argument('--model_precision', type=str, default='fp16', help='the model precision: fp16/fp32')
    #model settings
    model_group.add_argument('--backbone',type = str,default = 'hrnet',help = 'backbone model: resnet50 or hrnet')
    model_group.add_argument('--input-size',default = 512,type = int, help = 'input image size 512 or 256.')
    model_group.add_argument('--gmodel-path',type = str,default = '',help = 'trained model path of generator')

    train_group = parser.add_argument_group(title='training options')
    #basic training setting
    train_group.add_argument('--print-freq', type = int, default = 50, help = 'training epochs')
    train_group.add_argument('--fine_tune',type = bool,default = False,help = 'whether to run online')
    train_group.add_argument('--gpu',default='0',help='gpus',type=str)
    train_group.add_argument('--batch_size',default=64,help='batch_size',type=int)
    train_group.add_argument('--val_batch_size',default=64,help='valiation batch_size',type=int)
    train_group.add_argument('--nw',default=4,help='number of workers',type=int)

    eval_group = parser.add_argument_group(title='evaluation options')
    eval_group.add_argument('--calc_PVE_error',type = bool,default =False)

    dataset_group = parser.add_argument_group(title='datasets options')
    #dataset setting:
    dataset_group.add_argument('--dataset-rootdir',type=str, default=os.path.join(root_dir,'dataset/'), help= 'root dir of all datasets')
    eval_group = parser.add_argument_group(title='evaluation options')

    other_group = parser.add_argument_group(title='other options')
    #visulaization settings
    other_group.add_argument('--high_resolution',type = bool,default = True,help = 'whether to visulize with high resolution 500*500')

    #model save path and log file
    other_group.add_argument('--save-best-folder', type = str, default = os.path.join(root_dir,'checkpoints/'), help = 'Path to save models')
    other_group.add_argument('--log-path', type = str, default = os.path.join(root_dir,'log/'), help = 'Path to save log file')

    smpl_group = parser.add_argument_group(title='SMPL options')
    #smpl info
    smpl_group.add_argument('--total-param-count',type = int,default = 85, help = 'the count of param param')
    smpl_group.add_argument('--smpl-mean-param-path',type = str,default = os.path.join(model_dir,'satistic_data','neutral_smpl_mean_params.h5'),
        help = 'the path for mean smpl param value')
    smpl_group.add_argument('--smpl-model',type = str,default = os.path.join(model_dir,'statistic_data','neutral_smpl_with_cocoplus_reg.txt'),
        help = 'smpl model path')

    smplx_group = parser.add_argument_group(title='SMPL-X options')
    smpl_group.add_argument('--smplx-model',type = bool,default = True, help = 'the count of param param')
    smpl_group.add_argument('--cam_dim',type = int,default = 3, help = 'the dimention of camera param')
    smpl_group.add_argument('--beta_dim',type = int,default = 10, help = 'the dimention of SMPL shape param, beta')
    smpl_group.add_argument('--smpl_joint_num',type = int,default = 22)
    smpl_group.add_argument('--smpl_model_path',type = str,default = os.path.join(model_dir),help = 'smpl model path')
    smpl_group.add_argument('--smpl_uvmap',type = str,default = os.path.join(model_dir, 'smpl', 'uv_table.npy'),help = 'smpl UV Map coordinates for each vertice')
    smpl_group.add_argument('--smpl_female_texture',type = str,default = os.path.join(model_dir, 'smpl', 'SMPL_sampleTex_f.jpg'),help = 'smpl UV texture for the female')
    smpl_group.add_argument('--smpl_male_texture',type = str,default = os.path.join(model_dir, 'smpl', 'SMPL_sampleTex_m.jpg'),help = 'smpl UV texture for the male')
    smpl_group.add_argument('--smpl_J_reg_h37m_path',type = str,default = os.path.join(model_dir, 'smpl', 'J_regressor_h36m.npy'),help = 'SMPL regressor for 17 joints from H36M datasets')
    smpl_group.add_argument('--smpl_J_reg_extra_path',type = str,default = os.path.join(model_dir, 'smpl', 'J_regressor_extra.npy'),help = 'SMPL regressor for 9 extra joints from different datasets')

    parsed_args = parser.parse_args(args=input_args)
    parsed_args.kernel_sizes = [5]
    with open(parsed_args.configs_yml) as file:
        configs_update = yaml.full_load(file)
    for key, value in configs_update['ARGS'].items():
        if isinstance(value,str):
            exec("parsed_args.{} = '{}'".format(key, value))
        else:
            exec("parsed_args.{} = {}".format(key, value))

    hrnet_pretrain = os.path.join(project_dir,'trained_models/pretrain.pkl') #os.path.join(model_dir,'pretrain_models','pose_higher_hrnet_w32_512.pth') #
    parsed_args.tab = '{}_cm{}_{}'.format(parsed_args.backbone,
                                          parsed_args.centermap_size,
                                          parsed_args.tab)

    return parsed_args


class ConfigContext(object):
    """
    Class to manage the active current configuration, creates temporary `yaml`
    file containing the configuration currently being used so it can be
    accessed anywhere.
    """
    yaml_filename = "active_context.yaml"
    def __init__(self, parsed_args):
        self.parsed_args = parsed_args

    def __enter__(self):
        # if a yaml is left over here, remove it
        self.clean()
        # store all the parsed_args in a yaml file
        with open(self.yaml_filename, 'w') as f:
            d = self.parsed_args.__dict__
            yaml.dump(d, f)

    def clean(self):
        if os.path.exists(self.yaml_filename):
            os.remove(self.yaml_filename)

    def __exit__(self, exception_type, exception_value, traceback):
        # delete the yaml file
        self.clean()

def args():
    # have to pass something or it'll try and read stdin, it should get overwritten on file load
    parsed_args = parse_args(['--tab', 'ROMP_v1']) 
    with open(ConfigContext.yaml_filename, 'r') as f:
        argsdict = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in argsdict.items():
        parsed_args.__dict__[k] = v
    return parsed_args

if __name__ == "__main__":
    _args = parse_args()
    with ConfigContext(_args):
        with open(ConfigContext.yaml_filename, 'r') as f:
            print(f.read())
        for k, v in args().__dict__.items():
            print(k, v)
            assert _args.__dict__[k] == v
