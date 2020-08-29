import os
import argparse
import math
import numpy as np
import torch
import yaml

code_dir = os.path.abspath(__file__).replace('config.py','')
project_dir = os.path.abspath(__file__).replace('/src/config.py','')
root_dir = os.path.abspath(__file__).replace('/CenterHMR/src/config.py','')
model_dir = os.path.join(project_dir,'models')
trained_model_dir = os.path.join(project_dir,'trained_models')

parser = argparse.ArgumentParser(description = 'CenterHMR: center-based multi-person 3D Mesh Recovery.')
parser.add_argument('--tab',type = str,default = 'CenterHMR',help = 'additional tabs')
parser.add_argument('--configs_yml',type = str,default = 'configs/basic_test.yml',help = 'setting for training')
parser.add_argument('--demo_image_folder',type = str,default = 'None',help = 'absolute path to the image folder containing the input images for evaluation')

mode_group = parser.add_argument_group(title='mode options')
#mode setting
mode_group.add_argument('--multi_person',type = bool,default = True,help = 'whether to make Multi-person Recovery')
mode_group.add_argument('--use_coordmaps',type = bool,default = True,help = 'use the coordmaps')
mode_group.add_argument('--head_conv_block_num',type=int,default = 2,help = 'number of conv block for head')

mode_group.add_argument('--kp3d_format', type=str, default='smpl24', help='the joint defination of KP 3D joints: smpl24 or coco25')
mode_group.add_argument('--eval',type = bool,default = False,help = 'whether to evaluation')
mode_group.add_argument('--max_person',default=16,type=int,help = 'max person number')
mode_group.add_argument('--BN_type', type=str, default='BN', help='BN layer type: BN, IBN')
mode_group.add_argument('--Rot_type', type=str, default='6D', help='rotation representation type: angular, 6D')

model_group = parser.add_argument_group(title='model settings')
model_group.add_argument('--center_idx', type=int, default=1, help='the index of person center joints in coco25 format')
model_group.add_argument('--centermap_size', type=int, default=64, help='the size of center map')
model_group.add_argument('--HMloss_type', type=str, default='MSE', help='the type of heatmap: MSE or focal loss')
model_group.add_argument('--model_precision', type=str, default='fp32', help='the model precision: fp16/fp32')
#model settings
model_group.add_argument('--baseline',type = str,default = 'hrnetv4',help = 'baseline model: hrnet')
model_group.add_argument('--input-size',default = 512,type = int, help = 'input image size 512 or 256.')
model_group.add_argument('--gmodel-path',type = str,default = '',help = 'trained model path of generator')
model_group.add_argument('--best-save-path',type = str,default = '',help = 'trained model path of best generator')

train_group = parser.add_argument_group(title='training options')
#basic training setting
train_group.add_argument('--print-freq', type = int, default = 50, help = 'training epochs')
train_group.add_argument('--epoch', type = int, default = 300, help = 'training epochs')
train_group.add_argument('--fine_tune',type = bool,default = False,help = 'whether to run online')
train_group.add_argument('--lr', help='lr',default=3e-4,type=float)
train_group.add_argument('--weight_decay', help='weight_decay',default=1e-5,type=float)
train_group.add_argument('--gpu',default=0,help='gpus',type=str)
train_group.add_argument('--batch_size',default=64,help='batch_size',type=int)
train_group.add_argument('--val_batch_size',default=64,help='valiation batch_size',type=int)
train_group.add_argument('--nw',default=4,help='number of workers',type=int)

dataset_group = parser.add_argument_group(title='datasets options')
#dataset setting:
dataset_group.add_argument('--dataset-rootdir',type=str, default=os.path.join(root_dir,'dataset/'), help= 'root dir of all datasets')
dataset_group.add_argument('--dataset',type=str, default='h36m,mpii,coco,aich,up,ochuman,lsp,movi' ,help = 'which datasets are used')
dataset_group.add_argument('--voc_dir', type = str, default = os.path.join(root_dir,'dataset/VOCdevkit/VOC2012/'), help = 'VOC dataset path')

other_group = parser.add_argument_group(title='other options')
#visulaization settings
other_group.add_argument('--high_resolution',type = bool,default = True,help = 'whether to visulize with high resolution 500*500')

#model save path and log file
other_group.add_argument('--save-best-folder', type = str, default = os.path.join(root_dir,'checkpoints/'), help = 'Path to save models')
other_group.add_argument('--log_path', type = str, default = os.path.join(root_dir,'log/'), help = 'Path to save log file')

smpl_group = parser.add_argument_group(title='SMPL options')
#smpl info
smpl_group.add_argument('--total-param-count',type = int,default = 85, help = 'the count of param param')
smpl_group.add_argument('--smpl_model_path',type = str,default = model_dir,help = 'smpl model path')


args = parser.parse_args()
args.adjust_lr_epoch = []
args.kernel_sizes = []
with open(args.configs_yml) as file:
    configs_update = yaml.full_load(file)
for key, value in configs_update['ARGS'].items():
    if isinstance(value,str):
        exec("args.{} = '{}'".format(key, value))
    else:
        exec("args.{} = {}".format(key, value))

print('-'*16)
print('Configuration:')
print(vars(args))
print('-'*16)