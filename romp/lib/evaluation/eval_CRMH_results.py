import pickle
import pickle as pkl
import zipfile
import torch
import numpy as np
import os,sys,glob
import joblib
import time
import cv2
import json
from scipy.sparse import csr_matrix
sys.path.append(os.path.abspath(__file__).replace('evaluation/eval_CRMH_results.py',''))
from utils.util import transform_rot_representation
from evaluation import compute_error_verts, compute_similarity_transform, compute_similarity_transform_torch, \
                    batch_compute_similarity_transform_torch, compute_mpjpe

from core.base import *
from core.eval import val_result,print_results

class Evaluate(Base):
    def __init__(self):
        super(Evaluate, self).__init__()
        self.set_up_smplx()
        self.dataset_eval_crmh = 'pw3d-hc'
        self.load_gt()
        self.collect_results()
        print('Initialization finished!')
        if self.dataset_eval_crmh == '3DOH50K':
            self.test_3doh50k()
        elif self.dataset_eval_crmh =='pw3d-hc':
            self.test_3dpw_hc()

    def collect_results(self):
        print('loading results..')
        if self.dataset_eval_crmh == '3DOH50K':
            self.results = np.load("/export/home/suny/multiperson/mmdetection/3DOH50K_test_results.npz",allow_pickle=True)['results'][()]
        elif self.dataset_eval_crmh == 'pw3d-hc':
            self.results = np.load("/export/home/suny/dataset/CRMH_results.npz",allow_pickle=True)['results'][()]

    def test_3doh50k(self):
        real_3d,predicts = [],[]
        J_regressor = self.smplx.J_regressor.cpu()
        for img_name, annot in self.annotations.items():
            img_name+='.jpg'
            if img_name not in self.results:
                print('missing {}'.format(img_name))
                continue
            pred_vertices = self.results[img_name]['verts']
            bboxes = self.results[img_name]['bbox']
            confidences = []
            for box_id, bbox in enumerate(bboxes):
                confidences.append(bbox[-1])
            person_id = np.argmax(np.array(confidences))
            kp3d_preds = torch.einsum('bik,ji->bjk', [pred_vertices, J_regressor])[person_id]
            kp3d_gt = torch.from_numpy(np.array(annot['smpl_joints_3d'])/10.)
            
            real_3d.append(kp3d_gt)
            predicts.append(kp3d_preds)

        real_3d, predicts = torch.stack(real_3d), torch.stack(predicts)
        abs_error = self.calc_mpjpe(real_3d, predicts, lrhip=self.lr_hip_idx_smpl24).float().cpu().numpy()*1000
        rt_error = self.calc_pampjpe(real_3d, predicts, lrhip=self.lr_hip_idx_smpl24).float().cpu().numpy()*1000
        print('evaluated on test set of 3DOH50K, get MPJPE: {} ; PAMPJPE: {}'.format(abs_error.mean(), rt_error.mean()))

    def load_gt(self):
        print('loading gt ..')
        if self.dataset_eval_crmh == '3DOH50K':
            self.annotations = json.load(open("/export/home/suny/dataset/3DOH50K/test/annots.json", 'rb'))
            #self._create_single_data_loader(dataset='oh', train_flag=False, split='test',joint_format='smpl24')
        elif self.dataset_eval_crmh == 'pw3d-hc':
            self.data_loader = self._create_single_data_loader(dataset='pw3d', train_flag = False, split='all', mode='HC', joint_format='smpl24')
            PW3D_HCsubset = {'courtyard_basketball_00':[110,160],  'courtyard_basketball_00':[200,280], 'courtyard_captureSelfies_00':[150,270], 'courtyard_captureSelfies_00':[500,600],\
                'courtyard_dancing_00':[60,370],  'courtyard_dancing_01':[60,270], 'courtyard_hug_00':[100,500], 'downtown_bus_00':[1620,1900]}
            self.annotations = {}
            annot_dir = os.path.join('/export/home/suny/dataset/3DPW/','sequenceFiles')
            for action_name, clip_num in PW3D_HCsubset.items():
                for set_name in ['train', 'test', 'validation']:
                    annot_file_path = os.path.join(annot_dir, set_name, action_name+'.pkl')
                    if os.path.exists(annot_file_path):
                        break
                data_gt = pickle.load(open(annot_file_path, 'rb'), encoding='latin1')
                poses2d_gt = data_gt['poses2d']#[:,clip_num[0]:clip_num[1]]
                poses2d_gt = np.array(poses2d_gt).transpose(0,1,3,2)
                center_loc = np.zeros((poses2d_gt.shape[0],poses2d_gt.shape[1],2))
                for subject_id, poses in enumerate(poses2d_gt):
                    for frame_id, pose in enumerate(poses):
                        center_loc[subject_id, frame_id] = pose[pose[:,-1]>0,:2].mean(0)

                self.annotations[action_name] = center_loc[:,:,:2]

    def test_3dpw_hc(self):
        real_3d,predicts = [],[]
        self.joint_regressor = self.smplx.J_regressor.T.cpu()

        with torch.no_grad():
            for test_iter,data_3d in enumerate(self.data_loader):
                for batch_id, imgpath in enumerate(data_3d['imgpath']):
                    action_name = imgpath.split('/')[-2]
                    img_name = os.path.basename(imgpath)
                    frame_id = int(img_name.replace('.jpg','').replace('image_',''))
                    center_gts = self.annotations[action_name][:,frame_id]
                    pred_result = self.results[action_name][frame_id]
                    annot_gt = data_3d['kp_3d'][batch_id]
                    annot_gt = annot_gt[annot_gt[:,0,0]>-2.]
                    
                    
                    bbox_detected = pred_result['bbox']
                    bbox_center_gts = self.annotations[action_name][:,frame_id]
                    pred_verts = pred_result['verts']
                    pred_kp3ds = self.regress_kp3d_from_verts(pred_verts).numpy()  
                    kp3d_matched = [] 
                    gt_valid_idx = []
                    for gt_idx,bbox_center_gt in enumerate(bbox_center_gts):
                        if np.isnan(bbox_center_gt[0]):
                            continue
                        gt_valid_idx.append(gt_idx)
                        bbox_center_gt = bbox_center_gt[::-1]
                        frame_dist_dict = {}
                        for pred_subject_id,bbox_pred in enumerate(bbox_detected):
                            bbox_center_pred = calc_center(bbox_pred)
                            dist = np.sqrt((((bbox_center_pred-bbox_center_gt)/10.)**2).sum())
                            frame_dist_dict[dist] = pred_subject_id
                        closet_frame_dist = np.min(np.array(list(frame_dist_dict.keys())))
                        closet_subject_id = frame_dist_dict[closet_frame_dist]
                        kp3d_matched.append(pred_kp3ds[closet_subject_id])
                    kp3d_matched = np.array(kp3d_matched)
                    gt_valid_idx = np.array(gt_valid_idx)
                    predicts.append(torch.from_numpy(kp3d_matched).float())
                    real_3d.append(annot_gt[gt_valid_idx])

        real_3d, predicts = torch.cat(real_3d,0), torch.cat(predicts,0)
        abs_error = self.calc_mpjpe(real_3d, predicts, lrhip=self.lr_hip_idx_smpl24).float().cpu().numpy()*1000
        rt_error = self.calc_pampjpe(real_3d, predicts, lrhip=self.lr_hip_idx_smpl24).float().cpu().numpy()*1000
        print('evaluated on test set of 3DPW-HC, get MPJPE: {} ; PAMPJPE: {}'.format(abs_error.mean(), rt_error.mean()))


    def regress_kp3d_from_verts(self,verts):
        #verts = torch.from_numpy(verts)
        joint_x = torch.matmul(verts[:, :, 0], self.joint_regressor)
        joint_y = torch.matmul(verts[:, :, 1], self.joint_regressor)
        joint_z = torch.matmul(verts[:, :, 2], self.joint_regressor)

        joints = torch.stack([joint_x, joint_y, joint_z], dim = 2)
        return joints

    def set_parent_tree(self):
        self.parents = [-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
        self.sellect_joints = [0, 1,2,4,5,16,17,18,19]
        self.parent_tree = []
        for idx, joint_idx in enumerate(self.sellect_joints):
            parent = []
            while joint_idx>-1:
                parent.append(joint_idx)
                joint_idx = int(self.parents[joint_idx])
            self.parent_tree.append(parent)

    def collect_3DPW_layout(self):
        self.layout = {}
        for split in os.listdir(self.ds_root_dir):
            for action in os.listdir(os.path.join(self.ds_root_dir,split)):
                action_name = action.strip('.pkl')
                label_path = os.path.join(self.ds_root_dir,split,action)
                raw_labels = self.read_pickle(label_path)
                sequence_info = raw_labels['sequence']
                frame_num = len(raw_labels['img_frame_ids'])
                subject_num = len(raw_labels['poses'])
                pose2d = raw_labels['poses2d']
                self.layout[action_name] = [sequence_info, split, subject_num, frame_num, pose2d]


def calc_center(bbox):
    tl_y, tl_x, rb_y, rb_x= bbox
    bbox_center = np.array([(tl_x+rb_x)/2.,(tl_y+rb_y)/2.])
    return bbox_center

if __name__ == '__main__':
    Evaluate()