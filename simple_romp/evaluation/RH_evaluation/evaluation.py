import os, sys
import os.path as osp
import numpy as np
import cv2
import torch

from .matching import match_2d_greedy


Relative_Human_dir = '/home/yusun/data_drive/dataset/Relative_human'
results_path = '/home/yusun/data_drive/evaluation_results/Relative_results/zip_files/CRMH_RH_results.npz'

relative_age_types = ['adult', 'teen', 'kid', 'baby']
relative_depth_types = ['eq', 'cd', 'fd']

BK_19 = {
    'Head_top': 0, 'Nose': 1, 'Neck': 2, 'L_Eye': 3, 'R_Eye': 4, 'L_Shoulder': 5, 'R_Shoulder': 6, 'L_Elbow': 7, 'R_Elbow': 8, 'L_Wrist': 9, 'R_Wrist': 10,\
    'L_Hip': 11, 'R_Hip': 12, 'L_Knee':13, 'R_Knee':14,'L_Ankle':15, 'R_Ankle':16,'L_BigToe':17, 'R_BigToe':18
}

OCHuman_19 = {
    'R_Shoulder':0, 'R_Elbow':1, 'R_Wrist':2, 'L_Shoulder':3, 'L_Elbow':4, 'L_Wrist':5, \
    'R_Hip': 6, 'R_Knee':7, 'R_Ankle':8, 'L_Hip':9, 'L_Knee':10, 'L_Ankle':11, 'Head_top':12, 'Neck':13,\
    'R_Ear':14, 'L_Ear':15, 'Nose':16, 'R_Eye':17, 'L_Eye':18
    }

Crowdpose_14 = {"L_Shoulder":0, "R_Shoulder":1, "L_Elbow":2, "R_Elbow":3, "L_Wrist":4, "R_Wrist":5,\
     "L_Hip":6, "R_Hip":7, "L_Knee":8, "R_Knee":9, "L_Ankle":10, "R_Ankle":11, "Head_top":12, "Neck_LSP":13}

def joint_mapping(source_format, target_format):
    mapping = np.ones(len(target_format),dtype=np.int32)*-1
    for joint_name in target_format:
        if joint_name in source_format:
            mapping[target_format[joint_name]] = source_format[joint_name]
    return np.array(mapping)

def _calc_relative_depth_error_weak_(pred_depths, depth_ids, reorganize_idx, age_gts=None, matched_mask=None):
    depth_ids = depth_ids.to(pred_depths.device)
    depth_ids_vmask = depth_ids != -1
    pred_depths_valid = pred_depths[depth_ids_vmask]
    valid_inds = reorganize_idx[depth_ids_vmask]
    depth_ids = depth_ids[depth_ids_vmask]
    age_gts = age_gts[depth_ids_vmask]
    error_dict = {'eq': [], 'cd': [], 'fd':[], 'eq_age': [], 'cd_age': [], 'fd_age':[]}
    error_each_age = {age_type:[] for age_type in relative_age_types}
    for b_ind in torch.unique(valid_inds):
        sample_inds = valid_inds == b_ind
        if matched_mask is not None:
            sample_inds *= matched_mask[depth_ids_vmask]
        did_num = sample_inds.sum()
        if did_num > 1:
            pred_depths_sample = pred_depths_valid[sample_inds]
            triu_mask = torch.triu(torch.ones(did_num, did_num), diagonal=1).bool()
            dist_mat = (pred_depths_sample.unsqueeze(0).repeat(did_num, 1) - pred_depths_sample.unsqueeze(1).repeat(1,did_num))[triu_mask]
            did_mat = (depth_ids[sample_inds].unsqueeze(0).repeat(did_num, 1) - depth_ids[sample_inds].unsqueeze(1).repeat(1,did_num))[triu_mask]
            
            error_dict['eq'].append(dist_mat[did_mat==0])
            error_dict['cd'].append(dist_mat[did_mat<0])
            error_dict['fd'].append(dist_mat[did_mat>0])
            if age_gts is not None:
                age_sample = age_gts[sample_inds]
                age_mat = torch.cat([age_sample.unsqueeze(0).repeat(did_num, 1).unsqueeze(-1), age_sample.unsqueeze(1).repeat(1, did_num).unsqueeze(-1)], -1)[triu_mask]
                error_dict['eq_age'].append(age_mat[did_mat==0])
                error_dict['cd_age'].append(age_mat[did_mat<0])
                error_dict['fd_age'].append(age_mat[did_mat>0])
            # error_dict['all'].append([len(eq_dists), len(cd_dists), len(fd_dists)]) 
            # error_dict['correct'].append([(torch.abs(eq_dists)<thresh).sum().item(), (cd_dists<-thresh).sum().item(), (fd_dists>thresh).sum().item()])

    return error_dict

def _calc_matched_PCKh_(real, pred, kp2d_mask, error_thresh=0.143):
    # error_thresh is set as the ratio between the head and the body.
    # he head / body for normal people are between 6~8, therefore, we set it to 1/7=0.143
    PCKs = torch.ones(len(kp2d_mask)).float().cuda()*-1.
    if kp2d_mask.sum()>0:
        vis = (real>-1.).sum(-1)==real.shape[-1]
        error = torch.norm(real-pred, p=2, dim=-1)
        
        for ind, (e, v) in enumerate(zip(error, vis)):
            if v.sum() < 2:
                continue
            real_valid = real[ind,v]
            person_scales = torch.sqrt((real_valid[:,0].max(-1).values - real_valid[:,0].min(-1).values)**2 + \
                            (real_valid[:,1].max(-1).values - real_valid[:,1].min(-1).values)**2)
            error_valid = e[v]
            correct_kp_mask = ((error_valid / person_scales) < error_thresh).float()
            PCKs[ind] = correct_kp_mask.sum()/len(correct_kp_mask)
    return PCKs

def compute_prf1(count, miss, fp):
    if count == 0:
        return 0, 0, 0
    all_tp = count - miss
    all_fp = fp
    all_fn = miss
    all_f1_score = round(all_tp / (all_tp + 0.5 * (all_fp + all_fn)), 2)
    all_recall = round(all_tp / (all_tp + all_fn), 2)
    all_precision = round(all_tp / (all_tp + all_fp), 2)
    return all_precision, all_recall, all_f1_score

def get_results(depth_relative, missed_age_ids, dr_thresh=0.2, miss_fine=0.3):
    eval_results = {}
    eq_dists = torch.cat(depth_relative['eq'], 0)
    cd_dists = torch.cat(depth_relative['cd'], 0)
    fd_dists = torch.cat(depth_relative['fd'], 0)
    eq_age_ids = torch.cat(depth_relative['eq_age'], 0)
    cd_age_ids = torch.cat(depth_relative['cd_age'], 0)
    fd_age_ids = torch.cat(depth_relative['fd_age'], 0)
    dr_age_ids = torch.cat([eq_age_ids, cd_age_ids, fd_age_ids], 0)
    dr_all = np.array([len(eq_dists), len(cd_dists), len(fd_dists), len(missed_age_ids)*miss_fine])

    dr_corrects = [torch.abs(eq_dists)<dr_thresh, cd_dists<-dr_thresh, fd_dists>dr_thresh]
    print('Thresh: {} | Equal {:.2f} close {:.2f} far {:.2f}'.format(dr_thresh, dr_corrects[0].sum().item() / dr_all[0] * 100, \
                                        dr_corrects[1].sum().item() / dr_all[1] * 100, dr_corrects[2].sum().item() / dr_all[2] * 100))
    dr_corrects = torch.cat(dr_corrects,0)

    eval_results['PCRD_{}'.format(dr_thresh)] = dr_corrects.sum() / dr_all.sum()
    for age_ind, age_name in enumerate(relative_age_types):
        age_mask = (dr_age_ids == age_ind).sum(-1).bool()
        if age_mask.sum()>0:
            missed_num = (missed_age_ids == age_ind).sum()*miss_fine
            eval_results['PCRD_{}_{}'.format(dr_thresh, age_name)] = dr_corrects[age_mask].sum() / (age_mask.sum() + missed_num)
    return eval_results

def write2txt(path, contents):
    with open(path, 'w') as f:
        for line in contents:
            f.write(line+"\n")

class RH_Evaluation(object):
    def __init__(self, results_path, RH_dir, set_name='test'):
        super(RH_Evaluation, self).__init__()
        self.set_name = set_name
        self.load_gt(RH_dir)
        self.collect_results(results_path)
        self.results_txt_save_path = results_path.replace('.npz', '_results.txt')
        
        self.kp2d_mapper_BK = joint_mapping(BK_19, Crowdpose_14)
        self.kp2d_mapper_OCH = joint_mapping(OCHuman_19, Crowdpose_14)
        self.match_kp2ds()
        print('Results on {} set'.format(self.set_name))
        self.calc_error()
       
    def collect_results(self, results_path):
        print('loading results ...')
        self.results = np.load(results_path, allow_pickle=True)['results'][()]

    def no_predictions(self, miss_num):
        self.pr['all'].append(0)
        self.pr['falsePositive'].append(0)
        self.pr['miss'].append(miss_num)

    def load_gt(self, RH_dir):
        print('loading gts ...')
        annot_dir = osp.join(RH_dir, '{}_annots.npz'.format(self.set_name))
        self.annots = np.load(annot_dir, allow_pickle=True)['annots'][()]
        print(f'We got GTs of {len(self.annots)} images for evaluation.')
    
    def miss_mat(self, miss_gt_ids):
        return np.stack([np.ones(len(miss_gt_ids))*-1, miss_gt_ids],1).astype(np.int32)

    def match_kp2ds(self):
        self.match_results = {}
        self.missed_ids = {}
        self.pr = {'all':[], 'falsePositive':[], 'miss':[],}
        self.kp2ds = {'gts':{}, 'preds':{}}
        for img_name in self.annots.keys():
            annots = self.annots[img_name]
            gt_kp2ds = []
            gt_inds = []
            for idx,annot in enumerate(annots):
                vbox = np.array(annot['bbox'])
                if 'kp2d' in annot:
                    if annot['kp2d'] is not None:
                        joint = np.array(annot['kp2d']).reshape((-1,3))
                        invalid_kp_mask = joint[:,2]==0
                        joint[invalid_kp_mask] = -2.
                        joint[:,2] = joint[:,2]>0
                        if len(joint) == 19:
                            is_BK = len(os.path.basename(img_name).replace('.jpg',''))==7
                            if is_BK:
                                joints = joint[self.kp2d_mapper_BK]
                                joints[self.kp2d_mapper_BK==-1] = -2
                            else:
                                joints = joint[self.kp2d_mapper_OCH]
                                joints[self.kp2d_mapper_BK==-1] = -2
                        elif len(joint) == 14:
                            joints = joint
                        gt_kp2ds.append(joints)
                        gt_inds.append(idx)
            gt_kp2ds = np.array(gt_kp2ds)
            
            if img_name not in self.results:
                self.no_predictions(len(gt_inds))
                self.missed_ids[img_name] = np.array(gt_inds)
                continue
            results = self.results[img_name]
            if isinstance(results, list):
                pred_kp2ds = np.array([r['kp2ds'] for r in results])
            elif isinstance(results, dict):
                pred_kp2ds = results['kp2ds']
            
            valid_kps = gt_kp2ds[:,:,2]>0
            valid_person = valid_kps.sum(-1)>0
            valid_kps = valid_kps[valid_person]
            gt_kp2ds = gt_kp2ds[valid_person]
            assert len(pred_kp2ds)>0, print('no prediction')
            assert len(gt_kp2ds)>0, print('no GT')
            bestMatch, falsePositives, misses = match_2d_greedy(pred_kp2ds, gt_kp2ds[:,:,:2], valid_kps, imgPath=img_name)
            if len(bestMatch)>0:
                pred_ids, gt_ids = bestMatch[:,0], bestMatch[:,1]
                self.kp2ds['gts'][img_name] = gt_kp2ds[gt_ids,:,:2]
                self.kp2ds['preds'][img_name] = pred_kp2ds[pred_ids]
                bestMatch[:,1] = np.array([gt_inds[ind] for ind in gt_ids])
            if len(misses)>0:
                self.missed_ids[img_name] = np.array([gt_inds[ind] for ind in misses])
            self.match_results[img_name] = bestMatch
            self.pr['all'].append(len(pred_kp2ds))
            self.pr['falsePositive'].append(len(falsePositives))
            self.pr['miss'].append(len(misses))

        all_precision, all_recall, all_f1_score = compute_prf1(sum(self.pr['all']), sum(self.pr['miss']), sum(self.pr['falsePositive']))
        
        print('Precision: {} | Recall: {} | F1 score: {}'.format(all_precision, all_recall, all_f1_score))
        
    def calc_error(self):
        self.mPCKh = []
        depth_relative = {'eq': [], 'cd': [], 'fd':[], 'eq_age': [], 'cd_age': [], 'fd_age':[]}
        for img_name, match_mat in self.match_results.items():
            if len(match_mat)==0:
                continue
            pred_ids, gt_ids = match_mat[:,0], match_mat[:,1]
            annots = self.annots[img_name]
            depth_ids = torch.from_numpy(np.array([annots[ind]['depth_id'] for ind in gt_ids]))
            age_ids = torch.from_numpy(np.array([annots[ind]['age'] for ind in gt_ids]))

            results = self.results[img_name]
            if isinstance(results, list):
                pred_depth = torch.from_numpy(np.array([results[ind]['trans'][2] for ind in pred_ids]))
            elif isinstance(results, dict):
                pred_depth = torch.from_numpy(results['trans'][pred_ids,2])
            default_organize_idx = torch.zeros(len(depth_ids))

            mPCKh = _calc_matched_PCKh_(torch.from_numpy(self.kp2ds['gts'][img_name]).float(), torch.from_numpy(self.kp2ds['preds'][img_name]).float(), torch.ones(len(self.kp2ds['gts'][img_name])).bool())
            self.mPCKh.append(mPCKh)
            matched_mask = torch.ones(len(depth_ids)).bool()
            relative_depth_errors = _calc_relative_depth_error_weak_(pred_depth, depth_ids, default_organize_idx, age_ids, matched_mask=matched_mask.cpu())
            for dr_type in relative_depth_types:
                depth_relative[dr_type] += relative_depth_errors[dr_type]
                depth_relative[dr_type+'_age'] += relative_depth_errors[dr_type+'_age']
        
        missed_age_ids = []
        for img_name, missed_id in self.missed_ids.items():
            annots = self.annots[img_name]
            missed_age_ids.append(np.array([annots[ind]['age'] for ind in missed_id]))
        missed_age_ids = torch.from_numpy(np.concatenate(missed_age_ids, 0))
        
        print_results = []
        all_mPCKh = torch.cat(self.mPCKh).mean()
        print_results.append('mPCKh_0.6: {:.2f}'.format(all_mPCKh * 100))
        eval_results = get_results(depth_relative, missed_age_ids)
        for key, item in eval_results.items():
            print_results.append('{}: {:.2f}'.format(key, float(item.item())*100))
        for numbers in print_results:
            print(numbers)
        write2txt(self.results_txt_save_path, print_results)


if __name__ == '__main__':
    RH_Evaluation(results_path, Relative_Human_dir, set_name='test')
