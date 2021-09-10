import pickle
import pickle as pkl
import zipfile
import torch
import numpy as np
import os,sys,glob
import joblib
import time
from scipy.sparse import csr_matrix
sys.path.append(os.path.abspath(__file__).replace('core/collect_VIBE_3DPW_results.py',''))
from evaluation import compute_error_verts, compute_similarity_transform, compute_similarity_transform_torch, \
                    batch_compute_similarity_transform_torch, compute_mpjpe


class Submit(object):
    def __init__(self):
        super(Submit, self).__init__()
        self.evaluation_mode = 'nottracking'
        self.output_dir = '/export/home/suny/results/'
        self.results_dir = os.path.join(self.output_dir,'VIBE_3DPW_results')
        self.ds_root_dir = "/export/home/suny/dataset/3DPW/sequenceFiles/"
        self.project_dir = '/export/home/suny/CenterMesh/'
        self.set_parent_tree()
        self.collect_3DPW_layout()
        save_dir = os.path.join(self.output_dir, '多人notrack-VIBE')#"多人{}-VIBE-retracking".format(self.evaluation_mode)) # 取对应人数结果-取第一结果填补
        self.eval_code_dir = os.path.join(os.path.join(self.project_dir,'src/evaluation'))
        print('Initialization finished!')

        self.joint_regressor = torch.from_numpy(csr_matrix.toarray(self.read_pickle(os.path.join(self.project_dir,'models/smpl_original/basicModel_f_lbs_10_207_0_v1.0.0.pkl'))['J_regressor'])).float().T

        self.eval_HC()
        #if 1:#not os.path.exists(os.path.join(save_dir, 'results.zip')):
        if self.evaluation_mode == 'tracking':
            self.get_results()
        else:
            self.results_matched_to_gt_bbox()
        params_results = self.pack_results(save_dir)

        self.eval_pve(save_dir, params_results)
        self.run_official_evaluation(save_dir)

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
                frame_num = len(raw_labels['img_frame_ids'])
                subject_num = len(raw_labels['poses'])
                pose2d = raw_labels['poses2d']
                self.layout[action_name] = [split, subject_num, frame_num, pose2d]

    def get_gt_bbox(self, pose2ds):
        gt_bbox = []
        for person_id,pose2d in enumerate(pose2ds):
            gt_bbox.append([])
            for pose in pose2d[:,:,:8]:
                bbox = pose[:2][:,pose[2]>0.05].mean(-1)
                # if missing pose 2d, use the bbox of previous frame
                if np.isnan(bbox[0]):
                    bbox = gt_bbox[person_id][-1]
                gt_bbox[person_id].append(bbox)
        return np.array(gt_bbox)

    def get_first_frame_bbox(self, bboxes):
        first_frame_bbox = []
        for person_id in range(len(bboxes)):
            first_frame_bbox.append(bboxes[person_id][0])
        return np.array(first_frame_bbox)

    def match_first_bbox(self, bbox_pred, bbox_gt):
        dist = np.linalg.norm(bbox_pred[None]-bbox_gt, axis=1)
        matched_id = np.argmin(dist)
        return matched_id

    def eval_pve(self, submit_dir, params_results):
        # Get all the GT and submission paths in paired list form
        truth_dir = self.ds_root_dir
        fnames_gt, fnames_pred = get_paths(submit_dir, truth_dir)
        params_preds, params_gts = get_data(params_results, fnames_gt)
        params_preds[:,:3] = 0
        params_gts[:,:3] = 0
        PVE = np.mean(compute_error_verts(target_theta=torch.from_numpy(params_gts).float(), pred_theta=torch.from_numpy(params_preds).float())) * 1000
        print('PVE: ', PVE)

    def eval_HC(self, submit_dir, params_results):
        # Get all the GT and submission paths in paired list form
        truth_dir = self.ds_root_dir
        fnames_gt, fnames_pred = get_paths(submit_dir, truth_dir)
        params_preds, params_gts = get_data(params_results, fnames_gt)
        params_preds[:,:3] = 0
        params_gts[:,:3] = 0
        PVE = np.mean(compute_error_verts(target_theta=torch.from_numpy(params_gts).float(), pred_theta=torch.from_numpy(params_preds).float())) * 1000
        print('PVE: ', PVE)

    def results_matched_to_gt_bbox(self):
        self.results = {}
        self.params_results = {}
        for action_name, details in self.layout.items():
            split, subject_num, frame_num, pose2d = details
            gt_bbox = self.get_gt_bbox(pose2d)
            result_path = os.path.join(self.results_dir, action_name+'_output.pkl') # action_name, 
            #result_path = os.path.join(action_name+'_output.pkl')
            if os.path.exists(result_path):
                result = joblib.load(result_path)
                self.results[action_name] = {}

                pred_results = {}
                for person_id in result.keys():
                    frame_ids = result[person_id]['frame_ids']
                    bboxes = result[person_id]['bboxes']
                    poses = result[person_id]['pose']
                    verts = result[person_id]['verts']
                    shapes = result[person_id]['betas']
                    kp3ds = self.regress_kp3d_from_verts(verts).numpy()
                    for idx, frame_id in enumerate(frame_ids):
                        if frame_id not in pred_results:
                            pred_results[frame_id] = {}
                        pred_results[frame_id][person_id] = {}
                        pred_results[frame_id][person_id]['bbox'] = bboxes[idx][:2]#(cx,cy, w,h)
                        pred_results[frame_id][person_id]['pose'] = poses[idx]
                        pred_results[frame_id][person_id]['kp3d'] = kp3ds[idx]
                        pred_results[frame_id][person_id]['verts'] = verts[idx]
                        pred_results[frame_id][person_id]['shapes'] = shapes[idx]          

                for subject_id in range(subject_num):
                    frame_ids, kp3d_results, pose_results, shape_results, bbox_results, verts_results= [], [], [], [],[],[]
                    gt_bbox_subj = gt_bbox[subject_id]
                    for frame_id in pred_results:
                        bbox_gt = gt_bbox_subj[frame_id]
                        frame_dist_dict = {}
                        for pred_subject_id in pred_results[frame_id]:
                            bbox_pred = pred_results[frame_id][pred_subject_id]['bbox']
                            dist = np.linalg.norm(bbox_pred-bbox_gt)
                            frame_dist_dict[dist] = pred_subject_id
                        closet_frame_dist = np.min(np.array(list(frame_dist_dict.keys())))
                        closet_subject_id = frame_dist_dict[closet_frame_dist]
                        #print(closet_frame_dist, closet_subject_id, frame_dist_dict)
                        frame_ids.append(frame_id)
                        kp3d_results.append(pred_results[frame_id][closet_subject_id]['kp3d'])
                        pose_results.append(pred_results[frame_id][closet_subject_id]['pose'])
                        bbox_results.append(pred_results[frame_id][closet_subject_id]['bbox'])
                        verts_results.append(pred_results[frame_id][closet_subject_id]['verts'])
                        shape_results.append(pred_results[frame_id][closet_subject_id]['shapes'])

                    self.results[action_name][subject_id] = [np.array(pose_results), np.array(kp3d_results), np.array(bbox_results), np.array(frame_ids), shape_results,verts_results]
            else:
                print(result_path, 'missing')

    def re_tracking(self, data, first_frame_bbox_gt):
        result, _, subject_num, action_name, person_id_list, frame_num = data
        pose_retracked, kp3d_retracked, frame_ids_retracked, bbox_retracked = [], [], [], []
        
        first_frame_dist = {}
        # get the first frame matched person
        for person_id_pred in person_id_list:
            bboxes_pred = result[person_id_pred]['bboxes']
            frame_ids = result[person_id_pred]['frame_ids']
            if frame_ids[0]==0:
                dist = np.linalg.norm(bboxes_pred[0][:2]-first_frame_bbox_gt)
                first_frame_dist[dist] = person_id_pred
        print(first_frame_dist)
        assert len(first_frame_dist)!=0, 'Error! first frame no person!'
        closet_first_frame_dist = np.min(np.array(list(first_frame_dist.keys())))
        closet_subject_id_first_frame = first_frame_dist[closet_first_frame_dist]

        bbox_retracked.append(result[closet_subject_id_first_frame]['bboxes'][:,:2])
        pose_retracked.append(result[closet_subject_id_first_frame]['pose'])
        kp3d_retracked.append(self.regress_kp3d_from_verts(result[closet_subject_id_first_frame]['verts']))
        frame_ids_retracked.append(np.array(result[closet_subject_id_first_frame]['frame_ids']))
        print('first frame subject_num {} with {} frames tracked'.format(closet_subject_id_first_frame, len(frame_ids_retracked[-1])))
        person_id_list.remove(closet_subject_id_first_frame)
        
        last_subject_bbox = result[closet_subject_id_first_frame]['bboxes'][-1][:2]

        while frame_ids_retracked[-1][-1]<frame_num-1 and len(person_id_list)>0:
            last_frame_id = frame_ids_retracked[-1][-1]

            tracking_margin = 10
            frame_dist = {}
            while len(frame_dist) == 0:
                for person_id_pred in person_id_list:
                    bboxes_pred = result[person_id_pred]['bboxes']
                    frame_ids = result[person_id_pred]['frame_ids']
                    if np.abs(frame_ids[0]-last_frame_id)<tracking_margin:
                        dist = np.linalg.norm(bboxes_pred[0][:2]-first_frame_bbox_gt)
                        frame_dist[dist] = person_id_pred
                if len(frame_dist)==0:
                    tracking_margin+=20
            #assert len(frame_dist)!=0, 'Error! tracking loss!'

            closet_frame_dist = np.min(np.array(list(frame_dist.keys())))
            closet_subject_id = frame_dist[closet_frame_dist]

            bbox_retracked.append(result[closet_subject_id]['bboxes'][:,:2])
            pose_retracked.append(result[closet_subject_id]['pose'])
            kp3d_retracked.append(self.regress_kp3d_from_verts(result[closet_subject_id]['verts']))
            frame_ids_retracked.append(np.array(result[closet_subject_id]['frame_ids']))
            print('latest frame subject_num {} with {} frames tracked, from {} to {}'.format(closet_subject_id, \
                len(frame_ids_retracked[-1]), frame_ids_retracked[-1][0],frame_ids_retracked[-1][-1]))
            person_id_list.remove(closet_subject_id)

        results_retracked = [np.concatenate(pose_retracked,0), np.concatenate(kp3d_retracked,0), \
                             np.concatenate(bbox_retracked,0), np.concatenate(frame_ids_retracked,0)]
        return results_retracked


    def match_tracking(self, data):
        result, first_frame_bboxes_gt, subject_num, action_name, person_id_list, frame_num = data
        #sellect the top N detected person's results on more frames
        target_ids = np.argsort(np.array([len(result[person_id]['frame_ids']) for person_id in person_id_list]))[::-1][:subject_num]
        # if the number of detected person is less than the number of annotated target, fill up the empty with the last/first id detected.
        while len(target_ids)<subject_num:
            print('detection failed on one person', action_name)
            target_ids = np.concatenate([target_ids,np.array([target_ids[-1]])])

        person_id_sellected = np.array(person_id_list)[target_ids]
        frame_ids_sellected = [result[person_id]['frame_ids'] for person_id in person_id_sellected]
        person_ids_matched, results = [], []
        for idx, person_id in enumerate(person_id_sellected):
            frame_ids = frame_ids_sellected[idx]
            if frame_ids[0] == 0 and len(frame_ids)>frame_num-2:
                first_frame_bbox_pred = result[person_id]['bboxes'][0][:2]
                person_id_matched = self.match_first_bbox(first_frame_bbox_pred, first_frame_bboxes_gt)
                verts = result[person_id]['verts']
                kp3d_results = self.regress_kp3d_from_verts(verts)
                person_ids_matched.append(int(person_id_matched))
                results.append([result[person_id]['pose'], kp3d_results, result[person_id]['bboxes'][:,:2], frame_ids])
                del result[person_id]
                person_id_list.remove(person_id)
            else:
                print('before retracking, frame number tracked by YOLO is',len(result[person_id]['frame_ids']), \
                    ',from', result[person_id]['frame_ids'][0], 'to', result[person_id]['frame_ids'][-1], \
                    ',and the total frame number of {} is {}'.format(frame_num, action_name))
        
        #对于没有tracking成功的，通过拼接各个tracking小段结果，获得完整的tracking结果
        for person_id, first_frame_bbox_gt in enumerate(first_frame_bboxes_gt):
            if person_id in person_ids_matched:
                continue
            else:
                print('Re_tracking person', person_id, 'of', action_name)
                results_retracked = self.re_tracking(data, first_frame_bbox_gt)
                person_ids_matched.append(person_id)
                results.append(results_retracked)
        
        person_ids_matched = np.array(person_ids_matched).astype(np.uint8)

        print(person_ids_matched)
        return person_ids_matched, results

    def get_results(self):
        self.results = {}
        for action_name, details in self.layout.items():
            split, subject_num, frame_num, pose2d = details
            gt_bbox = self.get_gt_bbox(pose2d)
            first_frame_bboxes_gt = self.get_first_frame_bbox(gt_bbox)
            result_path = os.path.join(self.results_dir, action_name+'_output.pkl') # action_name, 
            #result_path = os.path.join(action_name+'_output.pkl')
            if os.path.exists(result_path):
                result = joblib.load(result_path)
                self.results[action_name] = {}
                person_id_list = list(result.keys())
                person_ids_matched, results = \
                self.match_tracking((result, first_frame_bboxes_gt, subject_num, action_name, person_id_list, frame_num))
                for person_id_matched, result in zip(person_ids_matched, results):
                    self.results[action_name][person_id_matched] = result
            else:
                print(result_path, 'missing')


    def regress_kp3d_from_verts(self,verts):
        verts = torch.from_numpy(verts)
        joint_x = torch.matmul(verts[:, :, 0], self.joint_regressor)
        joint_y = torch.matmul(verts[:, :, 1], self.joint_regressor)
        joint_z = torch.matmul(verts[:, :, 2], self.joint_regressor)

        joints = torch.stack([joint_x, joint_y, joint_z], dim = 2)
        return joints

    def pack_results(self,save_dir):
        os.makedirs(save_dir, exist_ok=True)
        results = {}
        params_results = {}
        bbox_results = {}
        for split in ['train','validation','test']:
            os.makedirs(os.path.join(save_dir,split), exist_ok=True)
            results[split] = {}
        for action_name in self.results:
            split, subject_num, frame_num, pose2d = self.layout[action_name]
            subject_num = len(list(self.results[action_name].keys()))
            results[split][action_name] = [np.zeros((subject_num, frame_num, 24,3)), np.zeros((subject_num, frame_num, 9,3,3))]
            bbox_results[action_name] = np.zeros((subject_num, frame_num, 2))
            params_results[action_name] = [np.zeros((subject_num, frame_num, 72)), np.zeros((subject_num, frame_num, 10))]

        for action_id,action_name in enumerate(self.results):
            for subject_id,[pose_preds, kp3d_smpl, bboxes,frame_ids,shape_results,verts_results] in self.results[action_name].items():
                print('processing ', action_name, '{}/{}'.format(action_id,60))
                split, subject_num, frame_num, pose2d = self.layout[action_name]
                subject_id = int(subject_id)
                #print(action_name,subject_id,np.array(frame_ids))
                params_results[action_name][0][subject_id][np.array(frame_ids)] = pose_preds.copy()
                params_results[action_name][1][subject_id][np.array(frame_ids)] = shape_results
                pose_result = []
                for pose_pred in pose_preds:
                    params_processed = self.process_params(torch.from_numpy(pose_pred))
                    pose_result.append(params_processed)
                results[split][action_name][0][subject_id][np.array(frame_ids)] = kp3d_smpl
                results[split][action_name][1][subject_id][np.array(frame_ids)] = np.array(pose_result)
                bbox_results[action_name][subject_id][np.array(frame_ids)] = np.array(bboxes)

        #self.write_results(results, save_dir)
        print('Saving results in ',save_dir)
        #results, bbox_results = self.fill_empty(results, bbox_results)
        np.savez(os.path.join(save_dir,'bbox_tracked.npz'),bbox=bbox_results)
        self.write_results(results, save_dir)
        self.zip_folder(save_dir)
        return params_results

    def fill_empty(self,results, bbox_results):
        for action_name in self.layout:
            split, subject_num, frame_num, pose2d = self.layout[action_name]
            kp3ds_mat, params_mat = results[split][action_name]
            for subject_id in range(kp3ds_mat.shape[0]):
                missing_frame = []
                for frame_id in range(frame_num):
                    empty_flag = kp3ds_mat[subject_id, frame_id,0,0] == 0
                    if empty_flag:
                        missing_frame.append(frame_id)
                        #print(split,action_name,subject_id,frame_id,'is missing..')
                        if frame_id!=0:
                            #print('fill the empty using the results of previous frames')
                            results[split][action_name][0][int(subject_id),frame_id] = results[split][action_name][0][int(subject_id),frame_id-1]
                            results[split][action_name][1][int(subject_id),frame_id] = results[split][action_name][1][int(subject_id),frame_id-1]
                            bbox_results[action_name][int(subject_id),frame_id] = bbox_results[action_name][int(subject_id),frame_id-1]
                        else:
                            #print('special case, the first frame results missing')
                            valid_id = np.where(results[split][action_name][0][int(subject_id)][:,0,0]!=0)[0][0]
                            results[split][action_name][0][int(subject_id),frame_id] = results[split][action_name][0][int(subject_id),valid_id]
                            results[split][action_name][1][int(subject_id),frame_id] = results[split][action_name][1][int(subject_id),valid_id]
                            bbox_results[action_name][int(subject_id),frame_id] = bbox_results[action_name][int(subject_id),valid_id]
                print(split,action_name,subject_id,'missing {} frames:'.format(len(missing_frame)),missing_frame)
        return results, bbox_results

    def process_params(self,params):
        '''
        calculate absolute rotation matrix in the global coordinate frame of K body parts. 
        The rotation is the map from the local bone coordinate frame to the global one.
        K= 9 parts in the following order: 
        root (JOINT 0) , left hip  (JOINT 1), right hip (JOINT 2), left knee (JOINT 4), right knee (JOINT 5), 
        left shoulder (JOINT 16), right shoulder (JOINT 17), left elbow (JOINT 18), right elbow (JOINT 19).
        parent kinetic tree [-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
        '''
        rotation_matrix = batch_rodrigues(params.reshape(-1,3)).numpy()
        rotation_final = []
        for idx, sellected_idx in enumerate(self.sellect_joints):
            rotation_global = np.eye(3)#init_matrix
            parents = self.parent_tree[idx]
            for parent_idx in parents:
                rotation_global = np.dot(rotation_matrix[parent_idx],rotation_global)
            rotation_final.append(rotation_global)
        rotation_final = np.array(rotation_final)
        return rotation_final

    def write_results(self, results,save_dir):
        for split in results:
            for action in results[split]:
                kp3d_result, rotation_result = results[split][action]
                save_dict = {'jointPositions':kp3d_result, 'orientations':rotation_result}
                save_path = os.path.join(save_dir, split, action+'.pkl')
                self.save_pickle(save_dict, save_path)

    def zip_folder(self, save_dir):
        os.chdir(save_dir)
        os.system('zip -r results.zip *')

    def run_official_evaluation(self, save_dir):
        os.chdir(self.eval_code_dir)
        truth_dir = os.path.join('/export/home/suny/dataset','3DPW','sequenceFiles')
        os.system('python pw3d_eval/evaluate.py {}'.format(save_dir))
    
    def print_results(self, MPJPE, PAMPJPE):
        print('MPJPE',np.concatenate(MPJPE,axis=0).mean())
        print('PAMPJPE',np.concatenate(PAMPJPE,axis=0).mean())

    def read_pickle(self,file_path):
        return pickle.load(open(file_path,'rb'),encoding='iso-8859-1')

    def save_pickle(self, content, file_path):
        with open(file_path, 'wb') as handle:
            pickle.dump(content, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_paths(submit_dir, truth_dir):
    """
    submit_dir: The location of the submit directory
    truth_dir: The location of the truth directory
    Return: two lists
            fnames_gt : the list of all files in ground truth folder
            fnames_pred : the list of all files in the predicted folder
    """
    fnames_gt = []
    fnames_pred = []

    keys = ['train', 'validation', 'test']

    for key in keys:
        fnames_gt_temp = sorted(glob.glob(os.path.join(truth_dir, key, "") + "*.pkl"))
        fnames_pred_temp = sorted(glob.glob(os.path.join(submit_dir, key, "") + "*.pkl"))
        fnames_gt = fnames_gt + fnames_gt_temp
        fnames_pred = fnames_pred + fnames_pred_temp

    assert len(fnames_gt) == len(fnames_pred)
    return sorted(fnames_gt), sorted(fnames_pred)

def check_valid_inds(poses2d, camposes_valid):
    """
    Computes the indices where further computations are required
    :param poses2d: N x 18 x 3 array of 2d Poses
    :param camposes_valid: N x 1 array of indices where camera poses are valid
    :return: array of indices indicating frame ids in the sequence which are to be evaluated
    """

    # find all indices in the N sequences where the sum of the 18x3 array is not zero
    # N, numpy array
    poses2d_mean = np.mean(np.mean(np.abs(poses2d), axis=2), axis=1)
    poses2d_bool = poses2d_mean == 0
    poses2d_bool_inv = np.logical_not(poses2d_bool)

    # find all the indices where the camposes are valid
    camposes_valid = np.array(camposes_valid).astype('bool')

    final = np.logical_and(poses2d_bool_inv, camposes_valid)
    indices = np.array(np.where(final == True)[0])

    return indices

def get_data(params_results, paths_gt):
    """
    The function reads all the ground truth.
    """
    params_gts, params_preds = [], []

    # construct the data structures -
    for path_gt in paths_gt:
        data_gt = pkl.load(open(path_gt, 'rb'), encoding='latin1')
        genders = data_gt['genders']

        for i in range(len(genders)):
            poses2d_gt = data_gt['poses2d']
            poses2d_gt_i = poses2d_gt[i]

            camposes_valid = data_gt['campose_valid']
            camposes_valid_i = camposes_valid[i]

            valid_indices = check_valid_inds(poses2d_gt_i, camposes_valid_i)
            # Get the ground truth SMPL body parameters - poses, betas and translation parameters
            pose_params = np.array(data_gt['poses'])
            pose_params = pose_params[i, valid_indices, :]

            shape_params = np.array(data_gt['betas'][i])
            shape_params = np.expand_dims(shape_params, 0)
            shape_params = shape_params[:, :10]
            shape_params = np.tile(shape_params, (pose_params.shape[0], 1))

            action_name = path_gt.split('/')[-1].strip('.pkl')
            #params_results[action_name][0][subject_id][np.array(frame_ids)] = pose_preds
            #params_results[action_name][1][subject_id][np.array(frame_ids)] = shape_results
            pose_pred = params_results[action_name][0][i][valid_indices]
            shape_pred = params_results[action_name][1][i][valid_indices]

            params_gt = np.concatenate([pose_params, shape_params],1)
            params_pred = np.concatenate([pose_pred, shape_pred],1)

            params_gts.append(params_gt)
            params_preds.append(params_pred)

    params_gts = np.concatenate(params_gts, 0)
    params_preds = np.concatenate(params_preds, 0)


    return params_preds, params_gts

def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


if __name__ == '__main__':
    submitor = Submit()