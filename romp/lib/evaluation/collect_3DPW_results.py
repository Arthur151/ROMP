import pickle
import zipfile
import sys, os
sys.path.append(os.path.abspath(__file__).replace('evaluation/collect_3DPW_results.py',''))
sys.path.append(os.path.abspath(__file__).replace('lib/evaluation/collect_3DPW_results.py',''))
from base import *
np.set_printoptions(precision=2, suppress=True)

class Submit(Base):
    def __init__(self):
        super(Submit, self).__init__()
        self.pw3d_path = os.path.join(args().dataset_rootdir, '3DPW')
        self.set_smpl_parent_tree()
        self._build_model_()
        self.collect_3DPW_layout()

        self.loader_val = self._create_single_data_loader(dataset='pw3d',train_flag=False,split='all', mode='normal')
        self.output_dir = args().output_dir
        print('Initialization finished!')

        save_dir = os.path.join(self.output_dir, 'R_'+os.path.basename(self.model_path).replace('.pkl',''))#time.strftime("results_%Y-%m-%d_%H:%M:%S", time.localtime())

        final_results_path = os.path.join(save_dir,'results.zip')
        print('final results will be saved to ',final_results_path)
        if not os.path.exists(final_results_path):
            self.evaluation()
            self.pack_results(save_dir)
        else:
            print(final_results_path, 'already exists. Going direct to evaluation')
        self.run_official_evaluation(save_dir)

    def collect_3DPW_layout(self):
        self.layout = {}
        root_dir = os.path.join(self.pw3d_path,"sequenceFiles/")
        for split in os.listdir(root_dir):
            for action in os.listdir(os.path.join(root_dir,split)):
                action_name = action.strip('.pkl')
                label_path = os.path.join(root_dir,split,action)
                raw_labels = read_pickle(label_path)
                frame_num = len(raw_labels['img_frame_ids'])
                subject_num = len(raw_labels['poses'])
                self.layout[action_name] = [split, subject_num, frame_num]

    def set_smpl_parent_tree(self):
        parents = torch.Tensor([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16,17, 18, 19, 20, 21])
        self.sellect_joints = [0,1,2,4,5,16,17,18,19]
        self.parent_tree = []
        for idx, joint_idx in enumerate(self.sellect_joints):
            parent = []
            while joint_idx>-1:
                parent.append(joint_idx)
                joint_idx = int(parents[joint_idx])
            self.parent_tree.append(parent)

    @torch.no_grad()
    def evaluation(self):
        eval_model = nn.DataParallel(self.model.module).eval()
        MPJPE, PAMPJPE, PCK3D, MPJAE = [],[],[],[]
        self.results = {}
        self.results_save = {}
        
        start_time = time.time()
        for test_iter,meta_data in enumerate(self.loader_val):
            ds_org, imgpath_org = get_remove_keys(meta_data,keys=['data_set','imgpath'])
            meta_data['batch_ids'] = torch.arange(len(meta_data['params']))
            meta_data_org = meta_data.copy()
            if self.model_precision=='fp16':
                with autocast():
                    outputs = eval_model(meta_data, **self.eval_cfg)
            else:
                outputs = eval_model(meta_data, **self.eval_cfg)

            outputs['meta_data']['data_set'], outputs['meta_data']['imgpath'] = reorganize_items([ds_org, imgpath_org], outputs['reorganize_idx'].cpu().numpy())
            meta_data = outputs['meta_data']
            params_pred = outputs['params']
            pose_pred = torch.cat([params_pred['global_orient'],params_pred['body_pose']],1).cpu()
            shape_pred = params_pred['betas'].cpu()
            kp3d_smpl = outputs['joints_smpl24']
            subject_ids = meta_data['subject_ids']
            imgpaths = meta_data['imgpath']
            
            kp3d_smpl, pose_pred = kp3d_smpl.cpu(), pose_pred.cpu()
            for idx,(imgpath, subject_id) in enumerate(zip(imgpaths, subject_ids)):
                imgpath = imgpath.replace(os.path.join(self.pw3d_path,'imageFiles/'),'')
                if imgpath not in self.results:
                    self.results[imgpath] = {}
                self.results[imgpath][subject_id] = [pose_pred[idx], shape_pred[idx], kp3d_smpl[idx,:24]]
            if test_iter%60==0:
                print('Processing {}/{}'.format(test_iter, len(self.loader_val)))

        print('Runtime: {},per sample {}'.format(time.time()-start_time, (time.time()-start_time)/50534))

    def pack_results(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        results = {}
        for split in ['train','validation','test']:
            os.makedirs(os.path.join(save_dir,split), exist_ok=True)
            results[split] = {}
        for action_name, [split, subject_num, frame_num] in self.layout.items():
            results[split][action_name] = [np.zeros((subject_num, frame_num, 24,3)), np.zeros((subject_num, frame_num, 82)), np.zeros((subject_num, frame_num, 9,3,3))]

        for imgpath in self.results:
            action_name, frame_id = imgpath.split('/')[0],int(imgpath.split('/')[1].replace('image_','').strip('.jpg'))
            for subject_id, [pose_pred, shape_pred, kp3d_smpl] in self.results[imgpath].items():
                split, subject_num, frame_num = self.layout[action_name]
                assert frame_id<frame_num, print('frame_id {} out range'.format(frame_id))
                assert subject_id<subject_num, print('subject_id {} out range'.format(subject_id))
                results[split][action_name][0][int(subject_id),frame_id] = kp3d_smpl
                results[split][action_name][1][int(subject_id),frame_id] = torch.cat([pose_pred, shape_pred])
                params_processed = self.process_params(pose_pred)
                results[split][action_name][2][int(subject_id),frame_id] = params_processed

        print('Saving results in ',save_dir)
        results = self.fill_empty(results)
        self.write_results(results, save_dir)
        self.zip_folder(save_dir)

    def fill_empty(self,results):
        for action_name, [split, subject_num, frame_num] in self.layout.items():
            for subject_id in range(subject_num):
                missing_frame = []
                for frame_id in range(frame_num):
                    empty_flag = results[split][action_name][0][subject_id, frame_id,0,0] == 0
                    if empty_flag:
                        missing_frame.append(frame_id)
                        sampling_frame = frame_id-1 if frame_id != 0 else 1
                        for inds in range(len(results[split][action_name])):
                            results[split][action_name][inds][int(subject_id),frame_id] = results[split][action_name][inds][int(subject_id),sampling_frame]

                #print(split,action_name,subject_id,'missing {} frames:'.format(len(missing_frame)),missing_frame)
        return results

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
                kp3d_result, params_pred, rotation_result = results[split][action]
                save_dict = {'jointPositions':kp3d_result, 'orientations':rotation_result, 'smpl_params':params_pred}
                save_path = os.path.join(save_dir, split, action+'.pkl')
                save_pickle(save_dict, save_path)

    def zip_folder(self, save_dir):
        os.chdir(save_dir)
        os.system('zip -r results.zip *')

    def run_official_evaluation(self, save_dir):
        print('Saving dir:', save_dir)
        os.chdir(os.path.join(config.code_dir,'evaluation'))
        os.system("python pw3d_eval/evaluate.py {} {}".format(\
            save_dir.replace(' ','\ '), os.path.join(self.pw3d_path,'sequenceFiles').replace(' ','\ ')))
        #os.system('cp {} {}'.format(self.model_path, save_dir))

def read_pickle(file_path):
    return pickle.load(open(file_path,'rb'),encoding='iso-8859-1')

def save_pickle(content, file_path):
    with open(file_path, 'wb') as handle:
        pickle.dump(content, handle, protocol=pickle.HIGHEST_PROTOCOL)

def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
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
    input_args = sys.argv[1:]
    if sum(['configs_yml' in input_arg for input_arg in input_args])==0:
        input_args.append("--configs_yml=configs/eval_3dpw_challenge.yml")
    with ConfigContext(parse_args(input_args)):
        print(args().configs_yml)
        submitor = Submit()