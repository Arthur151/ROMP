from ..base import *
from utils.cam_utils import convert_cam_to_3d_trans
from utils.demo_utils import save_meshes, get_video_bn, Time_counter
import platform
from utils.util import save_result_dict_tonpz
from dataset.internet import img_preprocess
from torch.cuda.amp import autocast

class Predictor(Base):
    def __init__(self, **kwargs):
        super(Predictor, self).__init__(**kwargs)
        self._build_model_()
        self._prepare_modules_()
        self.demo_cfg = {'mode':'parsing', 'calc_loss': False}
        if self.character == 'nvxia':
            assert os.path.exists(os.path.join('model_data','characters','nvxia')), \
                'Current released version does not support other characters, like Nvxia.'
            from romp.lib.models.nvxia import create_nvxia_model
            self.character_model = create_nvxia_model(self.nvxia_model_path)

    def net_forward(self, meta_data, cfg=None):
        ds_org, imgpath_org = get_remove_keys(meta_data,keys=['data_set','imgpath'])
        meta_data['batch_ids'] = torch.arange(len(meta_data['image']))
        if self.model_precision=='fp16':
            with autocast():
                outputs = self.model(meta_data, **cfg)
        else:
            outputs = self.model(meta_data, **cfg)
        outputs['detection_flag'], outputs['reorganize_idx'] = justify_detection_state(outputs['detection_flag'], outputs['reorganize_idx'])
        meta_data.update({'imgpath':imgpath_org, 'data_set':ds_org})
        outputs['meta_data']['data_set'], outputs['meta_data']['imgpath'] = reorganize_items([ds_org, imgpath_org], outputs['reorganize_idx'].cpu().numpy())
        return outputs

    def _prepare_modules_(self):
        self.model.eval()
        self.demo_dir = os.path.join(config.project_dir, 'demo')

    def __initialize__(self):
        if self.save_mesh:
            self.smpl_faces = pickle.load(open(os.path.join(args().smpl_model_path, 'SMPL_NEUTRAL.pkl'),'rb'), encoding='latin1')['f']
        print('Initialization finished!')

    def single_image_forward(self,image):
        meta_data = img_preprocess(image, '0', input_size=args().input_size, single_img_input=True)
        if '-1' not in self.GPUS:
            meta_data['image'] = meta_data['image'].cuda()
        outputs = self.net_forward(meta_data, cfg=self.demo_cfg)
        return outputs

    def reorganize_results(self, outputs, img_paths, reorganize_idx):
        results = {}
        cam_results = outputs['params']['cam'].detach().cpu().numpy().astype(np.float16)
        trans_results = outputs['cam_trans'].detach().cpu().numpy().astype(np.float16)
        smpl_pose_results = outputs['params']['poses'].detach().cpu().numpy().astype(np.float16)
        smpl_shape_results = outputs['params']['betas'].detach().cpu().numpy().astype(np.float16)
        joints_54 = outputs['j3d'].detach().cpu().numpy().astype(np.float16)
        kp3d_smpl24_results = outputs['joints_smpl24'].detach().cpu().numpy().astype(np.float16)
        kp3d_spin24_results = joints_54[:,constants.joint_mapping(constants.SMPL_ALL_54, constants.SPIN_24)]
        kp3d_op25_results = joints_54[:,constants.joint_mapping(constants.SMPL_ALL_54, constants.OpenPose_25)]
        verts_results = outputs['verts'].detach().cpu().numpy().astype(np.float16)
        pj2d_results = outputs['pj2d'].detach().cpu().numpy().astype(np.float16)
        pj2d_org_results = outputs['pj2d_org'].detach().cpu().numpy().astype(np.float16)
        center_confs = outputs['centers_conf'].detach().cpu().numpy().astype(np.float16)

        vids_org = np.unique(reorganize_idx)
        for idx, vid in enumerate(vids_org):
            verts_vids = np.where(reorganize_idx==vid)[0]
            img_path = img_paths[verts_vids[0]]                
            results[img_path] = [{} for idx in range(len(verts_vids))]
            for subject_idx, batch_idx in enumerate(verts_vids):
                results[img_path][subject_idx]['cam'] = cam_results[batch_idx]
                results[img_path][subject_idx]['cam_trans'] = trans_results[batch_idx]
                results[img_path][subject_idx]['poses'] = smpl_pose_results[batch_idx]
                results[img_path][subject_idx]['betas'] = smpl_shape_results[batch_idx]
                results[img_path][subject_idx]['j3d_all54'] = joints_54[batch_idx]
                results[img_path][subject_idx]['j3d_smpl24'] = kp3d_smpl24_results[batch_idx]
                results[img_path][subject_idx]['j3d_spin24'] = kp3d_spin24_results[batch_idx]
                results[img_path][subject_idx]['j3d_op25'] = kp3d_op25_results[batch_idx]
                results[img_path][subject_idx]['verts'] = verts_results[batch_idx]
                results[img_path][subject_idx]['pj2d'] = pj2d_results[batch_idx]
                results[img_path][subject_idx]['pj2d_org'] = pj2d_org_results[batch_idx]
                # wrong trans, please use cam_trans instead.
                #results[img_path][subject_idx]['trans'] = convert_cam_to_3d_trans(cam_results[batch_idx])
                results[img_path][subject_idx]['center_conf'] = center_confs[batch_idx]
        return results


    
        

    
