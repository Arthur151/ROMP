from dataset.image_base import *
from maps_utils.centermap import _calc_radius_

class Image_base_relative(Image_base):
    def __init__(self, train_flag=True, regress_smpl = False, **kwargs):
        super(Image_base_relative,self).__init__(train_flag=train_flag, regress_smpl=regress_smpl)
        self.depth_degree_thresh = [0.36,0.18,0]
        self.regress_smpl = regress_smpl

    def get_item_single_frame(self,index, augment_cfgs=None):
        # valid annotation flags for 
        # 0: 2D pose/bounding box(True/False), # 7: detecting all person/front-view person(True/False)
        # 1: 3D pose, 2: subject id, 3: smpl root rot, 4: smpl pose param, 5: smpl shape param, 6: global translation, 7: vertex of SMPL model
        valid_masks = np.zeros((self.max_person, 8), dtype=np.bool)
        info = self.get_image_info(index)

        position_augments, pixel_augments = self._calc_augment_confs(info['image'], info['kp2ds'], is_pose2d=info['vmask_2d'][:,0])

        img_info = process_image(info['image'], info['kp2ds'], augments=position_augments, is_pose2d=info['vmask_2d'][:,0])

        image, image_wbg, full_kps, offsets = img_info
        centermap, person_centers, full_kp2ds, used_person_inds, valid_masks[:,0], bboxes_hw_norm, heatmap, AE_joints = \
            self.process_kp2ds_bboxes(full_kps, img_shape=image.shape, is_pose2d=info['vmask_2d'][:,0])

        all_person_detected_mask = info['vmask_2d'][0,2]
        subject_ids, valid_masks[:,2] = self.process_suject_ids(info['track_ids'], used_person_inds, valid_mask_ids=info['vmask_2d'][:,1])
        image, dst_image, org_image = self.prepare_image(image, image_wbg, augments=pixel_augments)

        # valid mask of 3D pose, smpl root rot, smpl pose param, smpl shape param, global translation
        kp3d, valid_masks[:,1] = self.process_kp3ds(info['kp3ds'], used_person_inds, \
            augments=position_augments, valid_mask_kp3ds=info['vmask_3d'][:, 0])
        params, valid_masks[:,3:6] = self.process_smpl_params(info['params'], used_person_inds, \
            augments=position_augments, valid_mask_smpl=info['vmask_3d'][:, 1:4])
        verts_processed, valid_masks[:,7], root_trans_processed, valid_masks[:,6] = self.process_verts(info['verts'], info['root_trans'], used_person_inds, \
            augments=position_augments, valid_mask_verts=info['vmask_3d'][:, 4], valid_mask_depth=info['vmask_3d'][:, 5])
        
        rot_flip = np.array([position_augments[0], position_augments[1]]) if position_augments is not None else np.array([0,0])

        input_data = {
            'image': torch.from_numpy(dst_image).float(),
            'image_org': torch.from_numpy(org_image),
            'full_kp2d': torch.from_numpy(full_kp2ds).float(),
            'person_centers':torch.from_numpy(person_centers).float(), 
            'subject_ids':torch.from_numpy(subject_ids).long(),
            'centermap': centermap.float(),
            'kp_3d': torch.from_numpy(kp3d).float(),
            'verts': torch.from_numpy(verts_processed).float(),
            'params': torch.from_numpy(params).float(),
            'valid_masks':torch.from_numpy(valid_masks).bool(),
            'root_trans': torch.from_numpy(root_trans_processed).float(),
            'offsets': torch.from_numpy(offsets).float(),
            'rot_flip': torch.from_numpy(rot_flip).float(),
            'all_person_detected_mask':torch.Tensor([all_person_detected_mask]).bool(),
            'imgpath': info['imgpath'],
            'data_set': info['ds']}
        input_data = self.add_cam_parameters(input_data, info)

        if self.train_flag:
            img_scale = 1 if position_augments is None else position_augments[3]
            input_data['img_scale'] = torch.Tensor([img_scale]).float()

        if args().learn_2dpose:
            input_data.update({'heatmap':torch.from_numpy(heatmap).float()})
        if args().learn_AE:
            input_data.update({'AE_joints': torch.from_numpy(AE_joints).long()})

        if args().perspective_proj:
            root_trans, cam_params, cam_mask = self._calc_normed_cam_params_(full_kp2ds, kp3d, valid_masks[:, 1], info['ds'])
            centermap_3d, valid_centermap3d_mask = self.generate_centermap_3d(person_centers, cam_params, cam_mask, bboxes_hw_norm, all_person_detected_mask)
            input_data.update({'cams': cam_params,'cam_mask': cam_mask, 'root_trans':root_trans,\
                'centermap_3d':centermap_3d.float(),'valid_centermap3d_mask':torch.Tensor([valid_centermap3d_mask]).bool()})

        if args().learn_relative:
            # age; gender; depth level; body type
            depth_info = self._organize_depth_info_(info, used_person_inds)
            input_data.update({'depth_info': torch.from_numpy(depth_info).long()})
            input_data['kid_shape_offsets'] = self._organize_kid_shape_offsets_(info, used_person_inds)
            
        return input_data

    def generate_dynamic_depth(self, cam_params, cam_mask, bboxes_hw_norm, all_person_detected_mask):
        radius_list = torch.Tensor(_calc_radius_(bboxes_hw_norm)).long()
        person_scales = torch.zeros(self.max_person).long()
        person_num = min(self.max_person, len(radius_list))
        person_scales[:person_num] = radius_list[:person_num]
        return person_scales

    def _organize_depth_info_(self, info, used_person_inds):
        prepared_info = np.ones((self.max_person, 4))*-1
        if 'depth' in info:
            if len(info['depth']) != 0:
                prepared_info[:len(used_person_inds)] = np.array(info['depth'])[used_person_inds]
        return prepared_info

    def _organize_kid_shape_offsets_(self, info, used_person_inds):
        kid_shape_offsets_processed = torch.ones(self.max_person).float()*-1
        if 'kid_shape_offsets' in info:
            kid_shape_offsets_processed[:len(used_person_inds)] = torch.from_numpy(info['kid_shape_offsets'][used_person_inds]).float()
        return kid_shape_offsets_processed


    def __getitem__(self, index):
        return self.get_item_single_frame(index)
        try:
            return self.get_item_single_frame(index)
        except Exception as error:
            logging.error(error)
            index = np.random.randint(len(self))
            return self.get_item_single_frame(index)


name_dict = {
    'depth_id': {0: '最前排', 1: '第二排', 2: '第三排', 3: '第四排', 4: '第五排', 5: '第六排', 6: '第七排', 7: '第八排', 8: '第九排', 9: '第十排', -1: '深度不明'},
    'age': {0: '成年', 1: '青少年', 2: '小孩', 3:'婴幼儿', -1: '年龄不明'},
    'body_type': {0: '正常', 1: '微胖', 2: '胖', 3: '强壮'},
    'occluded_by_others': {0: '无遮挡', 1: '遮挡'},
    'gender': {0: '男', 1: '女', -1:'性别不明'}
}

def visualize_3d_hmap(hmap,save_name):
    if not (type(hmap) is np.ndarray):
        try:
            hmap = hmap.cpu().numpy()
        except:
            hmap = hmap.detach().cpu().numpy()

    hmap[hmap < 0] = 0
    hmap[hmap > 1] = 1
    hmap = (hmap * 255).astype(np.uint8)
    for d, x in enumerate(hmap):
        x = cv2.applyColorMap(x, colormap=cv2.COLORMAP_JET)
        x = cv2.putText(x, f'{d}', (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 128, 128), 2, cv2.LINE_AA)
        cv2.imwrite(save_name+'_{}.jpg'.format(d), x)

def test_image_relative_dataset(dataset,with_3d=False,with_smpl=False):
    print('testing relative dataset loading')
    print('configs_yml:', args().configs_yml)
    print('model_version:',args().model_version)
    
    from visualization.visualization import Visualizer
    test_projection_part = True if args().model_version in [4,5,6,7] else False
    print('test_projection_part:',test_projection_part)

    save_dir = os.path.join(config.project_dir,'test')
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)
    print('Initialized dataset')

    batch_size, model_type= 2, 'smpl'
    dataloader = DataLoader(dataset = dataset,batch_size = batch_size,shuffle = True,\
        drop_last = False,pin_memory = True,num_workers = 1)
    visualizer = Visualizer(resolution = (512,512,3), result_img_dir=save_dir,with_renderer=True)
    print('Initialized visualizer')

    from visualization.visualization import make_heatmaps, draw_skeleton_multiperson
    from utils.cam_utils import denormalize_cam_params_to_trans
    if with_smpl:
        from smpl_family.smpl import SMPL
        smpl = SMPL(args().smpl_model_path, model_type='smpl')
        from smpl_family.smpla import SMPLA_parser
        smpl_family = SMPLA_parser(args().smpla_model_path, args().smil_model_path, baby_thresh=args().baby_threshold)
    print('Initialized SMPL models')

    img_size = 512
    bones, cm = constants.All54_connMat, constants.cm_All54
    print('Start loading data.')
    for _,r in enumerate(dataloader):
        if _%100==0:
            for key,value in r.items():
                if isinstance(value,torch.Tensor):
                    print(key,value.shape)
                elif isinstance(value,list):
                    print(key,len(value))
        for inds in range(2):
            img_bsname = os.path.basename(r['imgpath'][inds])
            image = r['image'][inds].numpy().astype(np.uint8)[:,:,::-1]

            full_kp2d = (r['full_kp2d'][inds].numpy() + 1) * img_size / 2.0
            person_centers = (r['person_centers'][inds].numpy() + 1) * img_size / 2.0
            subject_ids = r['subject_ids'][inds]
            image_kp2d = visualizer.draw_skeleton_multiperson(image.copy(), full_kp2d, bones=bones, cm=cm)
            
            if test_projection_part and r['cam_mask'][inds].sum()>0:
                cam_mask = r['cam_mask'][inds]
                kp3d_tp = r['kp_3d'][inds][cam_mask].clone()
                kp2d_tp = r['full_kp2d'][inds][cam_mask].clone()
                pred_cam_t = denormalize_cam_params_to_trans(r['cams'][inds][cam_mask].clone())
                
                pred_keypoints_2d = perspective_projection(kp3d_tp,translation=pred_cam_t,focal_length=args().focal_length, normalize=False)+512//2
                invalid_mask = np.logical_or(kp3d_tp[:,:,-1]==-2., kp2d_tp[:,:,-1]==-2.)
                pred_keypoints_2d[invalid_mask] = -2.
                image_kp2d_projection = visualizer.draw_skeleton_multiperson(image.copy(), pred_keypoints_2d, bones=bones, cm=cm)
                cv2.imwrite('{}/{}_{}_projection.jpg'.format(save_dir,_,img_bsname), image_kp2d_projection)

            for pinds, (person_center, subject_id) in enumerate(zip(person_centers,subject_ids)):
                y,x = person_center.astype(np.int)
                if y>0 and x>0:
                    cv2.circle(image_kp2d, (x,y), 6, [0,0,255],-1)
                    text = '{}'.format(subject_id)
                    if 'depth_info' in r:
                        depth_info = r['depth_info'][inds][pinds]
                        text+= '{}'.format(depth_info.numpy().tolist())
                        #'\n {}\n {}\n {}\n {}\n'.format(name_dict['depth_id'][depth_info[0].item()], name_dict['age'][depth_info[1].item()], \
                        #    name_dict['body_type'][depth_info[2].item()], name_dict['gender'][depth_info[3].item()])
                    #cv2.putText(image_kp2d, text, (x,y),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),1)            
          
            centermap_color = make_heatmaps(image.copy(), r['centermap'][inds])
            image_vis = np.concatenate([image_kp2d, centermap_color],1)
            cv2.imwrite('{}/{}_{}_centermap.jpg'.format(save_dir,_,img_bsname), image_vis)
            if 'heatmap' in r:
                heatmap_color = make_heatmaps(image.copy(), r['heatmap'][inds])
                cv2.imwrite('{}/{}_{}_heatmap.jpg'.format(save_dir,_,img_bsname), heatmap_color)

            # if 'valid_centermap3d_mask' in r:
            #     for rind, c3d_mask in enumerate(r['valid_centermap3d_mask']):
            #         if c3d_mask:
            #             centermap_3d = r['centermap_3d'][rind]
            #             visualize_3d_hmap(centermap_3d, '{}/{}_{}'.format(save_dir, _, rind))

            person_centers_onmap = ((r['person_centers'][inds].numpy() + 1)/ 2.0 * (args().centermap_size-1)).astype(np.int)
            positive_position = torch.stack(torch.where(r['centermap'][inds,0]==1)).permute(1,0)

        if with_smpl and r['valid_masks'][0,0,4]:
            params, subject_ids = r['params'][0],  r['subject_ids'][0]
            image = r['image'][0].numpy().astype(np.uint8)[:,:,::-1]
            valid_mask = torch.where(r['valid_masks'][0,:,4])[0]
            subject_ids = subject_ids[valid_mask]
            pose = params[valid_mask][:,:66].float()
            if r['valid_masks'][0,valid_mask,5].sum()>0:
                betas = params[valid_mask][:,-10:].float()
            else:
                betas = torch.zeros(len(pose),10)
            pose = torch.cat([pose, torch.zeros(len(pose),6)],-1).float()
            if 'kid_shape_offsets' in r:
                kso_vmask = r['kid_shape_offsets'][0][valid_mask]!=-1
                betas = torch.cat([betas, torch.zeros(len(betas), 1)], 1)
                if (kso_vmask).sum()>0:
                    betas[kso_vmask,-1] = r['kid_shape_offsets'][0][valid_mask][kso_vmask]

                verts,joints = smpl_family(poses=pose, betas=betas)
                print('using kid shape offset to create mesh')
            else:
                verts,joints = smpl(poses=pose, betas=betas, get_skin = True)

            if test_projection_part and r['cam_mask'][inds].sum()>0:
                if r['cam_mask'][0][valid_mask].sum()>0:
                    trans = denormalize_cam_params_to_trans(r['cams'][0][valid_mask].clone(), positive_constrain=True)
                else:
                    trans = r['root_trans'][0][valid_mask]  

                pred_keypoints_2d = perspective_projection(joints,translation=trans,focal_length=args().focal_length, normalize=False)+512//2
   
                render_img = visualizer.visualize_renderer_verts_list([verts.cuda()], trans=[trans.cuda()], images=image[None])[0]
                rendered_img_bv = visualizer.visualize_renderer_verts_list([verts.cuda()], trans=[trans.cuda()], bird_view=True, auto_cam=True)[0]
                person_centers = (r['person_centers'][0].numpy() + 1) * img_size / 2.0
                image = np.array(image).astype(np.uint8)
                if len(person_centers) == len(trans):
                    for pinds, person_center in enumerate(person_centers):
                        y,x = person_center.astype(np.int)
                        if y>0 and x>0:
                            cv2.circle(np.array(image), (x,y), 6, [0,0,255],-1)
                            text = '{:.2f}'.format(trans[pinds,2])
                            print((x,y), text)
                            cv2.putText(image, text, (x,y), cv2.FONT_HERSHEY_COMPLEX,3,(255,0,255),3)     
                cv2.imwrite('{}/mesh_{}.png'.format(save_dir,_), np.concatenate([image, render_img, rendered_img_bv],1)) #image_kp2d_projection_smpl24, image_kp2d_projection_extra30
            else:
                verts[:,:,2] += 5
                render_img = visualizer.visualize_renderer_verts_list([verts.cuda()], images=image[None])[0]
                cv2.imwrite('{}/mesh_{}.png'.format(save_dir,_), render_img)
        j3ds = r['kp_3d'][0,0]
        image = r['image'][0].numpy().astype(np.uint8)[:,:,::-1]
        if r['valid_masks'][0,0,1]:
            pj2d = (j3ds[:,:2] + 1) * img_size / 2.0
            pj2d[j3ds[:,-1]==-2.] = -2.
            image_pkp3d = visualizer.draw_skeleton(image.copy(), pj2d, bones=bones, cm=cm)
            cv2.imwrite('{}/pkp3d_{}_{}.png'.format(save_dir,_,r['subject_ids'][0, 0]), image_pkp3d)
