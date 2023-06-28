from base import *
from utils.demo_utils import video2frame
import glob
import os
import joblib
import lap

from tracker.byte_tracker_3dcenter import Tracker
from visualization.BEV_visualizer import Renderer, convert_front_view_to_bird_view_video


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def seq_traj_rids2track_ids(seq_traj_rids, pred_num):
    track_ids = torch.ones(pred_num).long()* -1
    for pred_id in range(pred_num):
        track_id = torch.where(seq_traj_rids == pred_id)[0]
        if len(track_id)>0:
            track_ids[pred_id] = track_id
    return track_ids


def combining_motion_offsets_and_detection_to_trajectory(outputs, sequence_length, add_offsets=True):
    # very like a tracker that use estimated motion_offsets to replace the effect of kalman filter.
    motion_offsets, cam_preds, clip_frame_ids = outputs['motion_offsets'].detach().cpu(), outputs['cam'].detach().cpu(), outputs['reorganize_idx'].cpu()
    if add_offsets:
        cam_offset_preds = cam_preds + motion_offsets
    else:
        cam_offset_preds = cam_preds
    # take the entire sequence in one batch
    result_inds = torch.arange(len(motion_offsets))

    # possible max_subject_num is larger than max frame_detection_num, bacause people go in and out.
    #frame_detection_num = [
    #    (clip_frame_ids==seq_fid).sum() for seq_fid in torch.unique(clip_frame_ids)]
    #max_subject_num = max(frame_detection_num)
    max_subject_num = 160
    subject_num_now = 0
    alert_num = 80
    add_num = 100

    seq_traj_cams = torch.ones(max_subject_num, sequence_length, 3) * -2.
    seq_traj_rids = torch.ones(max_subject_num, sequence_length).long() * -1

    forget_time = 24
    matching_thresh = 1.4 #1.
   
    for fid in range(sequence_length):
        current_fid_mask = clip_frame_ids == fid
        if current_fid_mask.sum() == 0:
            continue
        
        # get the index of latest valid cams of each subjects
        valid_inds = [[],[]]
        
        for pc_ind in range(subject_num_now):
            previous_inds = torch.where(seq_traj_rids[pc_ind, :fid]!=-1)[0]
            
            if len(previous_inds) > 0:
                latest_ind = previous_inds.max()
                if (fid-latest_ind)<forget_time:
                    valid_inds[0].append(pc_ind)
                    valid_inds[1].append(latest_ind)
        
        if len(valid_inds[0]) == 0:
            seq_traj_cams[:current_fid_mask.sum(), fid] = cam_offset_preds[current_fid_mask]
            seq_traj_rids[:current_fid_mask.sum(), fid] = result_inds[current_fid_mask]
            subject_num_now += current_fid_mask.sum()
        else:
            previous_cams = seq_traj_cams[valid_inds[0], valid_inds[1]]
            current_cams = cam_preds[current_fid_mask]

            cost_distance = np.linalg.norm(
                previous_cams.numpy()[:, None]-current_cams.numpy()[None], ord=2, axis=2)
            cost_time = (-np.array(valid_inds[1])[:, None] + (np.ones([len(current_cams)])*fid)[None] - 1) 
            cost_matrix = cost_distance + cost_time / forget_time * matching_thresh
            #print(fid, 'cost_matrix:', cost_matrix.shape,  cost_matrix.min(1))
            matched, unmatched_pre, unmatched_cur = linear_assignment(cost_matrix, thresh=matching_thresh)
            
            if len(matched)>0:
                seq_traj_cams[np.array(valid_inds[0])[
                    matched[:, 0]], fid] = cam_offset_preds[current_fid_mask][matched[:, 1]]
                seq_traj_rids[np.array(valid_inds[0])[
                    matched[:, 0]], fid] = result_inds[current_fid_mask][matched[:, 1]]
            
            left_detection_num = len(unmatched_cur)
            seq_traj_cams[subject_num_now:subject_num_now+left_detection_num, fid] = cam_offset_preds[current_fid_mask][unmatched_cur]
            seq_traj_rids[subject_num_now:subject_num_now+left_detection_num, fid] = result_inds[current_fid_mask][unmatched_cur]
            subject_num_now += left_detection_num
        
        if (max_subject_num-subject_num_now) < alert_num:
            seq_traj_cams = torch.cat([seq_traj_cams, torch.ones(add_num, sequence_length, 3) * -2.], 0)
            seq_traj_rids = torch.cat([seq_traj_rids, torch.ones(add_num, sequence_length).long() * -1], 0)
            max_subject_num += add_num
    valid_seq_traj_rids = seq_traj_rids[:subject_num_now]
    track_ids = seq_traj_rids2track_ids(valid_seq_traj_rids, len(cam_preds))
    return valid_seq_traj_rids, track_ids
    
    

def combining_motion_offsets_and_detection_to_trajectory_batch(outputs, add_offsets=True):
    # very like a tracker that use estimated motion_offsets to replace the effect of kalman filter.
    motion_offsets, cam_preds, batch_ids = outputs['motion_offsets'].detach().cpu(), outputs['cam'].detach().cpu(), outputs['reorganize_idx'].cpu()
    if add_offsets:
        cam_offset_preds = cam_preds + motion_offsets
    else:
        cam_offset_preds = cam_preds

    seq_ids = batch_ids // args().temp_clip_length
    clip_frame_ids = batch_ids % args().temp_clip_length
    result_inds = torch.arange(len(motion_offsets))
    batch_seq_traj_rids = []
    for seq_id in torch.unique(seq_ids):
        seq_mask = seq_ids == seq_id
        seq_cam = cam_preds[seq_mask]
        seq_cam_off = cam_offset_preds[seq_mask]
        seq_frame_ids = clip_frame_ids[seq_mask]
        seq_result_inds = result_inds[seq_mask]

        frame_detection_num = [
            (seq_frame_ids==seq_fid).sum() for seq_fid in torch.unique(seq_frame_ids)]
        max_subject_num = max(frame_detection_num)

        seq_traj_cams = torch.ones(max_subject_num, args().temp_clip_length, 3) * -2.
        seq_traj_rids = torch.ones(max_subject_num, args().temp_clip_length).long() * -1

        seq_traj_cams[:(seq_frame_ids == 0).sum(),
                      0] = seq_cam_off[seq_frame_ids == 0]
        seq_traj_rids[:(seq_frame_ids == 0).sum(),
                      0] = seq_result_inds[seq_frame_ids == 0]
        
        for fid in range(1, args().temp_clip_length):
            current_fid_mask = seq_frame_ids == fid
            if current_fid_mask.sum() == 0:
                continue
            
            # get the index of latest valid cams of each subjects
            valid_inds = [[],[]]
            invalid_inds = []
            for pc_ind in range(max_subject_num):
                previous_inds = torch.where(seq_traj_rids[pc_ind, :fid]!=-1)[0]
                if len(previous_inds) > 0:
                    valid_inds[0].append(pc_ind)
                    valid_inds[1].append(previous_inds.max())
                else:
                    invalid_inds.append(pc_ind)
            if len(valid_inds[0]) == 0:
                seq_traj_cams[:current_fid_mask.sum(), fid] = seq_cam_off[current_fid_mask]
                seq_traj_rids[:current_fid_mask.sum(), fid] = seq_result_inds[current_fid_mask]
            else:
                previous_cams = seq_traj_cams[valid_inds[0], valid_inds[1]]
                current_cams = seq_cam[current_fid_mask]

                cost_matrix = np.linalg.norm(
                    previous_cams.numpy()[:, None]-current_cams.numpy()[None], ord=2, axis=2)
                print('cost_matrix:', cost_matrix)
                matched, unmatched_pre, unmatched_cur = linear_assignment(cost_matrix, thresh=100)
                
                seq_traj_cams[np.array(valid_inds[0])[
                    matched[:, 0]], fid] = seq_cam_off[current_fid_mask][matched[:, 1]]
                seq_traj_rids[np.array(valid_inds[0])[
                    matched[:, 0]], fid] = seq_result_inds[current_fid_mask][matched[:, 1]]
                
                # for the unmatched current new detection, it is because no previous detection, just put them in invalid_inds
                for um_ind, um_cur_id in enumerate(unmatched_cur):
                    seq_traj_cams[invalid_inds[um_ind],
                                  fid] = seq_cam_off[current_fid_mask][um_cur_id]
                    seq_traj_rids[invalid_inds[um_ind],
                                  fid] = seq_result_inds[current_fid_mask][um_cur_id]
        batch_seq_traj_rids.append(seq_traj_rids)
    return batch_seq_traj_rids

def pj2ds_to_bbox(pj2ds):
    tracked_bbox = np.array([pj2ds[:,0].min(), pj2ds[:,1].min(), pj2ds[:,0].max(), pj2ds[:,1].max()])
    # left, top, right, down -> left, top, width, height
    tracked_bbox[2:] = tracked_bbox[2:] - tracked_bbox[:2]
    return tracked_bbox

def collect_sequence_tracking_results(outputs, img_paths, reorganize_idx, visualize_results=False):
    track_ids = outputs['track_ids'].numpy()
    pj2d_org_results = outputs['pj2d_org'].detach().cpu().numpy().astype(np.float16)
    #center_confs = outputs['center_confs'].detach().cpu().numpy().astype(np.float16)

    tracking_results = {}
    for frame_id, img_path in enumerate(img_paths):
        pred_ids = np.where(reorganize_idx==frame_id)[0]
        img_name = os.path.basename(img_path)
        tracking_results[img_name] = {'track_ids':[], 'track_bbox':[], 'pj2ds':[]}
        for batch_id in pred_ids:
            track_id = track_ids[batch_id]
            pj2d_org = pj2d_org_results[batch_id]
            bbox = pj2ds_to_bbox(pj2d_org)
            tracking_results[img_name]['track_ids'].append(track_id)
            tracking_results[img_name]['track_bbox'].append(bbox)
            tracking_results[img_name]['pj2ds'].append(pj2d_org)
            #results[img_name][track_id]['center_conf'] = center_confs[batch_id]
        if visualize_results:
            vis_track_bbox(img_path, tracking_results[img_name]['track_ids'], tracking_results[img_name]['track_bbox'])
    return tracking_results

def vis_track_bbox(image_path, tracked_ids, tracked_bbox):
    org_img = cv2.imread(image_path)
    for tid, bbox in zip(tracked_ids, tracked_bbox):
        org_img = cv2.rectangle(org_img, tuple(bbox[:2]), tuple(bbox[2:]+bbox[:2]), (255,0,0), 3)
        org_img = cv2.putText(org_img, "{}".format(tid), tuple(bbox[:2]), cv2.FONT_HERSHEY_COMPLEX, 3, (0,255,0),2)
    h,w = org_img.shape[:2]
    cv2.imshow('bbox', cv2.resize(org_img, (w//2, h//2)))
    cv2.waitKey(10)

def save_tracking_results(tracking_results, tracking_save_folder, visualize_results=False):
    for video_folder, seq_results in tracking_results.items():
        video_name = video_folder.split(os.path.sep)[-2]
        video_tracking_save_path = os.path.join(tracking_save_folder, video_name + ".pkl")
        
        if os.path.exists(video_tracking_save_path):
            print(video_name, 'already processed')
            continue
        sorted_results_path = sorted(list(seq_results.keys()))
        video_result_dict = {}
        for fid, frame_name in enumerate(sorted_results_path):
            tracked_ids = seq_results[frame_name]['track_ids']
            tracked_bbox = seq_results[frame_name]['track_bbox']
            
            if visualize_results:
                image_path = os.path.join(video_folder, frame_name)
                org_img = cv2.imread(image_path)
                for bbox, tid in zip(tracked_bbox,tracked_ids):
                    org_img = cv2.rectangle(org_img, tuple(bbox[:2]), tuple(bbox[2:]+bbox[:2]), (255,0,0), 3)
                    org_img = cv2.putText(org_img, "{}".format(tid), tuple(bbox[:2]), cv2.FONT_HERSHEY_COMPLEX, 3, (0,255,0),2)
                cv2.imshow('bbox', org_img)
                cv2.waitKey(600)
                
            video_result_dict[frame_name] = [tracked_ids, tracked_bbox]
        #print('save to ',video_tracking_save_path)
        joblib.dump(video_result_dict, video_tracking_save_path)
    return tracking_save_folder

class Demo(Base):
    def __init__(self):
        super(Demo, self).__init__()
        self._build_model_()
        self.test_cfg = {'mode':'parsing', 'calc_loss': False,'with_nms':True,'new_training': args().new_training}
        self.eval_dataset = args().eval_dataset
        self.save_mesh = False
        self.pyrender_render = Renderer(self.visualizer.smpl_face[0], focal_length=args().focal_length*2,height=1024,width=1024)
        self.pyrender_render_bv = Renderer(self.visualizer.smpl_face[0], focal_length=1000,height=2048,width=2048)
        #self.vedo_vis = Vedo_visualizer()
        print('Initialization finished!')

    def net_forward(self,meta_data,mode='val'):
        if mode=='val':
            cfg_dict = self.test_cfg
        elif mode=='eval':
            cfg_dict = self.eval_cfg
        ds_org, imgpath_org = get_remove_keys(meta_data,keys=['data_set','imgpath'])
        meta_data['batch_ids'] = torch.arange(len(meta_data['image']))
        if self.model_precision=='fp16':
            with autocast():
                outputs = self.model(meta_data, **cfg_dict)
        else:
            outputs = self.model(meta_data, **cfg_dict)
        meta_data['data_set'], meta_data['imgpath'] = ds_org, imgpath_org
        outputs['meta_data']['data_set'], outputs['meta_data']['imgpath'] = reorganize_items([ds_org, imgpath_org], outputs['reorganize_idx'].cpu().numpy())
        return outputs

    def process_demo(self, seq_folder_list, seq_save_paths):
        self.model.eval()
        seq_save_npzs = []
        for seq_id, (seq_folder, seq_save_dir) in enumerate(zip(seq_folder_list, seq_save_paths)):
            os.makedirs(seq_save_dir,exist_ok=True)
            file_list = glob.glob(os.path.join(seq_folder, '*.jpg'))
            result_save_path = seq_save_dir+'.npz'
            seq_save_npzs.append(result_save_path)

            if os.path.exists(result_save_path):
                continue

            data_loader = self._create_single_data_loader(dataset='internet', file_list=file_list)
            seq_results_dict = {}
            
            with torch.no_grad():
                for test_iter,meta_data in enumerate(data_loader):
                    seq_save_dirs = [seq_save_dir for _ in range(len(meta_data['image']))]
                    try:
                        outputs = self.net_forward(meta_data, mode='val')
                    except:
                        continue
                    results = reorganize_results(outputs, outputs['meta_data']['imgpath'], outputs['reorganize_idx'].cpu().numpy())
                    seq_results_dict = {**seq_results_dict, **results}

                    if self.visualize_all_results:
                        rendering_mesh_to_image(self, outputs, seq_save_dirs)
                    if test_iter % 6 == 0:
                        print('{} / {}'.format(test_iter, len(data_loader)))
            
            np.savez(result_save_path, results=seq_results_dict)
            
        return seq_save_npzs

    def perform_tracking(self, seq_result_paths, tracking_save_folder, eval_tool='motmetrics'):
        os.makedirs(tracking_save_folder,exist_ok=True)

        depth2image_scale = 60

        for seq_result_path in seq_result_paths:
            seq_result_dict = np.load(seq_result_path, allow_pickle=True)['results'][()]
            frame_paths = sorted(list(seq_result_dict.keys()))
            video_result_dict = {}

            if self.mp_tracker == 'norfair':
                distance_threshold = 300 # 60 # 10
                tracker = Tracker(distance_function=euclidean_distance, distance_threshold=distance_threshold)
            elif self.mp_tracker == 'byte':
                tracker = Tracker()
            
            for fid, image_path in enumerate(frame_paths):
                frame_results = seq_result_dict[image_path]
                org_img = cv2.imread(image_path)
                frame_name = os.path.basename(image_path)
                cam_trans = np.array([frame_results[i]['cam_trans'] for i in range(len(frame_results))])
                pj2d_orgs = np.array([frame_results[i]['pj2d_org'] for i in range(len(frame_results))])
                det_confs = np.array([frame_results[i]['center_conf'] for i in range(len(frame_results))])

                tracked_bbox = np.array([[pj2d_orgs[i,:,0].min(), pj2d_orgs[i,:,1].min(), \
                        pj2d_orgs[i,:,0].max()-pj2d_orgs[i,:,0].min(), pj2d_orgs[i,:,1].max()-pj2d_orgs[i,:,1].min()] for i in range(len(pj2d_orgs))])

                #print('tracking_points:', tracking_points)
                if self.mp_tracker == 'norfair':
                    tracking_points = np.zeros((len(cam_trans), 2, 2))
                    for cp_id, (tran, pj2d, bbox) in enumerate(zip(cam_trans, pj2d_orgs, tracked_bbox)):
                        # use the 0-th center kp + depth + bbox height for tracking
                        tracking_points[cp_id] = np.array([pj2d[0], [tran[2]*depth2image_scale, bbox[3]]])
                    detections = [Detection(points=points)
                                  for points in tracking_points]
                    if fid == 0:
                        for _ in range(8):
                            tracked_objects = tracker.update(detections=detections)
                    tracked_objects = tracker.update(detections=detections)
                    if len(tracked_objects) == 0:
                        continue
                    tracked_ids, tracked_bbox_ids = get_tracked_ids(
                        detections, tracked_objects)
                    
                elif self.mp_tracker == 'byte':
                    tracking_points = np.zeros((len(cam_trans),4))
                    for cp_id, (tran, pj2d, bbox) in enumerate(zip(cam_trans, pj2d_orgs, tracked_bbox)):
                        # use the 0-th center kp + depth for tracking
                        tracking_points[cp_id] = np.array([pj2d[0][0], pj2d[0][1], tran[2]*depth2image_scale, bbox[3]])

                    #print('det_confs:', det_confs)
                    #print('tracking_points', tracking_points)
                    tracked_objects = tracker.update(
                        tracking_points, det_confs)
                    #print('byte tracking outputs:',tracked_objects)
                    if len(tracked_objects) == 0:
                        continue
                    tracked_ids, tracked_bbox_ids = get_tracked_ids_byte(
                        tracking_points, tracked_objects)
                
                tracked_bbox = tracked_bbox[tracked_bbox_ids]
                    
                for bbox, tid in zip(tracked_bbox,tracked_ids):
                    org_img = cv2.rectangle(org_img, tuple(bbox[:2]), tuple(bbox[2:]+bbox[:2]), (255,0,0), 3)
                    org_img = cv2.putText(org_img, "{}".format(tid), tuple(bbox[:2]), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0),2)
                print(org_img.shape)
                cv2.imshow('bbox', org_img)
                cv2.waitKey(10)

                video_result_dict[frame_name] = [tracked_ids, tracked_bbox]
            video_name = os.path.basename(seq_result_path).replace('.npz', '')
            joblib.dump(video_result_dict, os.path.join(tracking_save_folder, video_name + ".pkl"))
        
    @staticmethod
    def toframe(video_file_path):
        assert isinstance(video_file_path, str), \
            print('We expect the input video file path is str, while recieved {}'.format(video_file_path))
        video_basename, video_ext = os.path.splitext(video_file_path)
        assert video_ext in constants.video_exts, \
            print('Video format {} is not currently supported, please convert it to the frames by yourself.'.format(video_ext))
        frame_list = video2frame(video_file_path, frame_save_dir=video_basename+'_frames')
        return video_basename, frame_list

    def visualize_result_meshes(self, outputs, meta_data, mesh_colors=None, put_org=True, drop_texts=False):
        used_org_inds, per_img_inds = process_idx(outputs['reorganize_idx'])
        img_inds_org = [inds[0] for inds in per_img_inds]
        img_names = np.array(meta_data['imgpath'])[img_inds_org]
        org_imgs = meta_data['image'].cpu().numpy().astype(np.uint8)[img_inds_org]
        rendered_imgs = self.visualizer.show_verts_on_imgs(outputs, meta_data, (used_org_inds, per_img_inds, img_inds_org), \
                                        org_imgs, img_names=img_names, put_org=put_org, drop_texts=drop_texts, mesh_colors=mesh_colors)
        return rendered_imgs, img_names

    def get_center_preds(self, outputs):
        # get the center location of the small-scale people
        if self.model_version==7:
            center_preds = outputs['center_preds'].detach().cpu().numpy()
        elif self.model_version==6:
            center_preds = outputs['center_preds'].detach().cpu().numpy()[:,[2,1]]
        return center_preds

def reorganize_results(outputs, img_paths, reorganize_idx):
    results = {}
    cam_results = outputs['cam_trans'].detach().cpu().numpy().astype(np.float16)
    pj2d_org_results = outputs['pj2d_org'].detach().cpu().numpy().astype(np.float16)
    center_confs = outputs['center_confs'].detach().cpu().numpy().astype(np.float16)

    vids_org = np.unique(reorganize_idx)
    for idx, vid in enumerate(vids_org):
        verts_vids = np.where(reorganize_idx==vid)[0]
        img_path = img_paths[verts_vids[0]]                
        results[img_path] = [{} for idx in range(len(verts_vids))]
        for subject_idx, batch_idx in enumerate(verts_vids):
            results[img_path][subject_idx]['cam_trans'] = cam_results[batch_idx]
            results[img_path][subject_idx]['pj2d_org'] = pj2d_org_results[batch_idx]
            results[img_path][subject_idx]['center_conf'] = center_confs[batch_idx]
    return results

color_list = np.array([[.7, .7, .6],[.7, .5, .5],[.5, .5, .7],  [.5, .55, .3],[.3, .5, .55],  \
 [1,0.855,0.725],[0.588,0.804,0.804],[1,0.757,0.757],  [0.933,0.474,0.258],[0.847,191/255,0.847],  [0.941,1,1]])

def rendering_mesh_to_image(self, outputs, seq_save_dirs):
    used_org_inds, per_img_inds = process_idx(outputs['reorganize_idx'])
    seq_save_dirs = [seq_save_dirs[ind] for ind in used_org_inds]
    mesh_colors = torch.Tensor([color_list[idx%len(color_list)] for idx in range(len(outputs['reorganize_idx']))])
    if args().model_version == 1:
        predicts_j3ds = outputs['j3d'].contiguous().detach().cpu().numpy()
        predicts_pj2ds = (outputs['pj2d'].detach().cpu().numpy()+1)*256
        predicts_j3ds = predicts_j3ds[:,:24]
        predicts_pj2ds = predicts_pj2ds[:,:24]
        outputs['cam_trans'] = estimate_translation(predicts_j3ds, predicts_pj2ds, \
                focal_length=args().focal_length, img_size=np.array([512,512]),pnp_algorithm='cv2').to(outputs['verts'].device)

    #verts_whole_image = (outputs['verts']+outputs['cam_trans'].unsqueeze(1)).detach()#.cpu().numpy()
    #faces_list = self.visualizer.smpl_face.repeat(len(verts_whole_image), 1, 1).to(verts_whole_image.device)
    #verts_bv = convert_front_view_to_bird_view(verts_whole_image)
    #verts_sv = convert_front_view_to_side_view(verts_whole_image)

    img_orgs = [outputs['meta_data']['image_1024'][img_id].cpu().numpy().astype(np.uint8) for img_id in range(len(per_img_inds))]
    img_verts = [outputs['verts'][inds] for inds in per_img_inds]
    img_trans = [outputs['cam_trans'][inds] for inds in per_img_inds]
    img_verts_bv = [convert_front_view_to_bird_view_video((iverts+itrans.unsqueeze(1)).detach(), None) for iverts, itrans in zip(img_verts, img_trans)]
    #img_verts_sv = [convert_front_view_to_side_view_video((iverts+itrans.unsqueeze(1)).detach(), None) for iverts, itrans in zip(img_verts, img_trans)]
    img_names = [np.array(outputs['meta_data']['imgpath'])[inds[0]] for inds in per_img_inds]
    
    for batch_idx, img_org in enumerate(img_orgs):
        try:
            img_org = img_org[:,:,::-1]
            rendered_img = self.pyrender_render(img_org[None],[img_verts[batch_idx]], [img_trans[batch_idx]])[0]
            result_image_bv = self.pyrender_render_bv([np.ones_like(img_org)*255], [img_verts_bv[batch_idx]], [torch.zeros_like(img_trans[batch_idx])])[0]
            render_fv = rendered_img.transpose((1,2,0))
            render_bv = result_image_bv.transpose((1,2,0))
            save_path = os.path.join(seq_save_dirs[batch_idx], os.path.basename(img_names[batch_idx]))
            img_results = np.concatenate([img_org, render_fv, np.ones((1024,1024,3))*255], 1)
            img_results[256:256+512, 1024*2:1024*2+512] = cv2.resize(render_bv, (512,512))
            img_results = img_results[200:-200]
            img_results = img_results[:,:-512]
            cv2.imwrite(save_path,img_results)
        except Exception as error:
            print(error)

def euclidean_distance(detection, tracked_object):
    dist = np.linalg.norm(detection.points - tracked_object.estimate)
    return dist

def get_tracked_ids(detections, tracked_objects):
    tracked_ids_out = np.array([obj.id for obj in tracked_objects])
    tracked_points = np.array([obj.last_detection.points for obj in tracked_objects])
    org_points = np.array([obj.points for obj in detections])
    tracked_ids, tracked_bbox_ids = [], []
    for tid, tracked_point in enumerate(tracked_points):
        org_p_id = np.argmin(np.array([np.linalg.norm(tracked_point-org_point) for org_point in org_points]))
        tracked_bbox_ids.append(org_p_id)
        tracked_ids.append(tracked_ids_out[tid])
    return tracked_ids, tracked_bbox_ids


def get_tracked_ids_byte(tracking_points, tracked_objects):
    tracked_ids_out = np.array([obj[4] for obj in tracked_objects])
    tracked_points = np.array([obj[:4] for obj in tracked_objects])

    tracked_ids, tracked_bbox_ids = [], []
    for tid, tracked_point in enumerate(tracked_points):
        org_p_id = np.argmin(np.array(
            [np.linalg.norm(tracked_point-org_point) for org_point in tracking_points]))
        tracked_bbox_ids.append(org_p_id)
        tracked_ids.append(int(tracked_ids_out[tid]))
    return tracked_ids, tracked_bbox_ids

def main():
    with ConfigContext(parse_args(sys.argv[1:])):
        demo = Demo()
        dataset_dir = '/home/yusun/DataCenter2/dataset'
        method = 'BEV+byte'
        if args().eval_dataset=='posetrack':
            seq_folder_list = glob.glob(os.path.join(dataset_dir, 'posetrack2018/images/val/*'))
            seq_save_paths = [seq_folder.replace('posetrack2018/images', '../tracking_results/posetrack2018_{}'.format(method)) for seq_folder in seq_folder_list]
            tracking_save_folder = os.path.join(dataset_dir, '../tracking_results/posetrack2018')
        elif args().eval_dataset == 'mupots':
            seq_folder_list = glob.glob(os.path.join(dataset_dir, 'MultiPersonTestSet/TS*'))
            seq_save_paths = [seq_folder.replace('MultiPersonTestSet', '../tracking_results/MultiPersonTestSet_{}'.format(method)) for seq_folder in seq_folder_list]
            tracking_save_folder = os.path.join(dataset_dir, '../tracking_results/MultiPersonTestSet')
        elif args().eval_dataset == 'dancetrack':
            seq_folder_list = [os.path.join(path,'img1') for path in glob.glob(os.path.join(dataset_dir, 'dancetrack','test','*'))]
            seq_save_paths = [seq_folder.replace(os.path.join('dancetrack','test'), '../tracking_results/dancetrack_{}_{}'.format(method,args().centermap_conf_thresh)).replace('/img1','') for seq_folder in seq_folder_list]
            tracking_save_folder = os.path.join(dataset_dir, '../tracking_results/dancetrack_BEV_{}'.format(args().centermap_conf_thresh))
        seq_save_npzs = demo.process_demo(seq_folder_list, seq_save_paths)
        demo.perform_tracking(seq_save_npzs, tracking_save_folder)

if __name__ == '__main__':
    main()


"""
NorFair Tracker
The class in charge of performing the tracking of the detections produced by the detector. The Tracker class first needs to get instantiated as an object, and then continuously updated inside a video processing loop by feeding new detections into its update method.

initialization_delay: 2      可能还是需要有点才行，有助于 卡尔曼滤波器形成不动的速度。
hit_inertia_min: 2     可能还是需要有点才行，有助于  滤除突然的噪声。
hit_inertia_max: 1000    尽可能延长tracker的生命，tracklets never die.



Arguments:
distance_function: Function used by the tracker to determine the distance between newly detected objects and the objects the tracker is currently tracking. This function should take 2 arguments, the first being a detection of type Detection, and the second a tracked object of type TrackedObject, and should return a float with the distance it calculates.
distance_threshold: Defines what is the maximum distance that can constitute a match. Detections and tracked objects whose distance are above this threshold won't be matched by the tracker.
hit_inertia_min (optional): Each tracked objects keeps an internal hit inertia counter which tracks how often it's getting matched to a detection, each time it gets a match this counter goes up, and each time it doesn't it goes down. If it doesn't get any match for a certain amount of frames, and it then gets below the value set by this argument, the object is destroyed. Defaults to 10.
hit_inertia_max (optional): Each tracked objects keeps an internal hit inertia counter which tracks how often it's getting matched to a detection, each time it gets a match this counter goes up, and each time it doesn't it goes down. This argument defines how large this inertia can grow, and therefore defines how long an object can live without getting matched to any detections. Defaults to 25.
initialization_delay (optional): Each tracked object waits till its internal hit intertia counter goes over hit_inertia_min to be considered as a potential object to be returned to the user by the Tracker. The argument initialization_delay determines by how much the object's hit inertia counter must exceed hit_inertia_min to be considered as initialized and get returned to the user as a real object. Defaults to (hit_inertia_max - hit_inertia_min) / 2.
detection_threshold (optional): Sets the threshold at which the scores of the points in a detection being fed into the tracker must dip below to be ignored by the tracker. Defaults to 0.
point_transience (optional): Each tracked object keeps track of how much often of the points its tracking has been getting matched. Points that are getting matches are said to be live, and points which aren't are said to not be live. This determines things like which points in a tracked object get drawn by draw_tracked_objects and which don't. This argument determines how short lived points not getting matched are. Defaults to 4.
filter_setup (optional): This parameter can be used to change the parameters of the Kalman Filter that is used by TrackedObject instances. Defaults to FilterSetup().
past_detections_length: How many past detections to save for each tracked object. Norfair tries to distribute these past detections uniformly through the object's lifetime so they're more representative of it. Very useful if you want to add metric learning to your model, as you can associate an embedding to each detection and access them in your distance function. Defaults to 4.
Tracker.update
The function through which the detections found in each frame must be passed to the tracker.

Arguments:
detections (optional): A list of Detections which represent the detections found in the current frame being processed. If no detections have been found in the current frame, or the user is purposely skipping frames to improve video processing time, this argument should be set to None or ignored, as the update function is needed to advance the state of the Kalman Filters inside the tracker. Defaults to None.
period (optional): The user can chose not to run their detector on all frames, so as to process video faster. This parameter sets every how many frames the detector is getting ran, so that the tracker is aware of this situation and can handle it properly. This argument can be reset on each frame processed, which is useful if the user is dynamically changing how many frames the detector is skipping on a video when working in real-time. Defaults to 1.
Returns:
A list of TrackedObjects.
Detection
Detections returned by the detector must be converted to a Detection object before being used by Norfair.

Arguments and Properties:
points: A numpy array of shape (number of points per object, 2), with each row being a point expressed as x, y coordinates on the image. The number of points per detection must be constant for each particular tracker.
scores: An array of length number of points per object which assigns a score to each of the points defined in points. This is used to inform the tracker of which points to ignore; any point with a score below detection_threshold will be ignored. This useful for cases in which detections don't always have every point detected, as is often the case in pose estimators.
data: The place to store any extra data which may be useful when calculating the distance function. Anything stored here will be available to use inside the distance function. This enables the development of more interesting trackers which can do things like assign an appearance embedding to each detection to aid in its tracking.
FilterSetup
This class can be used either to change some parameters of the KalmanFilter that the tracker uses, or to fully customize the predictive filter implementation to use (as long as the methods and properties are compatible). The former case only requires changing the default parameters upon tracker creation: tracker = Tracker(..., filter_setup=FilterSetup(R=100)), while the latter requires creating your own class extending FilterSetup, and rewriting its create_filter method to return your own customized filter.

Arguments:
Note that these arguments correspond to the same parameters of the filterpy.KalmanFilter (see docs) that this class returns.

R: Multiplier for the sensor measurement noise matrix. Defaults to 4.0.
Q: Multiplier for the process uncertainty. Defaults to 0.1.
P: Multiplier for the initial covariance matrix estimation, only in the entries that correspond to position (not speed) variables. Defaults to 10.0.
FilterSetup.create_filter
This function returns a new predictive filter instance with the current setup, to be used by each new TrackedObject that is created. This predictive filter will be used to estimate speed and future positions of the object, to better match the detections during its trajectory.

This method may be overwritten by a subclass of FilterSetup, in case that further customizations of the filter parameters or implementation are needed.

Arguments:
initial_detection: numpy array of shape (number of points per object, 2), corresponding to the Detection.points of the tracked object being born, which shall be used as initial position estimation for it.
Returns:
A new filterpy.KalmanFilter instance (or an API compatible object, since the class is not restricted by type checking).

TrackedObject
The objects returned by the tracker's update function on each iteration. They represent the objects currently being tracked by the tracker.

Properties:
estimate: Where the tracker predicts the point will be in the current frame based on past detections. A numpy array with the same shape as the detections being fed to the tracker that produced it.
id: The unique identifier assigned to this object by the tracker.
last_detection: The last detection that matched with this tracked object. Useful if you are storing embeddings in your detections and want to do metric learning, or for debugging.
last_distance: The distance the tracker had with the last object it matched with.
age: The age of this object measured in number of frames.
live_points: A boolean mask with shape (number of points per object). Points marked as True have recently been matched with detections. Points marked as False haven't and are to be considered as stale, and should be ignored. Functions like draw_tracked_objects use this property to determine which points not to draw.
initializing_id: On top of id, objects also have an initializing_id which is the id they are given internally by the Tracker, which is used for debugging. Each new object created by the Tracker starts as an uninitialized TrackedObject, which needs to reach a certain match rate to be converted into a full blown TrackedObject. This is the id assigned to TrackedObject while they are getting initialized.
"""
