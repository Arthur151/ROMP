import numpy as np
import torch
import glob
import os
import joblib
import lap
import cv2
from torch import nn
from trace2.tracker.tracker3D import Tracker
from trace2.evaluation.evaluate_tracking import evaluate_trackers
import pyrender, trimesh

def video2frame(video_name, frame_save_dir=None):
    cap = OpenCVCapture(video_name)
    os.makedirs(frame_save_dir, exist_ok=True)
    frame_list = []
    for frame_id in range(int(cap.length)):
        frame = cap.read(return_rgb=False)
        save_path = os.path.join(frame_save_dir, '{:06d}.jpg'.format(frame_id))
        cv2.imwrite(save_path, frame)
        frame_list.append(save_path)
    return frame_list

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
    
def combining_motion_offsets_and_detection_to_trajectory_batch(outputs, add_offsets=True, temp_clip_length=8):
    # very like a tracker that use estimated motion_offsets to replace the effect of kalman filter.
    motion_offsets, cam_preds, batch_ids = outputs['motion_offsets'].detach().cpu(), outputs['cam'].detach().cpu(), outputs['reorganize_idx'].cpu()
    if add_offsets:
        cam_offset_preds = cam_preds + motion_offsets
    else:
        cam_offset_preds = cam_preds

    seq_ids = batch_ids // temp_clip_length
    clip_frame_ids = batch_ids % temp_clip_length
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

        seq_traj_cams = torch.ones(max_subject_num, temp_clip_length, 3) * -2.
        seq_traj_rids = torch.ones(max_subject_num, temp_clip_length).long() * -1

        seq_traj_cams[:(seq_frame_ids == 0).sum(),
                      0] = seq_cam_off[seq_frame_ids == 0]
        seq_traj_rids[:(seq_frame_ids == 0).sum(),
                      0] = seq_result_inds[seq_frame_ids == 0]
        
        for fid in range(1, temp_clip_length):
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

def collect_sequence_tracking_results(outputs, img_paths, reorganize_idx, show=False):
    track_ids = outputs['track_ids'].detach().cpu().numpy()
    pj2d_org_results = outputs['pj2d_org'].detach().cpu().numpy().astype(np.float16)

    tracking_results = {}
    for frame_id, img_path in enumerate(img_paths):
        pred_ids = np.where(reorganize_idx==frame_id)[0]
        img_name = os.path.basename(img_path)
        if img_name not in tracking_results:
            tracking_results[img_name] = {'track_ids':[], 'track_bbox':[], 'pj2ds':[]}
        for batch_id in pred_ids:
            track_id = track_ids[batch_id]
            pj2d_org = pj2d_org_results[batch_id]
            bbox = pj2ds_to_bbox(pj2d_org)
            tracking_results[img_name]['track_ids'].append(track_id)
            tracking_results[img_name]['track_bbox'].append(bbox)
            tracking_results[img_name]['pj2ds'].append(pj2d_org)
            #results[img_name][track_id]['center_conf'] = center_confs[batch_id]
        if show:
            vis_track_bbox(img_path, tracking_results[img_name]['track_ids'], tracking_results[img_name]['track_bbox'])
    return tracking_results

def vis_track_bbox(image_path, tracked_ids, tracked_bbox):
    org_img = cv2.imread(image_path)
    for tid, bbox in zip(tracked_ids, tracked_bbox):
        bbox = np.array(bbox).astype(np.int32)
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

class Demo(nn.Module):
    def __init__(self):
        super(Demo, self).__init__()
        self._build_model_()
        self.test_cfg = {'mode':'parsing', 'calc_loss': False,'with_nms':True,'new_training': args().new_training}
        self.eval_dataset = args().eval_dataset
        self.save_mesh = False
        self.pyrender_render = Renderer(self.visualizer.smpl_face[0], focal_length=args().focal_length*2,height=1024,width=1024)
        self.pyrender_render_bv = Renderer(self.visualizer.smpl_face[0], focal_length=1000,height=2048,width=2048)
        print('Initialization finished!')

    def net_forward(self,meta_data,mode='val'):
        if mode=='val':
            cfg_dict = self.test_cfg
        elif mode=='eval':
            cfg_dict = self.eval_cfg
        ds_org, imgpath_org = get_remove_keys(meta_data,keys=['data_set','imgpath'])
        meta_data['batch_ids'] = torch.arange(len(meta_data['image']))
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

class Renderer(object):

    def __init__(self, faces, focal_length=1000, height=512, width=512):
        self.renderer = pyrender.OffscreenRenderer(height, width)
        self.faces = faces
        self.focal_length = focal_length
        self.colors = [
                        (.7, .7, .6, 1.),
                        (.7, .5, .5, 1.),  # Pink
                        (.5, .5, .7, 1.),  # Blue
                        (.5, .55, .3, 1.),  # capsule
                        (.3, .5, .55, 1.),  # Yellow
                    ]

    def __call__(self, images, vertices, translation, mesh_colors=None,focal_length=None,camera_pose=None):
        # List to store rendered scenes
        output_images = []
        # Need to flip x-axis
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        for i in range(len(images)):
            img = images[i]#.cpu().numpy()#.transpose(1, 2, 0)
            self.renderer.viewport_height = img.shape[0]
            self.renderer.viewport_width = img.shape[1]
            verts = vertices[i].detach().cpu().numpy()
            mesh_trans = translation[i].cpu().numpy()
            verts = verts + mesh_trans[:, None, ]
            num_people = verts.shape[0]

            # Create a scene for each image and render all meshes
            scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                                   ambient_light=(0.3, 0.3, 0.3))

            
            # Create camera. Camera will always be at [0,0,0]
            # CHECK If I need to swap x and y
            camera_center = np.array([img.shape[1] / 2., img.shape[0] / 2.])
            if camera_pose is None:
                camera_pose = np.eye(4)

            if focal_length is None:
                fx,fy = self.focal_length, self.focal_length
            else:
                fx,fy = focal_length, focal_length
            camera = pyrender.camera.IntrinsicsCamera(fx=fx, fy=fy,
                                                      cx=camera_center[0], cy=camera_center[1])
            scene.add(camera, pose=camera_pose)
            # Create light source
            light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.5)
            # for every person in the scene
            for n in range(num_people):
                mesh = trimesh.Trimesh(verts[n], self.faces)
                mesh.apply_transform(rot)
                trans = 0 * mesh_trans[n]
                trans[0] *= -1
                trans[2] *= -1
                if mesh_colors is None:
                    mesh_color = self.colors[n % len(self.colors)]
                else:
                    mesh_color = mesh_colors[n % len(mesh_colors)]
                material = pyrender.MetallicRoughnessMaterial(
                    metallicFactor=0.2,
                    alphaMode='OPAQUE',
                    baseColorFactor=mesh_color)
                mesh = pyrender.Mesh.from_trimesh(
                    mesh,
                    material=material)
                scene.add(mesh, 'mesh')

                # Use 3 directional lights
                light_pose = np.eye(4)
                light_pose[:3, 3] = np.array([0, -1, 1]) + trans
                scene.add(light, pose=light_pose)
                light_pose[:3, 3] = np.array([0, 1, 1]) + trans
                scene.add(light, pose=light_pose)
                light_pose[:3, 3] = np.array([1, 1, 2]) + trans
                scene.add(light, pose=light_pose)
            # Alpha channel was not working previously need to check again
            # Until this is fixed use hack with depth image to get the opacity
            color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            # color = color[::-1,::-1]
            # rend_depth = rend_depth[::-1,::-1]
            color = color.astype(np.float32)# / 255.0
            valid_mask = (rend_depth > 0)[:, :, None]
            output_img = (color[:, :, :3] * valid_mask +
                          (1 - valid_mask) * img)
            output_img = np.transpose(output_img, (2, 0, 1))
            output_images.append(output_img)

        return output_images


def convert_front_view_to_bird_view_video(verts_t, bv_trans=None, h=512,w=512, focal_length=50):
    R_bv = torch.zeros(3, 3, device=verts_t.device)
    R_bv[0, 0] = R_bv[2, 1] = 1
    R_bv[1, 2] = -1
    verts_tr = torch.einsum('bij,kj->bik', verts_t, R_bv)
    verts_tfar = verts_tr
    p_min, p_max = verts_tfar.view(-1, 3).min(0)[0], verts_tfar.view(-1, 3).max(0)[0]
    p_center = 0.5 * (p_min + p_max)
    verts_center = (verts_tfar.view(-1, 3) - p_center).view(verts_t.shape[0], -1, 3)

    dis_min, dis_max = (verts_tfar.view(-1, 3) - p_center).min(0)[0], (
            verts_tfar.view(-1, 3) - p_center).max(0)[0]
    # h, w = min(h, w), min(h, w)
    ratio_max = abs(0.9 - 0.5)
    z_x = dis_max[0] * focal_length / (ratio_max * w) + torch.abs(dis_min[2])
    z_y = dis_max[1] * focal_length / (ratio_max * h) + torch.abs(dis_min[2])
    z_x_0 = (-dis_min[0]) * focal_length / (ratio_max * w) + torch.abs(
        dis_min[2])
    z_y_0 = (-dis_min[1]) * focal_length / (ratio_max * h) + torch.abs(
        dis_min[2])
    z = max(z_x, z_y, z_x_0, z_y_0)
    if bv_trans is None:
        pass
        #print('bird view:', p_center, z)
    else:
        p_center, z = bv_trans
        p_center, z = p_center.to(verts_tfar.device), z.to(verts_tfar.device)
    verts_right = verts_tfar - p_center + torch.tensor([0, 0, z], device=verts_t.device)
    return verts_right

def rendering_mesh_to_image(self, outputs, seq_save_dirs):
    used_org_inds, per_img_inds = process_idx(outputs['reorganize_idx'])
    seq_save_dirs = [seq_save_dirs[ind] for ind in used_org_inds]
    mesh_colors = torch.Tensor([color_list[idx%len(color_list)] for idx in range(len(outputs['reorganize_idx']))])

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
        evaluate_trackers(tracking_save_folder, dataset=args().eval_dataset)

if __name__ == '__main__':
    main()
