from pycocotools.coco import COCO
from dataset.image_base import *
from dataset.base import Base_Classes, Test_Funcs

default_mode = args().image_loading_mode

def MuPoTS(base_class=default_mode):
    class MuPoTS(Base_Classes[base_class]):
        def __init__(self,train_flag=False, split='test', **kwargs):
            super(MuPoTS,self).__init__(train_flag)
            self.data_folder = os.path.join(self.data_folder,'MultiPersonTestSet/')
            self.split = split
            self.test2val_sample_ratio = 10
            self.annot_path = os.path.join(self.data_folder, 'MuPoTS_annots.npz') #'MuPoTS-3D.json'
            if not os.path.exists(self.annot_path):
                self.pack_data()

            self.image_folder = self.data_folder
            self.load_data() 
            self.root_idx = constants.SMPL_ALL_54['Pelvis']
            self.kp2d_mapper = constants.joint_mapping(constants.MuPoTS_17, constants.SMPL_ALL_54)
            self.kp3d_mapper = constants.joint_mapping(constants.MuPoTS_17, constants.SMPL_ALL_54)
            logging.info('MuPoTS dataset total {} samples, loading {} split'.format(self.__len__(), self.split))
        
        def load_data(self):
            annots = np.load(self.annot_path, allow_pickle=True)['annots'][()]
            sequence_dict = {}
            for seq_name, seq_annots in annots.items():
                sequence_dict[seq_name] = {fid:'img_{:06d}.jpg'.format(fid) for fid in range(seq_annots['frame_num'])}
            sequence_dict = OrderedDict(sequence_dict)

            self.annots, self.file_paths = {}, []
            for sid, seq_name in enumerate(sequence_dict):
                frame_ids = sorted(sequence_dict[seq_name].keys())
                for fid in frame_ids:
                    img_path = os.path.join(seq_name, sequence_dict[seq_name][fid])
                    self.file_paths.append(img_path)
                    self.annots[img_path] = [annots[seq_name]['kp2ds'][fid], annots[seq_name]['kp3ds'][fid], annots[seq_name]['track_ids'][fid], annots[seq_name]['camMats'][fid]]

        def get_image_info(self, index):
            img_name = self.file_paths[index]
            imgpath = os.path.join(self.image_folder,img_name)
            image = cv2.imread(imgpath)[:,:,::-1] 

            kp2ds, kp3ds = [], [] 
            for kp2d, kp3d in zip(self.annots[img_name][0], self.annots[img_name][1]):
                kp2ds.append(self.map_kps(kp2d,maps=self.kp2d_mapper))
                kp3d = self.map_kps(kp3d/1000.,maps=self.kp3d_mapper)
                kp3ds.append(kp3d)

            kp2ds, kp3ds = np.array(kp2ds), np.array(kp3ds)
            root_trans = kp3ds[:,self.root_inds].mean(1)
            valid_masks = np.array([self._check_kp3d_visible_parts_(kp3d) for kp3d in kp3ds])
            kp3ds -= root_trans[:,None]
            kp3ds[~valid_masks] = -2.
            fx, fy, cx, cy = self.annots[img_name][3]
            camMats = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            person_num = len(kp2ds)

            vis_masks = []
            for kp2d in kp2ds:
                vis_masks.append(_check_visible(kp2d,get_mask=True))
            kp2ds = np.concatenate([kp2ds, np.array(vis_masks)[:,:,None]],2)

            img_info = {'imgpath': imgpath, 'image': image, 'kp2ds': kp2ds, 'track_ids': None,\
                    'vmask_2d': np.array([[True,True,True] for _ in range(person_num)]), 'vmask_3d': np.array([[True,False,False,False,False,True] for _ in range(person_num)]),\
                    'kp3ds': kp3ds, 'params': None, 'root_trans': root_trans, 'verts': None,\
                    'camMats': camMats, 'img_size': image.shape[:2],'ds': 'mupots'}
            
            return img_info
        
        def pack_data(self):
            cam_mat = {}
            db = COCO(os.path.join(self.data_folder, 'MuPoTS-3D.json'))
            for aid in db.anns.keys():
                ann = db.anns[aid]
                image_id = ann['image_id']
                img = db.loadImgs(image_id)[0]
                img_path = img['file_name']
                fx, fy, cx, cy = img['intrinsic']
                intrinsic_params = np.array([fx, fy, cx, cy])
                if img_path not in cam_mat:
                    cam_mat[img_path] = intrinsic_params

            import scipy.io
            annots = {}
            track_id_cache = 0
            for seq_id in range(1, 21):
                sequence_name = 'TS{}'.format(seq_id)
                print('packing sequence ', sequence_name)
                annotation_file_path = os.path.join(self.data_folder, sequence_name, 'annot.mat')
                occlusion_file_path = os.path.join(self.data_folder, sequence_name, 'occlusion.mat')
                annotation = scipy.io.loadmat(annotation_file_path)['annotations']
                occlusion = scipy.io.loadmat(occlusion_file_path)['occlusion_labels']

                frame_num = len(annotation)
                assert frame_num == len(occlusion), \
                    'occlusion number mismatch, annotation has {}, while occlusion has {}'.format(frame_num, len(occlusion))
                subject_num = len(annotation[0])

                seq_kp2ds = np.zeros((frame_num, subject_num, 17, 2))
                seq_kp3ds = np.zeros((frame_num, subject_num, 17, 3))
                seq_univ_kp3ds = np.zeros((frame_num, subject_num, 17, 3))
                seq_valid_flag = np.zeros((frame_num, subject_num),dtype=np.bool)
                seq_joint_occlusion = np.zeros((frame_num, subject_num, 17),dtype=np.bool)
                seq_track_ids = np.zeros((frame_num, subject_num))
                seq_camMats = np.zeros((frame_num, 4))
                for frame_id in range(len(annotation)):
                    assert subject_num == len(annotation[frame_id]), \
                        'subject number mismatch, 0-th {}, while {} has {}'.format(subject_num, len(annotation[frame_id]))
                    for subject_id in range(subject_num):
                        subj_annot = annotation[frame_id][subject_id][0,0]
                        kp2ds, kp3ds, univ_kp3ds, valid_flag = subj_annot[0], subj_annot[1], subj_annot[2], subj_annot[3]
                        joint_occlusion = occlusion[frame_id][subject_id]
                        #print(joint_occlusion.shape, kp2ds.shape, kp3ds.shape, univ_kp3ds.shape, valid_flag)
                        #(1, 17) (2, 17) (3, 17) (3, 17) [[1]]

                        seq_kp2ds[frame_id,subject_id] = kp2ds.transpose((1,0))
                        seq_kp3ds[frame_id,subject_id] = kp3ds.transpose((1,0))
                        seq_univ_kp3ds[frame_id,subject_id] = univ_kp3ds.transpose((1,0))
                        seq_valid_flag[frame_id,subject_id] = bool(valid_flag[0,0])
                        seq_joint_occlusion[frame_id,subject_id] = joint_occlusion[0]
                        seq_track_ids[frame_id,subject_id] = subject_id + track_id_cache
                        img_path = os.path.join(sequence_name, 'img_{:06d}.jpg'.format(frame_id))
                        seq_camMats[frame_id] = cam_mat[img_path]
                        if not seq_valid_flag[frame_id,subject_id]:
                            print(sequence_name, frame_id, subject_id, 'invalid', valid_flag)
                
                annots[sequence_name] = {'kp2ds':seq_kp2ds, 'kp3ds':seq_kp3ds, 'univ_kp3ds':seq_univ_kp3ds, 'track_ids':seq_track_ids, 'camMats':seq_camMats,\
                    'valid_flag':seq_valid_flag, 'joint_occlusion':seq_joint_occlusion, 'frame_num':frame_num, 'subject_num':subject_num}
                track_id_cache += subject_num

            np.savez(self.annot_path, annots=annots)

        def evaluate_relative_pose(self, preds, result_dir):
            
            print('Evaluation start...')
            gts = self.data
            sample_num = len(preds)
            joint_num = self.original_joint_num
    
            pred_2d_save = {}
            pred_3d_save = {}
            for n in range(sample_num):
                
                gt = gts[n]
                f = gt['f']
                c = gt['c']
                bbox = gt['bbox']
                gt_3d_root = gt['root_cam']
                img_name = gt['img_path'].split('/')
                img_name = img_name[-2] + '_' + img_name[-1].split('.')[0] # e.g., TS1_img_0001
                
                # restore coordinates to original space
                pred_2d_kpt = preds[n].copy()
                # only consider eval_joint
                pred_2d_kpt = np.take(pred_2d_kpt, self.eval_joint, axis=0)
                pred_2d_kpt[:,2] = (pred_2d_kpt[:,2] / cfg.depth_dim * 2 - 1) * (cfg.bbox_3d_shape[0]/2) + gt_3d_root[2]

                # 2d kpt save
                if img_name in pred_2d_save:
                    pred_2d_save[img_name].append(pred_2d_kpt[:,:2])
                else:
                    pred_2d_save[img_name] = [pred_2d_kpt[:,:2]]

                vis = False
                if vis:
                    cvimg = cv2.imread(gt['img_path'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                    filename = str(random.randrange(1,500))
                    tmpimg = cvimg.copy().astype(np.uint8)
                    tmpkps = np.zeros((3,joint_num))
                    tmpkps[0,:], tmpkps[1,:] = pred_2d_kpt[:,0], pred_2d_kpt[:,1]
                    tmpkps[2,:] = 1
                    tmpimg = vis_keypoints(tmpimg, tmpkps, self.skeleton)
                    cv2.imwrite(filename + '_output.jpg', tmpimg)
                
                # 3d kpt save
                if img_name in pred_3d_save:
                    pred_3d_save[img_name].append(pred_3d_kpt)
                else:
                    pred_3d_save[img_name] = [pred_3d_kpt]
            
            output_path = osp.join(result_dir,'preds_2d_kpt_mupots.mat')
            sio.savemat(output_path, pred_2d_save)
            print("Testing result is saved at " + output_path)
            output_path = osp.join(result_dir,'preds_3d_kpt_mupots.mat')
            sio.savemat(output_path, pred_3d_save)
            print("Testing result is saved at " + output_path)
        
        def evaluate_pelvis_depth(self, preds, result_dir):
            print('Evaluation start...')
            pred_save = []

            gts = self.data
            sample_num = len(preds)
            for n in range(sample_num):
                
                gt = gts[n]
                image_id = gt['image_id']
                f = gt['f']
                c = gt['c']
                bbox = gt['bbox'].tolist()
                score = gt['score']
                
                # restore coordinates to original space
                pred_root = preds[n].copy()
                pred_root[0] = pred_root[0] / 64 * bbox[2] + bbox[0]
                pred_root[1] = pred_root[1] / 64 * bbox[3] + bbox[1]

                # back project to camera coordinate system
                pred_root = pixel2cam(pred_root[None,:], f, c)[0]

                pred_save.append({'image_id': image_id, 'root_cam': pred_root.tolist(), 'bbox': bbox, 'score': score})
            
            output_path = os.path.join(result_dir, 'bbox_root_mupots_output.json')
            with open(output_path, 'w') as f:
                json.dump(pred_save, f)
            print("Test result is saved at " + output_path)

            calculate_score(output_path, self.annot_path, 250)
    return MuPoTS
    
def calculate_score(output_path, annot_path, thr=250):
    ## Refer to https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE/blob/master/data/MuPoTS/MuPoTS_eval.py
    with open(output_path, 'r') as f:
        output = json.load(f)

    # AP measure
    def return_score(pred):
        return pred['score']
    output.sort(reverse=True, key=return_score)

    db = COCO(annot_path)
    gt_num = len([k for k,v in db.anns.items() if v['is_valid'] == 1])
    tp_acc = 0
    fp_acc = 0
    precision = []; recall = [];
    is_matched = {}
    for n in range(len(output)):
        image_id = output[n]['image_id']
        pred_root = output[n]['root_cam']
        score = output[n]['score']

        img = db.loadImgs(image_id)[0]
        ann_ids = db.getAnnIds(image_id)
        anns = db.loadAnns(ann_ids)
        valid_frame_num = len([item for item in anns if item['is_valid'] == 1])
        if valid_frame_num == 0:
            continue

        if str(image_id) not in is_matched:
            is_matched[str(image_id)] = [0 for _ in range(len(anns))]
        
        min_dist = 9999
        save_ann_id = -1
        for ann_id,ann in enumerate(anns):
            if ann['is_valid'] == 0:
                continue
            gt_root = np.array(ann['keypoints_cam'])
            root_idx = 14
            gt_root = gt_root[root_idx]

            dist = math.sqrt(np.sum((pred_root - gt_root) ** 2))
            if min_dist > dist:
                min_dist = dist
                save_ann_id = ann_id
        
        is_tp = False
        if save_ann_id != -1 and min_dist < thr:
            if is_matched[str(image_id)][save_ann_id] == 0:
                is_tp = True
                is_matched[str(image_id)][save_ann_id] = 1
        
        if is_tp:
            tp_acc += 1
        else:
            fp_acc += 1
            
        precision.append(tp_acc/(tp_acc + fp_acc))
        recall.append(tp_acc/gt_num)

    AP = 0
    for n in range(len(precision)-1):
        AP += precision[n+1] * (recall[n+1] - recall[n])

    print('AP_root: ' + str(AP))


def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / (cam_coord[:, 2] + 1e-8) * f[0] + c[0]
    y = cam_coord[:, 1] / (cam_coord[:, 2] + 1e-8) * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    return img_coord

def pixel2cam(pixel_coord, f, c):
    x = (pixel_coord[:, 0] - c[0]) / f[0] * pixel_coord[:, 2]
    y = (pixel_coord[:, 1] - c[1]) / f[1] * pixel_coord[:, 2]
    z = pixel_coord[:, 2]
    cam_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    return cam_coord

def world2cam(world_coord, R, t):
    cam_coord = np.dot(R, world_coord.transpose(1,0)).transpose(1,0) + t.reshape(1,3)
    return cam_coord

def get_bbox(joint_img):
    # bbox extract from keypoint coordinates
    bbox = np.zeros((4))
    xmin = np.min(joint_img[:,0])
    ymin = np.min(joint_img[:,1])
    xmax = np.max(joint_img[:,0])
    ymax = np.max(joint_img[:,1])
    width = xmax - xmin - 1
    height = ymax - ymin - 1
    
    bbox[0] = (xmin + xmax)/2. - width/2*1.2
    bbox[1] = (ymin + ymax)/2. - height/2*1.2
    bbox[2] = width*1.2
    bbox[3] = height*1.2

    return bbox

def process_bbox(bbox, width, height):
    # sanitize bboxes
    x, y, w, h = bbox
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
    y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
    if w*h > 0 and x2 >= x1 and y2 >= y1:
        bbox = np.array([x1, y1, x2-x1, y2-y1])
    else:
        return None

    # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w/2.
    c_y = bbox[1] + h/2.
    aspect_ratio = 512/512
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w*1.25
    bbox[3] = h*1.25
    bbox[0] = c_x - bbox[2]/2.
    bbox[1] = c_y - bbox[3]/2.
    return bbox

def multi_meshgrid(*args):
    """
    Creates a meshgrid from possibly many
    elements (instead of only 2).
    Returns a nd tensor with as many dimensions
    as there are arguments
    """
    args = list(args)
    template = [1 for _ in args]
    for i in range(len(args)):
        n = args[i].shape[0]
        template_copy = template.copy()
        template_copy[i] = n
        args[i] = args[i].view(*template_copy)
        # there will be some broadcast magic going on
    return tuple(args)


def flip(tensor, dims):
    if not isinstance(dims, (tuple, list)):
        dims = [dims]
    indices = [torch.arange(tensor.shape[dim] - 1, -1, -1,
                            dtype=torch.int64) for dim in dims]
    multi_indices = multi_meshgrid(*indices)
    final_indices = [slice(i) for i in tensor.shape]
    for i, dim in enumerate(dims):
        final_indices[dim] = multi_indices[i]
    flipped = tensor[final_indices]
    assert flipped.device == tensor.device
    assert flipped.requires_grad == tensor.requires_grad
    return flipped


def _check_visible(joints, w=2048, h=2048, get_mask=False):
    visibility = True
    # check that all joints are visible
    x_in = np.logical_and(joints[:, 0] < w, joints[:, 0] >= 0)
    y_in = np.logical_and(joints[:, 1] < h, joints[:, 1] >= 0)
    ok_pts = np.logical_and(x_in, y_in)
    if np.sum(ok_pts) < 16:
        visibility=False
    if get_mask:
        return ok_pts
    return visibility


import os
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl

def vis_keypoints(img, kps, kps_lines, kp_thresh=0.4, alpha=1):

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
        p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

def vis_3d_skeleton(kpt_3d, kpt_3d_vis, kps_lines, filename=None):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]

    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        x = np.array([kpt_3d[i1,0], kpt_3d[i2,0]])
        y = np.array([kpt_3d[i1,1], kpt_3d[i2,1]])
        z = np.array([kpt_3d[i1,2], kpt_3d[i2,2]])

        if kpt_3d_vis[i1,0] > 0 and kpt_3d_vis[i2,0] > 0:
            ax.plot(x, z, -y, c=colors[l], linewidth=2)
        if kpt_3d_vis[i1,0] > 0:
            ax.scatter(kpt_3d[i1,0], kpt_3d[i1,2], -kpt_3d[i1,1], c=colors[l], marker='o')
        if kpt_3d_vis[i2,0] > 0:
            ax.scatter(kpt_3d[i2,0], kpt_3d[i2,2], -kpt_3d[i2,1], c=colors[l], marker='o')

    if filename is None:
        ax.set_title('3D vis')
    else:
        ax.set_title(filename)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')
    ax.legend()
    
    plt.show()
    cv2.waitKey(0)

def vis_3d_multiple_skeleton(kpt_3d, kpt_3d_vis, kps_lines, filename=None):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]

    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]

        person_num = kpt_3d.shape[0]
        for n in range(person_num):
            x = np.array([kpt_3d[n,i1,0], kpt_3d[n,i2,0]])
            y = np.array([kpt_3d[n,i1,1], kpt_3d[n,i2,1]])
            z = np.array([kpt_3d[n,i1,2], kpt_3d[n,i2,2]])

            if kpt_3d_vis[n,i1,0] > 0 and kpt_3d_vis[n,i2,0] > 0:
                ax.plot(x, z, -y, c=colors[l], linewidth=2)
            if kpt_3d_vis[n,i1,0] > 0:
                ax.scatter(kpt_3d[n,i1,0], kpt_3d[n,i1,2], -kpt_3d[n,i1,1], c=colors[l], marker='o')
            if kpt_3d_vis[n,i2,0] > 0:
                ax.scatter(kpt_3d[n,i2,0], kpt_3d[n,i2,2], -kpt_3d[n,i2,1], c=colors[l], marker='o')

    if filename is None:
        ax.set_title('3D vis')
    else:
        ax.set_title(filename)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')
    ax.legend()
    
    plt.show()
    cv2.waitKey(0)

if __name__ == '__main__':
    dataset=MuPoTS(base_class=default_mode)(train_flag=False)
    Test_Funcs[default_mode](dataset)
    print('Done')