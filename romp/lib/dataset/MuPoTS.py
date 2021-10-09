from pycocotools.coco import COCO
import sys, os

from dataset.image_base import *

class MuPoTS(Image_base):
    def __init__(self,train_flag=True, split='val', **kwargs):
        super(MuPoTS,self).__init__(train_flag)
        self.data_folder = os.path.join(self.data_folder,'MultiPersonTestSet/')
        self.split = split
        self.test2val_sample_ratio = 10
        self.annot_path = os.path.join(self.data_folder, 'MuPoTS-3D.json')

        self.image_folder = self.data_folder
        self.load_data() 
        self.root_idx = constants.SMPL_ALL_54['Pelvis']

        self.file_paths = list(self.annots.keys())
        self.kp2d_mapper = constants.joint_mapping(constants.MuPoTS_17, constants.SMPL_ALL_54)
        self.kp3d_mapper = constants.joint_mapping(constants.MuPoTS_17, constants.SMPL_ALL_54)
        logging.info('MuPoTS dataset total {} samples, loading {} split'.format(self.__len__(), self.split))

    def load_data(self):
        annots = {}
        db = COCO(self.annot_path)
        logging.info("Get bounding box from groundtruth")
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_path = img['file_name']
            if img_path not in annots:
                annots[img_path] = [[],[],[]]

            fx, fy, cx, cy = img['intrinsic']
            intrinsic_params = np.array([fx, fy, cx, cy])

            kp3d = np.array(ann['keypoints_cam']) # [X, Y, Z] in camera coordinate
            kp2d = np.array(ann['keypoints_img'])
            bbox = np.array(ann['bbox'])
            img_width, img_height = img['width'], img['height']
            bbox = process_bbox(bbox, img_width, img_height)

            annots[img_path][0].append(kp2d)
            annots[img_path][1].append(kp3d)
            annots[img_path][2].append(intrinsic_params)
        
        if self.split == 'val':
            self.file_paths = list(annots.keys())[::self.test2val_sample_ratio]
            self.annots = {}
            for key in self.file_paths:
                self.annots[key] = annots[key]
            del annots
        elif self.split == 'test':
            self.file_paths = list(annots.keys())
            self.annots = annots
        else:
            print('split', self.split, 'is not recognized!')
            raise NotImplementedError

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
        fx, fy, cx, cy = self.annots[img_name][2][0]
        camMats = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        person_num = len(kp2ds)

        vis_masks = []
        for kp2d in kp2ds:
            vis_masks.append(_check_visible(kp2d,get_mask=True))
        kp2ds = np.concatenate([kp2ds, np.array(vis_masks)[:,:,None]],2)

        img_info = {'imgpath': imgpath, 'image': image, 'kp2ds': kp2ds, 'track_ids': None,\
                'vmask_2d': np.array([[True,False,True] for _ in range(person_num)]), 'vmask_3d': np.array([[True,False,False,False] for _ in range(person_num)]),\
                'kp3ds': kp3ds, 'params': None, 'camMats': camMats, 'img_size': image.shape[:2],'ds': 'mupots'}
        
        return img_info


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


if __name__ == '__main__':
    dataset=MuPoTS(train_flag=False)
    test_dataset(dataset)
    print('Done')