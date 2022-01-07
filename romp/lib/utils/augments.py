import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmenters import compute_paddings_to_reach_aspect_ratio, Crop, Pad
from imgaug.augmentables import Keypoint, KeypointsOnImage

import random
import cv2
import numpy as np
ia.seed(1)

import random
import math
import numpy as np
import torch
from PIL import Image
from PIL import ImageEnhance
import functools
import os, sys
import xml.etree.ElementTree
import matplotlib.pyplot as plt
import skimage.data
import PIL.Image

import sys, os
root_dir = os.path.join(os.path.dirname(__file__),'..')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
import constants


def calc_aabb(ptSets):
    ptLeftTop     = np.array([np.min(ptSets[:,0]),np.min(ptSets[:,1])])
    ptRightBottom = np.array([np.max(ptSets[:,0]),np.max(ptSets[:,1])])

    return np.array([ptLeftTop, ptRightBottom])

def flip_kps(kps, width=None, is_pose=True,flipped_parts=constants.All54_flip):
    if is_pose:
        kps = kps[flipped_parts]
    invalid_mask = kps[:,-1]==-2
    if width is not None:
        kps[:,0] = width - kps[:,0]
    else:
        kps[:,0] = - kps[:,0]
    kps[invalid_mask] = -2
    return kps

def rot_imgplane(kp3d, angle):
    if angle == 0:
        return kp3d
    invalid_mask = kp3d[:,-1]==-2
    # in-plane rotation
    rot_mat = np.eye(3)
    rot_rad = angle * np.pi / 180
    sn,cs = np.sin(rot_rad), np.cos(rot_rad)
    rot_mat[0,:2] = [cs, -sn]
    rot_mat[1,:2] = [sn, cs]
    kp3d = np.einsum('ij,kj->ki', rot_mat, kp3d) 
    kp3d[invalid_mask] = -2
    return kp3d

def rot_aa(aa, rot):
    """Rotate axis angle parameters."""
    # pose parameters
    R = np.array([[np.cos(np.deg2rad(rot)), -np.sin(np.deg2rad(rot)), 0],
                  [np.sin(np.deg2rad(rot)), np.cos(np.deg2rad(rot)), 0],
                  [0, 0, 1]])
    # find the rotation of the body in camera frame
    per_rdg, _ = cv2.Rodrigues(aa)
    # apply the global rotation to the global orientation
    resrot, _ = cv2.Rodrigues(np.dot(R,per_rdg))
    aa = (resrot.T)[0]
    return aa

def flip_pose(pose):
    #Flip pose.The flipping is based on SMPL parameters.

    flipped_parts = constants.SMPL_POSE_FLIP_PERM
    pose = pose[flipped_parts]
    # we also negate the second and the third dimension of the axis-angle
    pose[1::3] = -pose[1::3]
    pose[2::3] = -pose[2::3]
    return pose

def pose_processing(pose, rot, flip, valid_flag):
    """Process SMPL theta parameters  and apply all augmentation transforms."""
    
    # rotation or the pose parameters
    pose[:3] = rot_aa(pose[:3], rot)
    # flip the pose parameters
    if flip and valid_flag[1]:
        pose = flip_pose(pose)
    
    return pose


def image_crop_pad(image, kp2ds=None, crop_trbl=(0,0,0,0), bbox=None, pad_ratio=1., pad_trbl=None, draw_kp_on_image=False):
    '''
    Perform augmentation of image (and kp2ds) via x-y translation, rotation, and scale variation.
    Input args:
        image : np.array, size H x W x 3
        kp2ds : np.array, size N x K x 2/3, the K 2D joints of N people
        crop_trbl : tuple, size 4, represent the cropped size on top, right, bottom, left side, Each entry may be a single int.
        bbox : np.array/list/tuple, size 4, represent the left, top, right, bottom, we can derive the crop_trbl from the bbox
        pad_ratio : float, ratio = width / height
        pad_trbl: np.array/list/tuple, size 4, represent the pad size on top, right, bottom, left side, Each entry may be a single int.
    return:
        augmented image: np.array, size H x W x 3
        augmented kp2ds if given, in the same size as input kp2ds
    '''
    if bbox is not None:
        assert len(bbox) == 4, print('bbox input of image_crop_pad is supposed to be in length 4!, while {} is given'.format(bbox))
        def calc_crop_trbl_from_bbox(bbox, image_shape):
            l,t,r,b = bbox
            h,w = image_shape[:2]
            return (int(max(0,t)), int(max(0,w-r)), int(max(0,h-b)), int(max(0,l)))
        crop_trbl = calc_crop_trbl_from_bbox(bbox, image.shape)
    crop_func = iaa.Sequential([iaa.Crop(px=crop_trbl, keep_size=False)])
    image_aug = np.array(crop_func(image=image))
    if pad_trbl is None:
        pad_trbl = compute_paddings_to_reach_aspect_ratio(image_aug.shape, pad_ratio)
    pad_func = iaa.Sequential([iaa.Pad(px=pad_trbl, keep_size=False)])
    image_aug = pad_func(image=image_aug)

    if kp2ds is not None:
        # org_shape = kp2ds.shape
        # kp2ds_ia = convert2keypointsonimage(kp2ds.reshape(-1, org_shape[-1]), image.shape)
        # kp2ds_aug = pad_func(keypoints=crop_func(keypoints=kp2ds_ia)).to_xy_array().reshape(org_shape)
        leftTop = np.array([[crop_trbl[3]-pad_trbl[3], crop_trbl[0]-pad_trbl[0]]])
        leftTop3 = np.array([[crop_trbl[3]-pad_trbl[3], crop_trbl[0]-pad_trbl[0], 0]])
        kp2ds_aug = [kp2d-leftTop if kp2d.shape[-1]==2 else kp2d-leftTop3 for kp2d in kp2ds]
        # if draw_kp_on_image:
        #     for inds, kp2d in enumerate(kp2ds):
        #         kps = convert2keypointsonimage(kp2d[:,:2], image.shape)
        #         image = kps.draw_on_image(image, size=7)
        #         kps_aug = convert2keypointsonimage(kp2ds_aug[inds,:,:2], image_aug.shape)
        #         image_aug = kps_aug.draw_on_image(image_aug, size=7)
        return image_aug, kp2ds_aug, np.array([*image_aug.shape[:2], *crop_trbl, *pad_trbl])
    else:
        return image_aug, None, np.array([*image_aug.shape[:2], *crop_trbl, *pad_trbl])
    
def image_pad_white_bg(image, pad_trbl=None, pad_ratio=1.,pad_cval=255):
    if pad_trbl is None:
        pad_trbl = compute_paddings_to_reach_aspect_ratio(image.shape, pad_ratio)
    pad_func = iaa.Sequential([iaa.Pad(px=pad_trbl, keep_size=False,pad_mode='constant',pad_cval=pad_cval)])
    image_aug = pad_func(image=image)
    return image_aug, np.array([*image_aug.shape[:2], *[0,0,0,0], *pad_trbl])


def process_image(originImage, full_kp2ds=None, augments=None, is_pose2d=True, multiperson=False):
    crop_trbl, bbox = (0,0,0,0), None

    if augments is not None:
        height, width = originImage.shape[0], originImage.shape[1]
        scale, rot, flip = augments

        if rot != 0:
            originImage, full_kp2ds = img_kp_rotate(originImage, full_kp2ds, rot)

        if flip:
            originImage = np.fliplr(originImage)
            full_kp2ds = [flip_kps(kps_i, width=originImage.shape[1], is_pose=is_2d_pose) for kps_i, is_2d_pose in zip(full_kp2ds, is_pose2d)]

        if not multiperson and is_pose2d.sum()>0:
            kps_vis = full_kp2ds[np.where(np.array(is_pose2d))[0][np.random.randint(is_pose2d.sum())]]
            if (kps_vis[:,2]>0).sum()>2:
                box = calc_aabb(kps_vis[kps_vis[:,2]>0,:2].copy())
                leftTop, rightBottom = np.clip(box[0], 0, width), np.clip(box[1], 0, height)
                [l, t], [r, b] = get_image_cut_box(leftTop, rightBottom, scale)
                bbox = (l,t,r,b)

    orgImage_white_bg, pad_trbl = image_pad_white_bg(originImage)
    if full_kp2ds is None and augments is None:
        return orgImage_white_bg, pad_trbl
    
    image_aug, kp2ds_aug, offsets = image_crop_pad(originImage, kp2ds=full_kp2ds, crop_trbl=crop_trbl, bbox=bbox, pad_ratio=1.)
    return image_aug, orgImage_white_bg, kp2ds_aug, offsets

def get_image_cut_box(leftTop, rightBottom, ExpandsRatio, Center = None):
    ExpandsRatio = [ExpandsRatio, ExpandsRatio, ExpandsRatio, ExpandsRatio]

    def _expand_crop_box(lt, rb, scale):
        center = (lt + rb) / 2.0
        xl, xr, yt, yb = lt[0] - center[0], rb[0] - center[0], lt[1] - center[1], rb[1] - center[1]

        xl, xr, yt, yb = xl * scale[0], xr * scale[1], yt * scale[2], yb * scale[3]
        #expand it
        lt, rb = np.array([center[0] + xl, center[1] + yt]), np.array([center[0] + xr, center[1] + yb])
        lb, rt = np.array([center[0] + xl, center[1] + yb]), np.array([center[0] + xr, center[1] + yt])
        center = (lt + rb) / 2
        return center, lt, rt, rb, lb

    if Center == None:
        Center = (leftTop + rightBottom) // 2

    Center, leftTop, rightTop, rightBottom, leftBottom = _expand_crop_box(leftTop, rightBottom, ExpandsRatio)

    offset = (rightBottom - leftTop) // 2

    cx = offset[0]
    cy = offset[1]

    r = max(cx, cy, 1)

    cx = r
    cy = r

    x = int(Center[0])
    y = int(Center[1])

    return [x - cx, y - cy], [x + cx, y + cy]



class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
    -------------------------------------------------------------------------------------
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, sl = 0.01, sh = 0.03, r1 = 0.4, mean=[0.4914, 0.4822, 0.4465]):
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):
        img_h, img_w, img_c = img.shape
        for attempt in range(100):
            area = img_h * img_w
       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img_w and h < img_h:
                x1 = random.randint(0, img_h - h)
                y1 = random.randint(0, img_w - w)
                img[x1:x1+h, y1:y1+w] = 0

                return img

        return img

RE = RandomErasing()

def random_erase(image):
    return RE(image)

def RGB_mix(image, pn):
    # in the rgb image we add pixel noise in a channel-wise manner
    image[:,:,0] = np.minimum(255.0, np.maximum(0.0, image[:,:,0]*pn[0]))
    image[:,:,1] = np.minimum(255.0, np.maximum(0.0, image[:,:,1]*pn[1]))
    image[:,:,2] = np.minimum(255.0, np.maximum(0.0, image[:,:,2]*pn[2]))
    return image

def convert2keypointsonimage(kp2d, image_shape):
    kps = KeypointsOnImage([Keypoint(x=x, y=y) for x,y in kp2d], shape=image_shape)
    return kps

def img_kp_rotate(image, kp2ds=None, rotate=0):
    '''
    Perform augmentation of image (and kp2ds) via rotation.
    Input args:
        image : np.array, size H x W x 3
        kp2ds : np.array, size N x K x 2/3, the K 2D joints of N people
        rotate : int, radians angle of rotation on image plane, such as 30 degree
    return:
        augmented image: np.array, size H x W x 3
        augmented kp2ds if given, in the same size as input kp2ds
    '''
    aug_list = []
    if rotate != 0:
        aug_list += [iaa.Affine(rotate=rotate)]
        aug_seq = iaa.Sequential(aug_list)
        image_aug = np.array(aug_seq(image=image))
        if kp2ds is not None:
            kp2ds_aug = []
            for idx, kp2d in enumerate(kp2ds):
                kps = convert2keypointsonimage(kp2d[:,:2], image.shape)
                #image = kps.draw_on_image(image, size=7)
                kps_aug = aug_seq(keypoints=kps)
                #image_aug = kps_aug.draw_on_image(image_aug, size=7)
                kp2d[:,:2] = kps_aug.to_xy_array()
                kp2ds_aug.append(kp2d)
        else:
            kp2ds_aug=None

    if kp2ds is not None:
        return image_aug, kp2ds_aug
    else:
        return image_aug

def img_kp_trans_rotate_scale(image, kp2ds=None, rotate=0, trans=None, scale=None):
    '''
    Perform augmentation of image (and kp2ds) via x-y translation, rotation, and scale variation.
    Input args:
        image : np.array, size H x W x 3
        kp2ds : np.array, size N x K x 2/3, the K 2D joints of N people
        rotate : int, radians angle of rotation on image plane, such as 30 degree
        trans : np.array/list/tuple, (tx, ty), translation on the image plane along x, y axis
        scale : np.array/list/tuple, (sx, sy), scale variation on the image plane along x, y axis
    return:
        augmented image: np.array, size H x W x 3
        augmented kp2ds if given, in the same size as input kp2ds
    '''
    aug_list = []
    if trans is not None:
        tx, ty = trans
        aug_list += [iaa.TranslateX(px=tx), iaa.TranslateY(px=ty)]
    if rotate != 0:
        aug_list += [iaa.Affine(rotate=rotate)]
    if scale is not None:
        aug_list += [iaa.Affine(scale=scale)]

    aug_seq = iaa.Sequential(aug_list)
    image_aug = np.array(aug_seq(image=image))
    if kp2ds is not None:
        kp2ds_aug = []
        for idx, kp2d in enumerate(kp2ds):
            kps = convert2keypointsonimage(kp2d[:,:2], image.shape)
            image = kps.draw_on_image(image, size=7)
            kps_aug = aug_seq(keypoints=kps)
            image_aug = kps_aug.draw_on_image(image_aug, size=7)
            kp2d[:,:2] = kps_aug.to_xy_array()
            kp2ds_aug.append(kp2d)
        return image_aug, kp2ds_aug
    else:
        return image_aug



def augment_blur(image):
    choise = np.random.randint(4)
    if choise==0:
        image = cv2.blur(image,(3,3))
    elif choise==1:
        image = cv2.GaussianBlur(image,(3,3),0)
    elif choise==2:
        image = cv2.medianBlur(image,3)
    elif choise==3:
        sigma = np.random.randint(20,30)
        image = cv2.bilateralFilter(image,3,sigma,sigma)
    return image

'''
brought from https://github.com/isarandi/synthetic-occlusion/blob/master/augmentation.py

'''
class Synthetic_occlusion(object):
    def __init__(self,path):
        print('Loading occluders from Pascal VOC dataset...')
        # path = 'something/something/VOCtrainval_11-May-2012/VOCdevkit/VOC2012'
        self.occluders = load_occluders(pascal_voc_root_path=path)
        print('Found {} suitable objects'.format(len(self.occluders)))

    def __call__(self, img):
        occluded_img = occlude_with_objects(img, self.occluders)
        return occluded_img


def load_occluders(pascal_voc_root_path):
    occluders = []
    structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    
    annotation_paths = list_filepaths(os.path.join(pascal_voc_root_path, 'Annotations'))
    for annotation_path in annotation_paths:
        xml_root = xml.etree.ElementTree.parse(annotation_path).getroot()
        is_segmented = (xml_root.find('segmented').text != '0')

        if not is_segmented:
            continue

        boxes = []
        for i_obj, obj in enumerate(xml_root.findall('object')):
            is_person = (obj.find('name').text == 'person')
            is_difficult = (obj.find('difficult').text != '0')
            is_truncated = (obj.find('truncated').text != '0')
            if not is_person and not is_difficult and not is_truncated:
                bndbox = obj.find('bndbox')
                box = [int(bndbox.find(s).text) for s in ['xmin', 'ymin', 'xmax', 'ymax']]
                boxes.append((i_obj, box))

        if not boxes:
            continue

        im_filename = xml_root.find('filename').text
        seg_filename = im_filename.replace('jpg', 'png')

        im_path = os.path.join(pascal_voc_root_path, 'JPEGImages', im_filename)
        seg_path = os.path.join(pascal_voc_root_path,'SegmentationObject', seg_filename)

        im = np.asarray(PIL.Image.open(im_path))
        labels = np.asarray(PIL.Image.open(seg_path))

        for i_obj, (xmin, ymin, xmax, ymax) in boxes:
            object_mask = (labels[ymin:ymax, xmin:xmax] == i_obj + 1).astype(np.uint8)*255
            object_image = im[ymin:ymax, xmin:xmax]
            if cv2.countNonZero(object_mask) < 500:
                # Ignore small objects
                continue

            # Reduce the opacity of the mask along the border for smoother blending
            eroded = cv2.erode(object_mask, structuring_element)
            object_mask[eroded < object_mask] = 192
            object_with_mask = np.concatenate([object_image, object_mask[..., np.newaxis]], axis=-1)
            
            # Downscale for efficiency
            object_with_mask = resize_by_factor(object_with_mask, 0.5)
            occluders.append(object_with_mask)

    return occluders


def occlude_with_objects(im, occluders):
    """Returns an augmented version of `im`, containing some occluders from the Pascal VOC dataset."""

    result = im.copy()
    width_height = np.asarray([im.shape[1], im.shape[0]])
    im_scale_factor = min(width_height) / 256
    count = np.random.randint(1, 8)

    for _ in range(count):
        occluder = random.choice(occluders)

        random_scale_factor = np.random.uniform(0.2, 1.0)
        scale_factor = random_scale_factor * im_scale_factor
        occluder = resize_by_factor(occluder, scale_factor)

        center = np.random.uniform([0,0], width_height)
        paste_over(im_src=occluder, im_dst=result, center=center)

    return result


def paste_over(im_src, im_dst, center):
    """Pastes `im_src` onto `im_dst` at a specified position, with alpha blending, in place.
    Locations outside the bounds of `im_dst` are handled as expected (only a part or none of
    `im_src` becomes visible).
    Args:
        im_src: The RGBA image to be pasted onto `im_dst`. Its size can be arbitrary.
        im_dst: The target image.
        alpha: A float (0.0-1.0) array of the same size as `im_src` controlling the alpha blending
            at each pixel. Large values mean more visibility for `im_src`.
        center: coordinates in `im_dst` where the center of `im_src` should be placed.
    """

    width_height_src = np.asarray([im_src.shape[1], im_src.shape[0]])
    width_height_dst = np.asarray([im_dst.shape[1], im_dst.shape[0]])

    center = np.round(center).astype(np.int32)
    raw_start_dst = center - width_height_src // 2
    raw_end_dst = raw_start_dst + width_height_src

    start_dst = np.clip(raw_start_dst, 0, width_height_dst)
    end_dst = np.clip(raw_end_dst, 0, width_height_dst)
    region_dst = im_dst[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]]

    start_src = start_dst - raw_start_dst
    end_src = width_height_src + (end_dst - raw_end_dst)
    region_src = im_src[start_src[1]:end_src[1], start_src[0]:end_src[0]]
    color_src = region_src[..., 0:3]
    alpha = region_src[..., 3:].astype(np.float32)/255

    im_dst[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]] = (
            alpha * color_src + (1 - alpha) * region_dst)


def resize_by_factor(im, factor):
    """Returns a copy of `im` resized by `factor`, using bilinear interp for up and area interp
    for downscaling.
    """
    new_size = tuple(np.round(np.array([im.shape[1], im.shape[0]]) * factor).astype(int))
    interp = cv2.INTER_LINEAR if factor > 1.0 else cv2.INTER_AREA
    return cv2.resize(im, new_size, fx=factor, fy=factor, interpolation=interp)


def list_filepaths(dirpath):
    names = os.listdir(dirpath)
    paths = [os.path.join(dirpath, name) for name in names]
    return sorted(filter(os.path.isfile, paths))

if __name__ == '__main__':
    image = ia.quokka(size=(512, 256))
    kps = np.array([[[65,100],[75,200],[100,100],[200,80]]])
    bbox = []
    image_aug, pad_trbl = image_pad_white_bg(image)
    print(pad_trbl)
    #image, image_after, kps_aug = img_kp_trans_rotate_scale(image, kp2ds=kps, rotate=30, trans=(10,0)) #, scale=(1,1)
    #image, image_aug, kp2ds_aug = image_crop_pad(image, kp2ds=kps, crop_trbl=(20,30,40,50), bbox=None, pad_ratio=1., pad_trbl=None, draw_kp_on_image=True)
    #ia.imshow(image)
    ia.imshow(image_aug)
    #cv2.imwrite('image_before_after.png', np.concatenate([image,image_after], 1))
