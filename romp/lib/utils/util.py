#encoding=utf-8
import h5py
import torch
import numpy as np
import json
import torch.nn.functional as F
import cv2
import math
import hashlib
import shutil
import pickle
import yaml
import csv
import platform
import os,sys
import glob
from io import BytesIO
from scipy.spatial.transform import Rotation as R

TAG_CHAR = np.array([202021.25], np.float32)

def get_kp2d_on_org_img(kp2d, offset):
    assert kp2d.shape[1]>=2, print('Espected shape of kp2d is Kx2, while get {}'.formt(kp2d.shape))
    if torch.is_tensor(offset):
        offset = offset.detach().cpu().numpy()
    pad_size_h,pad_size_w, lt_h,rb_h,lt_w,rb_w,offset_h,size_h,offset_w,size_w,length = offset
    kp2d_onorg = np.ones_like(kp2d)
    kp2d_onorg[:,0] = (kp2d[:,0]+1)/2 * pad_size_w - offset_w+lt_w
    kp2d_onorg[:,1] = (kp2d[:,1]+1)/2 * pad_size_h - offset_h+lt_w
    return kp2d_onorg
    

class AverageMeter_Dict(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.dict_store = {}
        self.count = 0

    def update(self, val, n=1):
        for key,value in val.items():
            if key not in self.dict_store:
                self.dict_store[key] = []
            if torch.is_tensor(value):
                value = value.item()
            self.dict_store[key].append(value)
        self.count += n
    
    def sum(self):
        dict_sum = {}
        for k, v in self.dict_store.items():
            dict_sum[k] = round(float(sum(v)),2)
        return dict_sum

    def avg(self):
        dict_sum = self.sum()
        dict_avg = {}
        for k,v in dict_sum.items():
            dict_avg[k] = round(v/self.count,2)
        return dict_avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def normalize_kps(kps,image_shape,resize=512,set_minus=True):
    kps[:,0] *= 1.0 * resize / image_shape[0]
    kps[:,1] *= 1.0 * resize / image_shape[1]
    kps[:,:2] = 2.0 * kps[:,:2] / resize - 1.0

    if kps.shape[1]>2 and set_minus:
        kps[kps[:,2]<0.1,:2] = -2.
    kps=kps[:,:2]
    return kps

#______________IO__________________


def collect_image_list(image_folder=None, collect_subdirs=False, img_exts=None):
    
    def collect_image_from_subfolders(image_folder, file_list, collect_subdirs, img_exts):
        for path in glob.glob(os.path.join(image_folder,'*')):
            if os.path.isdir(path) and collect_subdirs:
                collect_image_from_subfolders(path, file_list, collect_subdirs, img_exts)
            elif os.path.splitext(path)[1] in img_exts:
                file_list.append(path)
        return file_list

    file_list = collect_image_from_subfolders(image_folder, [], collect_subdirs, img_exts)

    return file_list


def save_result_dict_tonpz(results, test_save_dir):
    for img_path, result_dict in results.items():
        if platform.system() == 'Windows':
            path_list = img_path.split('\\')
        else:
            path_list = img_path.split('/')
        file_name = '_'.join(path_list)
        file_name = '_'.join(os.path.splitext(file_name)).replace('.','') + '.npz'
        save_path = os.path.join(test_save_dir, file_name)
        # get the results: np.load('/path/to/person_overlap.npz',allow_pickle=True)['results'][()]
        np.savez(save_path, results=result_dict)


def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )

    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf

def plt2np(plt):
    #申请缓冲地址
    buffer_ = BytesIO()#using buffer,great way!
    #保存在内存中，而不是在本地磁盘，注意这个默认认为你要保存的就是plt中的内容
    plt.savefig(buffer_,format = 'png')
    buffer_.seek(0)
    #用PIL或CV2从内存中读取
    dataPIL = Image.open(buffer_)
    #转换为nparrary，PIL转换就非常快了,data即为所需
    data = np.asarray(dataPIL)

    buffer_.close()
    return data

def save_pkl(info,name='../data/info.pkl'):
    check_file_and_remake(name.replace(os.path.basename(name),''))
    if name[-4:] !='.pkl':
        name += '.pkl'
    with open(name,'wb') as outfile:
        pickle.dump(info, outfile, pickle.HIGHEST_PROTOCOL)

def save_yaml(dict_file,path):
    with open(path, 'w') as file:
        documents = yaml.dump(dict_file, file)

def read_pkl(name = '../data/info.pkl'):
    with open(name,'rb') as f:
        return pickle.load(f)

def read_pkl_coding(name = '../data/info.pkl'):
    with open(name, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        p = u.load()
    return p

def check_file_and_remake(path,remove=False):
    if remove:
        if os.path.isdir(path):
            shutil.rmtree(path)
    if not os.path.isdir(path):
        os.makedirs(path)

def save_h5(info,name):
    check_file_and_remake(name.replace(os.path.basename(name),''))
    if name[-3:] !='.h5':
        name += '.h5'
    f=h5py.File(name,'w')
    for item, value in info.items():
        f[item] = value
    f.close()

def read_h5(name):
    if name[-3:] !='.h5':
        name += '.h5'
    f=h5py.File(name,'r')
    info = {}
    for item, value in f.items():
        info[item] = np.array(value)
    f.close()
    return info

def save_obj(verts, faces, obj_mesh_name='mesh.obj'):
    #print('Saving:',obj_mesh_name)
    with open(obj_mesh_name, 'w') as fp:
        for v in verts:
            fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )

        for f in faces: # Faces are 1-based, not 0-based in obj files
            fp.write( 'f %d %d %d\n' %  (f[0] + 1, f[1] + 1, f[2] + 1) )

def save_json(dicts, name):
    json_str = json.dumps(dicts)
    with open(name, 'w') as json_file:
        json_file.write(json_str)


#______________model tools__________________
def BHWC_to_BCHW(x):
    """
    :param x: torch tensor, B x H x W x C
    :return:  torch tensor, B x C x H x W
    """
    return x.unsqueeze(1).transpose(1, -1).squeeze(-1)

#______________interesting tools__________________

def wrap(func, *args, unsqueeze=False):
    """
    对pytorch的函数进行封装，使其可以被nparray调用。
    Wrap a torch function so it can be called with NumPy arrays.
    Input and return types are seamlessly converted.
    """

    # Convert input types where applicable
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)
            if unsqueeze:
                args[i] = args[i].unsqueeze(0)

    result = func(*args)

    # Convert output types where applicable
    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) == torch.Tensor:
                if unsqueeze:
                    res = res.squeeze(0)
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) == torch.Tensor:
        if unsqueeze:
            result = result.squeeze(0)
        return result.numpy()
    else:
        return result


def write_words2img(img,height_use,words,line_length=20,line_height=36,width_min=1420,color=(0, 0 ,0),duan_space=True):
    font = ImageFont.truetype("/export/home/suny/shoes_attributes/data/song.ttf", 28)

    words_list = [words[i:i+line_length] for i in range(0,len(words),line_length)]
    w,h = img.size

    if height_use==0:
        img=np.asarray(img)
        img_new = np.zeros((h,w+width_min,3),dtype=np.uint8)
        img_new[:,:,:] = 255
        try:
            img_new[:h,:w,:] = img
        except Exception as error:
            print(error)
            return None,height_use,False
        img = Image.fromarray(np.uint8(img_new))
        w+=width_min

    if h<height_use+line_height*len(words_list)+1:
        img=np.asarray(img)
        img_new = np.zeros((height_use+line_height*(len(words_list)+1),w,3),dtype=np.uint8)
        img_new[:,:,:] = 255
        img_new[:h,:w,:] = img
        img = Image.fromarray(np.uint8(img_new))

    draw = ImageDraw.Draw(img)

    words = str(words)

    for num, line in enumerate(words_list):
        if num==0 and duan_space:
            height_use += line_height
            draw.text((w-width_min+10,height_use),line, fill = (255,0,0),font=font)
        else:
            draw.text((w-width_min+10,height_use),line, fill = color,font=font)
        height_use += line_height

    return img,height_use,True

def shrink(leftTop, rightBottom, width, height):
    xl = -leftTop[0]
    xr = rightBottom[0] - width

    yt = -leftTop[1]
    yb = rightBottom[1] - height

    cx = (leftTop[0] + rightBottom[0]) / 2
    cy = (leftTop[1] + rightBottom[1]) / 2

    r = (rightBottom[0] - leftTop[0]) / 2

    sx = max(xl, 0) + max(xr, 0)
    sy = max(yt, 0) + max(yb, 0)

    if (xl <= 0 and xr <= 0) or (yt <= 0 and yb <=0):
        return leftTop, rightBottom
    elif leftTop[0] >= 0 and leftTop[1] >= 0 : # left top corner is in box
        l = min(yb, xr)
        r = r - l / 2
        cx = cx - l / 2
        cy = cy - l / 2
    elif rightBottom[0] <= width and rightBottom[1] <= height : # right bottom corner is in box
        l = min(yt, xl)
        r = r - l / 2
        cx = cx + l / 2
        cy = cy + l / 2
    elif leftTop[0] >= 0 and rightBottom[1] <= height : #left bottom corner is in box
        l = min(xr, yt)
        r = r - l  / 2
        cx = cx - l / 2
        cy = cy + l / 2
    elif rightBottom[0] <= width and leftTop[1] >= 0 : #right top corner is in box
        l = min(xl, yb)
        r = r - l / 2
        cx = cx + l / 2
        cy = cy - l / 2
    elif xl < 0 or xr < 0 or yb < 0 or yt < 0:
        return leftTop, rightBottom
    elif sx >= sy:
        sx = max(xl, 0) + max(0, xr)
        sy = max(yt, 0) + max(0, yb)
        # cy = height / 2
        if yt >= 0 and yb >= 0:
            cy = height / 2
        elif yt >= 0:
            cy = cy + sy / 2
        else:
            cy = cy - sy / 2
        r = r - sy / 2

        if xl >= sy / 2 and xr >= sy / 2:
            pass
        elif xl < sy / 2:
            cx = cx - (sy / 2 - xl)
        else:
            cx = cx + (sy / 2 - xr)
    elif sx < sy:
        cx = width / 2
        r = r - sx / 2
        if yt >= sx / 2 and yb >= sx / 2:
            pass
        elif yt < sx / 2:
            cy = cy - (sx / 2 - yt)
        else:
            cy = cy + (sx / 2 - yb)


    return [cx - r, cy - r], [cx + r, cy + r]

def calc_aabb_batch(ptSets_batch):
    batch_size = ptSets_batch.shape[0]
    ptLeftTop     = np.array([np.min(ptSets_batch[:,:,0],axis=1),np.min(ptSets_batch[:,:,1],axis=1)]).T
    ptRightBottom = np.array([np.max(ptSets_batch[:,:,0],axis=1),np.max(ptSets_batch[:,:,1],axis=1)]).T
    bbox = np.concatenate((ptLeftTop.reshape(batch_size,1,2),ptRightBottom.reshape(batch_size,1,2)),axis=1)
    return bbox
'''
    calculate a obb for a set of points
    inputs:
        ptSets: a set of points
    return the center and 4 corners of a obb
'''
def calc_obb(ptSets):
    ca = np.cov(ptSets,y = None,rowvar = 0,bias = 1)
    v, vect = np.linalg.eig(ca)
    tvect = np.transpose(vect)
    ar = np.dot(ptSets,np.linalg.inv(tvect))
    mina = np.min(ar,axis=0)
    maxa = np.max(ar,axis=0)
    diff    = (maxa - mina)*0.5
    center  = mina + diff
    corners = np.array([center+[-diff[0],-diff[1]],center+[diff[0],-diff[1]],center+[diff[0],diff[1]],center+[-diff[0],diff[1]]])
    corners = np.dot(corners, tvect)
    return corners[0], corners[1], corners[2], corners[3]




#__________________transform tools_______________________


def transform_rot_representation(rot, input_type='mat',out_type='vec'):
    '''
    make transformation between different representation of 3D rotation
    input_type / out_type (np.array):
        'mat': rotation matrix (3*3)
        'quat': quaternion (4)
        'vec': rotation vector (3)
        'euler': Euler degrees in x,y,z (3)
    '''
    if input_type=='mat':
        r = R.from_matrix(rot)
    elif input_type=='quat':
        r = R.from_quat(rot)
    elif input_type =='vec':
        r = R.from_rotvec(rot)
    elif input_type =='euler':
        if rot.max()<4:
            rot = rot*180/np.pi
        r = R.from_euler('xyz',rot, degrees=True)
    
    if out_type=='mat':
        out = r.as_matrix()
    elif out_type=='quat':
        out = r.as_quat()
    elif out_type =='vec':
        out = r.as_rotvec()
    elif out_type =='euler':
        out = r.as_euler('xyz', degrees=False)
    return out


def compute_similarity_transform(S1, S2):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    # 7. Error:
    S1_hat = scale*R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat


def batch_rodrigues(param):
    #param N x 3
    batch_size = param.shape[0]
    #沿第二维（3个数）进行求二次范数：||x||，下面就是进行标准化，每三个数除以他们的范数。
    l1norm = torch.norm(param + 1e-8, p = 2, dim = 1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(param, angle)
    angle = angle * 0.5
    #上面算出的是一个向量的长度：sqrt(x**2+y**2+z**2)/2,所以这个长度的的cos
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    #用四元组表示三维旋转，有时间看一下×××××××××
    quat = torch.cat([v_cos, v_sin * normalized], dim = 1)

    return quat2mat(quat)

def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    把四元组的系数转化成旋转矩阵。四元组表示三维旋转
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat

def batch_global_rigid_transformation(Rs, Js, parent, rotate_base = False,root_rot_mat =None):
    '''
    进行成堆的全局刚性变换。
    '''
    N = Rs.shape[0]
    #确定根节点的旋转变换。
    if rotate_base:
        np_rot_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype = np.float)
        np_rot_x = np.reshape(np.tile(np_rot_x, [N, 1]), [N, 3, 3])
        rot_x = torch.from_numpy(np_rot_x).float().cuda()
        root_rotation = torch.matmul(Rs[:, 0, :, :],  rot_x)
    elif root_rot_mat is not None:
        np_rot_x = np.reshape(np.tile(root_rot_mat, [N, 1]), [N, 3, 3])
        rot_x =torch.from_numpy(np_rot_x).float().cuda()
        root_rotation = torch.matmul(Rs[:, 0, :, :],  rot_x)
    else:
        root_rotation = Rs[:, 0, :, :]
    Js = torch.unsqueeze(Js, -1)

    def make_A(R, t):
        R_homo = F.pad(R, [0, 0, 0, 1, 0, 0])
        t_homo = torch.cat([t, torch.ones(N, 1, 1).cuda()], dim = 1)
        return torch.cat([R_homo, t_homo], 2)

    A0 = make_A(root_rotation, Js[:, 0])
    results = [A0]

    for i in range(1, parent.shape[0]):
        j_here = Js[:, i] - Js[:, parent[i]]
        A_here = make_A(Rs[:, i], j_here)
        res_here = torch.matmul(results[parent[i]], A_here)
        results.append(res_here)

    results = torch.stack(results, dim = 1)

    new_J = results[:, :, :3, 3]
    #print('result',results)
    Js_w0 = torch.cat([Js, torch.zeros(N, 24, 1, 1).cuda()], dim = 2)
    #print('js w ',Js_w0)
    init_bone = torch.matmul(results, Js_w0)
    #print('init_bone before padded',init_bone)
    init_bone = F.pad(init_bone, [3, 0, 0, 0, 0, 0, 0, 0])
    #print('init_bone padded',init_bone)
    A = results - init_bone
    #print('new_J:',new_J)

    return new_J, A

def batch_global_rigid_transformation_cpu(Rs, Js, parent, rotate_base = False,root_rot_mat =None):
    '''
    进行成堆的全局刚性变换。
    '''
    N = Rs.shape[0]
    #确定根节点的旋转变换。
    if rotate_base:
        np_rot_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype = np.float)
        np_rot_x = np.reshape(np.tile(np_rot_x, [N, 1]), [N, 3, 3])
        rot_x =torch.from_numpy(np_rot_x).float()
        root_rotation = torch.matmul(Rs[:, 0, :, :],  rot_x)
    elif root_rot_mat is not None:
        np_rot_x = np.reshape(np.tile(root_rot_mat, [N, 1]), [N, 3, 3])
        rot_x =torch.from_numpy(np_rot_x).float()
        root_rotation = torch.matmul(Rs[:, 0, :, :],  rot_x)
    else:
        root_rotation = Rs[:, 0, :, :]
    Js = torch.unsqueeze(Js, -1)

    def make_A(R, t):
        R_homo = F.pad(R, [0, 0, 0, 1, 0, 0])
        t_homo = torch.cat([t, torch.ones(N, 1, 1)], dim = 1)
        return torch.cat([R_homo, t_homo], 2)

    A0 = make_A(root_rotation, Js[:, 0])
    results = [A0]

    for i in range(1, parent.shape[0]):
        j_here = Js[:, i] - Js[:, parent[i]]
        A_here = make_A(Rs[:, i], j_here)
        res_here = torch.matmul(results[parent[i]], A_here)
        results.append(res_here)

    results = torch.stack(results, dim = 1)

    new_J = results[:, :, :3, 3]
    Js_w0 = torch.cat([Js, torch.zeros(N, 24, 1, 1)], dim = 2)
    init_bone = torch.matmul(results, Js_w0)
    init_bone = F.pad(init_bone, [3, 0, 0, 0, 0, 0, 0, 0])
    A = results - init_bone

    return new_J, A

def batch_lrotmin(param):
    param = param[:,3:].contiguous()
    Rs = batch_rodrigues(param.view(-1, 3))
    e = torch.eye(3).float()
    Rs = Rs.sub(1.0, e)

    return Rs.view(-1, 23 * 9)


def rotation_matrix_to_angle_axis(rotation_matrix):
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert 3x4 rotation matrix to Rodrigues vector

    Args:
        rotation_matrix (Tensor): rotation matrix.

    Returns:
        Tensor: Rodrigues vector transformation.

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 3)`

    Example:
        >>> input = torch.rand(2, 3, 4)  # Nx4x4
        >>> output = tgm.rotation_matrix_to_angle_axis(input)  # Nx3
    """
    if rotation_matrix.shape[1:] == (3,3):
        rot_mat = rotation_matrix.reshape(-1, 3, 3)
        hom = torch.tensor([0, 0, 1], dtype=torch.float32,
                           device=rotation_matrix.device).reshape(1, 3, 1).expand(rot_mat.shape[0], -1, -1)
        rotation_matrix = torch.cat([rot_mat, hom], dim=-1)

    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    aa = quaternion_to_angle_axis(quaternion)
    aa[torch.isnan(aa)] = 0.0
    return aa


def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert quaternion vector to angle axis of rotation.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (torch.Tensor): tensor with quaternions.

    Return:
        torch.Tensor: tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = tgm.quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape Nx4 or 4. Got {}"
                         .format(quaternion.shape))
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta))

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert 3x4 rotation matrix to 4d quaternion vector

    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.

    Return:
        Tensor: the rotation in quaternion

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`

    Example:
        >>> input = torch.rand(4, 3, 4)  # Nx3x4
        >>> output = tgm.rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(rotation_matrix)))

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape))
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError(
            "Input size must be a N x 3 x 4  tensor. Got {}".format(
                rotation_matrix.shape))

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * ~mask_d0_d1
    mask_c2 = ~mask_d2 * mask_d0_nd1
    mask_c3 = ~mask_d2 * ~mask_d0_nd1
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa
    q *= 0.5
    return q

#__________________intersection tools_______________________

'''
    return whether two segment intersect
'''
def line_intersect(sa, sb):
    al, ar, bl, br = sa[0], sa[1], sb[0], sb[1]
    assert al <= ar and bl <= br
    if al >= br or bl >= ar:
        return False
    return True

'''
    return whether two rectangle intersect
    ra, rb left_top point, right_bottom point
'''
def rectangle_intersect(ra, rb):
    ax = [ra[0][0], ra[1][0]]
    ay = [ra[0][1], ra[1][1]]

    bx = [rb[0][0], rb[1][0]]
    by = [rb[0][1], rb[1][1]]

    return line_intersect(ax, bx) and line_intersect(ay, by)

def get_intersected_rectangle(lt0, rb0, lt1, rb1):
    if not rectangle_intersect([lt0, rb0], [lt1, rb1]):
        return None, None

    lt = lt0.copy()
    rb = rb0.copy()

    lt[0] = max(lt[0], lt1[0])
    lt[1] = max(lt[1], lt1[1])

    rb[0] = min(rb[0], rb1[0])
    rb[1] = min(rb[1], rb1[1])
    return lt, rb

def get_union_rectangle(lt0, rb0, lt1, rb1):
    lt = lt0.copy()
    rb = rb0.copy()

    lt[0] = min(lt[0], lt1[0])
    lt[1] = min(lt[1], lt1[1])

    rb[0] = max(rb[0], rb1[0])
    rb[1] = max(rb[1], rb1[1])
    return lt, rb

def get_rectangle_area(lt, rb):
    return (rb[0] - lt[0]) * (rb[1] - lt[1])

def get_rectangle_intersect_ratio(lt0, rb0, lt1, rb1):
    (lt0, rb0), (lt1, rb1) = get_intersected_rectangle(lt0, rb0, lt1, rb1), get_union_rectangle(lt0, rb0, lt1, rb1)

    if lt0 is None:
        return 0.0
    else:
        return 1.0 * get_rectangle_area(lt0, rb0) / get_rectangle_area(lt1, rb1)