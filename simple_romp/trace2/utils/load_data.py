import glob
import random
import cv2
import torch
import os
import os.path as osp
import numpy as np
from torch.utils.data import Dataset, DataLoader

default_frame_dir = os.path.join(os.path.expanduser('~'), 'TRACE_input_frames')

def video2frame(video_path, frame_save_dir=None):
    cap = cv2.VideoCapture(video_path)
    for frame_id in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        success_flag, frame = cap.read()
        if success_flag:
            save_path = os.path.join(frame_save_dir, '{:08d}.jpg'.format(frame_id))
            cv2.imwrite(save_path, frame)

def collect_frame_path(video_path, save_dir):
    assert osp.exists(video_path), video_path + 'not exist!'
    video_name, video_ext = osp.splitext(osp.basename(video_path))
    
    frame_save_dir = osp.join(save_dir, video_name+'_frames')
    print(f'Extracting the frames of input {video_path} to {frame_save_dir}')
    os.makedirs(frame_save_dir, exist_ok=True)
    try:
        video2frame(video_path, frame_save_dir)
    except:
        raise Exception(f"Failed in extracting the frames of {video_path} to {frame_save_dir}! \
            Please check the video. If you want to do this by yourself, please extracte frames to {frame_save_dir} and take it as input to ROMP. \
            For example, the first frame name is supposed to be {osp.join(frame_save_dir, '00000000.jpg')}")

    assert osp.isdir(frame_save_dir), frame_save_dir + 'is supposed to be a folder containing video frames.'
    frame_paths = [osp.join(frame_save_dir, frame_name) for frame_name in sorted(os.listdir(frame_save_dir))]
    return frame_paths

def prepare_video_frame_dict(video_path_list, img_ext='jpg', frame_dir=default_frame_dir):
    video_frame_dict = {}
    for video_path in video_path_list:
        if len(os.path.splitext(os.path.basename(video_path))[1])>0: # .mp4 files
            frame_list = collect_frame_path(video_path, frame_dir)
        else: # folder contains 
            frame_list = sorted(glob.glob(os.path.join(video_path,'*.'+img_ext)))
            if len(frame_list) ==0:
                frame_list = sorted(glob.glob(os.path.join(video_path,'*.png')))
        video_frame_dict[video_path] = frame_list
    return video_frame_dict

class InternetVideo(Dataset):
    def __init__(self, sequence_dict, **kwargs):
        super(InternetVideo,self).__init__()
        self.prepare_video_sequence(sequence_dict)
        self.video_clip_ids = self.split_sequence2clips()
        print('Loading {} image sequences to process'.format(len(self)))
    
    def prepare_video_sequence(self, sequence_dict):
        self.sequence_dict = sequence_dict
        self.file_paths, self.sequence_ids, self.sid_video_name = [], [], {}
        for sid, video_path in enumerate(self.sequence_dict):
            video_name = os.path.basename(video_path)
            #if seq_name_level == 2:
            #    video_name = video_path.split(os.path.sep)[-2]+'-'+video_name
            self.sid_video_name[sid] = video_name
            self.sequence_ids.append([])
            for fid, frame_path in enumerate(self.sequence_dict[video_path]):
                self.file_paths.append([sid,fid,os.path.join(video_path, frame_path)])
                self.sequence_ids[sid].append(len(self.file_paths)-1)
        print('sid_video_name:',self.sid_video_name)

    def get_image_info(self,index):
        return self.file_paths[index]
    
    def __len__(self):
        return len(self.video_clip_ids)

    def get_item_single_frame(self, index, **kwargs):
        sid, frame_id, imgpath = self.get_image_info(index)
        end_frame_flag = frame_id == (len(self.sequence_ids[sid])-1)
        image = cv2.imread(imgpath)
        if image is None:
            print('image load None', imgpath)

        input_data = img_preprocess(image, imgpath, input_size=512)
        input_data['seq_info'] = torch.Tensor([sid, frame_id, end_frame_flag])
        return input_data

    def split_sequence2clips(self, clip_max_length=8): # 这里必须是8，不然GRU那里的隐状态就不对了。
        video_clip_ids = []
        seq_end_flag = []
        
        for sid, seq_sample_ids in enumerate(self.sequence_ids):
            seq_length = len(seq_sample_ids)
            clip_num = int(np.ceil(seq_length/clip_max_length))
            for clip_id in range(clip_num):
                video_clip_ids.append([sid, seq_sample_ids[clip_max_length*clip_id : clip_max_length*(clip_id+1)]])
                seq_end_flag.append(clip_id==(clip_num-1))
        self.seq_end_flag = seq_end_flag
        return video_clip_ids

    def collect_entire_sequence_inputs(self, index):
        sequence_id, frame_ids = self.video_clip_ids[index]
        frame_data = [None for _ in range(len(frame_ids))]
        augment_cfgs = (None,None)
        for cid, fid in enumerate(frame_ids):
            frame_data[cid] = self.get_item_single_frame(fid, augment_cfgs=augment_cfgs)
        seq_data = self.pack_clip_data(frame_data)
        seq_data['seq_end_flag'] = torch.tensor([self.seq_end_flag[index]])
        seq_data['seq_name'] = self.sid_video_name[sequence_id]
        return seq_data
    
    def pack_clip_data(self, clip_data):
        all_data = {}
        for key in clip_data[0].keys():
            if isinstance(clip_data[0][key], torch.Tensor):
                all_data[key] = torch.stack([data[key] for data in clip_data])
            elif isinstance(clip_data[0][key], str):
                all_data[key] = [data[key] for data in clip_data]
                # dataloader will collect list to T x B (7x16), not B x T (16x7) as we espect. 
            elif isinstance(clip_data[0][key], int):
                all_data[key] = torch.Tensor([data[key] for data in clip_data])
        
        return all_data

    def __getitem__(self, index):
        return self.collect_entire_sequence_inputs(index)

def image_pad_white_bg(image, pad_trbl=None, pad_ratio=1.,pad_cval=255):
    import imgaug.augmenters as iaa
    if pad_trbl is None:
        from imgaug.augmenters import compute_paddings_to_reach_aspect_ratio
        pad_trbl = compute_paddings_to_reach_aspect_ratio(image.shape, pad_ratio)
    pad_func = iaa.Sequential([iaa.Pad(px=pad_trbl, keep_size=False,pad_mode='constant',pad_cval=pad_cval)])
    image_aug = pad_func(image=image)
    return image_aug, np.array([*image_aug.shape[:2], *[0,0,0,0], *pad_trbl])

def img_preprocess(image, imgpath, input_size=512, ds='internet', single_img_input=False):
    image = image[:,:,::-1]
    image_org, offsets = image_pad_white_bg(image)
    image = torch.from_numpy(cv2.resize(image_org, (input_size,input_size), interpolation=cv2.INTER_CUBIC))
    
    offsets = torch.from_numpy(offsets).float()
    name = os.path.basename(imgpath)

    if single_img_input:
        image = image.unsqueeze(0).contiguous()
        offsets = offsets.unsqueeze(0).contiguous()
        imgpath, name, ds = [imgpath], [name], [ds]
    input_data = {'image': image, 'imgpath': imgpath, 'offsets': offsets, 'data_set': ds}
    return input_data

def prepare_data_loader(sequence_dict, val_batch_size):
    datasets = InternetVideo(sequence_dict)
    data_loader = DataLoader(dataset=datasets, shuffle=False, batch_size=val_batch_size, drop_last=False, pin_memory=True)
    return data_loader

def extract_seq_data(meta_data, seq_num = 1):
    collect_items = ['image', 'data_set', 'imgpath', 'offsets']
    seq_data = {key:[] for key in collect_items}

    for key in seq_data:
        if isinstance(meta_data[key], torch.Tensor):
            seq_data[key].append(meta_data[key])
        elif isinstance(seq_data[key], list):
            seq_data[key] += [i[0] for i in meta_data[key]]
    seq_name = meta_data['seq_name'][0]
    
    for key in collect_items:
        if isinstance(seq_data[key][0], torch.Tensor):
            seq_data[key] = torch.cat(seq_data[key],1).squeeze(0)
    
    #clip_length = len(seq_data['image'])
    ##seq_data['seq_inds'] = torch.stack([torch.arange(seq_num).unsqueeze(1).repeat(1,clip_length).reshape(-1),\
    #                                    torch.arange(clip_length).unsqueeze(0).repeat(seq_num,1).reshape(-1),\
    #                                    torch.arange(seq_num*clip_length), torch.ones(seq_num*clip_length)], 1).long()
    return seq_data, seq_name