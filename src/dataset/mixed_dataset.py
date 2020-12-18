import torch
import numpy as np
import os,sys
sys.path.append(os.path.abspath(__file__).replace('dataset/mixed_dataset.py',''))
from dataset.internet import Internet
import config
from config import args

dataset_dict = {'internet':Internet}


class SingleDataset(torch.utils.data.Dataset):
    def __init__(self, dataset=None,**kwargs):
        assert dataset in dataset_dict, print('dataset {} not found while creating data loader!'.format(dataset))
        self.dataset = dataset_dict[dataset](**kwargs)
        self.length = len(self.dataset)            

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return self.length


if __name__ == '__main__':
    data_loader = SingleDataset(dataset='pw3d', joint_format='smpl24',train_flag=False,use_openpose_center=True, split='all')
    #data_loader = MixedDataset(train_flag=True)
    openpose_results = {}
    for idx in range(len(data_loader)):
        data  = data_loader.__getitem__(idx)
        full_kp2d,subject_ids,imgpath  = data['full_kp2d'], data['subject_ids'], data['imgpath']
        action_name = imgpath.split('/')[-2]
        frame_id = int(imgpath.split('/')[-1].replace('.jpg','').replace('image_',''))
        print(action_name, frame_id)
        if action_name not in openpose_results:
            openpose_results[action_name] = {}
        
        valid_id = subject_ids>-1
        for subject_id,op_kp2d in zip(subject_ids[valid_id],full_kp2d[valid_id]):
            subject_id = int(subject_id)
            if subject_id not in openpose_results[action_name]:
                openpose_results[action_name][subject_id] = {}
            openpose_results[action_name][subject_id][frame_id] = op_kp2d.cpu().numpy()
        if idx==400:
            np.savez('3dpw_openpose_results.npz', results=openpose_results)

    np.savez('3dpw_openpose_results.npz', results=openpose_results)

    for action_name in openpose_results:
        kp2ds = []
        for subject_id in openpose_results[action_name]:
            frame_ids = list(openpose_results[action_name][subject_id].keys())
            frame_ids.sort()
            kp2ds_subject = []
            for frame_id in frame_ids:
                kp2ds_subject.append(openpose_results[action_name][subject_id][frame_id])
            kp2ds_subject = np.array(kp2ds_subject)
            kp2ds.append(kp2ds_subject)
        kp2ds = np.array(kp2ds)
        np.savez('openpose_results/{}.npz'.format(action_name),kp2ds=kp2ds)