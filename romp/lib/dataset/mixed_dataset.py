import torch
import numpy as np
from .h36m import H36M
from .cmu_panoptic_eval import CMU_Panoptic_eval
from .mpii import MPII
from .AICH import AICH
from .up import UP
from .pw3d import PW3D
from .internet import Internet
from .coco14 import COCO14
from .lsp import LSP
from .posetrack import Posetrack
from .crowdpose import Crowdpose
from .crowdhuman import CrowdHuman
from .mpi_inf_3dhp import MPI_INF_3DHP
from .mpi_inf_3dhp_test import MPI_INF_3DHP_TEST
from .mpi_inf_3dhp_validation import MPI_INF_3DHP_VALIDATION
from .MuCo import MuCo
from .MuPoTS import MuPoTS

import sys, os
from prettytable import PrettyTable

from dataset.image_base import *
import config
from config import args
from collections import OrderedDict

dataset_dict = {'h36m': H36M, 'mpii': MPII, 'coco': COCO14, 'posetrack':Posetrack, 'aich':AICH, 'pw3d':PW3D, 'up':UP, 'crowdpose':Crowdpose, 'crowdhuman':CrowdHuman,\
 'lsp':LSP, 'mpiinf':MPI_INF_3DHP,'mpiinf_val':MPI_INF_3DHP_VALIDATION,'mpiinf_test':MPI_INF_3DHP_TEST, 'muco':MuCo, 'mupots':MuPoTS, \
 'cmup':CMU_Panoptic_eval,'internet':Internet, }

class MixedDataset(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        datasets_used = args().dataset.split(',')
        self.datasets = [dataset_dict[ds](**kwargs) for ds in datasets_used]

        self.lengths, self.partition, self.ID_num_list, self.ID_num = [], [], [], 0
        sample_prob_dict = args().sample_prob_dict
        if not 1.0001>sum(sample_prob_dict.values())>0.999:
            print('CAUTION: The sum of sampling rates is supposed to be 1, while currently we have {}, \n please properly set the sample_prob_dict {} in config.yml'\
                .format(sum(sample_prob_dict.values()), sample_prob_dict.values()))
        for ds_idx, ds_name in enumerate(datasets_used):
            self.lengths.append(len(self.datasets[ds_idx]))
            self.partition.append(sample_prob_dict[ds_name])
            if self.datasets[ds_idx].ID_num>0:
                self.ID_num_list.append(self.ID_num)
                self.ID_num += self.datasets[ds_idx].ID_num
            else:
                self.ID_num_list.append(0)
        dataset_info_table = PrettyTable([' ']+datasets_used)
        dataset_info_table.add_row(['Length']+self.lengths)
        dataset_info_table.add_row(['Sample Prob.']+self.partition)
        expect_length = (np.array(self.lengths)/np.array(self.partition)).astype(np.int)
        dataset_info_table.add_row(['Expected length']+expect_length.tolist())
        self.partition = np.array(self.partition).cumsum()
        dataset_info_table.add_row(['Accum. Prob.']+self.partition.astype(np.float16).tolist())
        dataset_info_table.add_row(['Accum. ID.']+self.ID_num_list)
        print(dataset_info_table)
        self.total_length = int(expect_length.max())
        logging.info('All dataset length: {}'.format(len(self)))

    def _get_ID_num_(self):
        return self.ID_num

    def __getitem__(self, index):
        p = float(index)/float(self.total_length)
        dataset_id = len(self.partition)-(self.partition>=p).sum()

        upper_bound = self.partition[dataset_id]
        lower_bound = self.partition[dataset_id-1] if dataset_id>0 else 0
        sample_prob = (p-lower_bound)/(upper_bound-lower_bound)

        omit_internal = self.lengths[dataset_id]//((upper_bound-lower_bound)*self.total_length)
        index_sample = int(min(self.lengths[dataset_id]* sample_prob + random.randint(0,omit_internal), self.lengths[dataset_id]-1))
        annots = self.datasets[dataset_id][index_sample]
        annots['subject_ids'] += self.ID_num_list[dataset_id]
        return annots

    def __len__(self):
        return self.total_length

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
    config.datasets_used = ['pw3d','crowdpose','posetrack','oh']
    datasets = MixedDataset(train_flag=True)
    from torch.utils.data import DataLoader
    data_loader = DataLoader(dataset = datasets,batch_size = 64,shuffle = True,drop_last = True, pin_memory = True,num_workers =1)
    for data in enumerate(data_loader):
        pass