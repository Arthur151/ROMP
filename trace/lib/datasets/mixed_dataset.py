from .h36m import H36M
from .pw3d import PW3D
from .mpi_inf_3dhp import MPI_INF_3DHP
from .mpi_inf_3dhp_test import MPI_INF_3DHP_TEST
from .mpi_inf_3dhp_validation import MPI_INF_3DHP_VALIDATION
from .internet_video import InternetVideo
from .penn_action import PennAction
from .DynaCamTranslation import DynaCamTranslation
from .DynaCamRotation import DynaCamRotation
import os
from datasets.image_base import *
from config import args
from prettytable import PrettyTable

dataset_dict = {'h36m': H36M, 'pw3d':PW3D, 'mpiinf':MPI_INF_3DHP,'mpiinf_val':MPI_INF_3DHP_VALIDATION,'mpiinf_test':MPI_INF_3DHP_TEST, \
                'internet_video':InternetVideo, 'pennaction':PennAction, 'DC': DynaCamTranslation, 'PDC': DynaCamRotation}

class MixedDataset(Dataset):
    def __init__(self, datasets_used, sample_prob_dict, loading_modes=None, max_length=args().batch_size*10000, **kwargs):
        if loading_modes is None:
            self.datasets = [dataset_dict[ds]()(**kwargs) for ds in datasets_used]
        else:
            self.datasets = [dataset_dict[ds](mode)(**kwargs) for ds, mode in zip(datasets_used, loading_modes)]
        self.lengths, self.partition, self.ID_num_list, self.ID_num = [], [], [], 0
        if not 1.001>sum(sample_prob_dict.values())>0.999:
            print('CAUTION: The sum of sampling rates is supposed to be 1, while currently we have {}, \n please properly set the sample_prob_dict {} in config.yml'\
                .format(sum(sample_prob_dict.values()), sample_prob_dict.values()))
        for ds_idx, ds_name in enumerate(datasets_used):
            self.lengths.append(len(self.datasets[ds_idx]))
            self.partition.append(sample_prob_dict[ds_name])
            self.ID_num_list.append(self.ID_num)
            self.ID_num += self.datasets[ds_idx].ID_num
        dataset_info_table = PrettyTable([' ']+datasets_used)
        dataset_info_table.add_row(['Length']+self.lengths)
        dataset_info_table.add_row(['Sample Prob.']+self.partition)
        expect_length = (np.array(self.lengths)/np.array(self.partition)).astype(np.int32)
        dataset_info_table.add_row(['Expected length']+expect_length.tolist())
        self.partition = np.array(self.partition).cumsum()
        dataset_info_table.add_row(['Accum. Prob.']+self.partition.astype(np.float16).tolist())
        dataset_info_table.add_row(['Accum. ID.']+self.ID_num_list)
        print(dataset_info_table)
        self.total_length = min(int(expect_length.max()), max_length)
        logging.info('All datasets length: {}'.format(len(self)))

    def _get_ID_num_(self):
        return self.ID_num

    def __getitem__(self, index):
        p = float(index)/float(self.total_length)
        dataset_id = len(self.partition)-(self.partition>p).sum()

        upper_bound = self.partition[dataset_id]
        lower_bound = self.partition[dataset_id-1] if dataset_id>0 else 0
        sample_prob = (p-lower_bound)/(upper_bound-lower_bound)

        assert sample_prob<1, 'sampling rate within a dataset is surpposed to be less than 1.'

        if (upper_bound-lower_bound)*self.total_length<self.lengths[dataset_id]:
            # due to the total length might be smaller than self.lengths[dataset_id]
            omit_internal = self.lengths[dataset_id]//((upper_bound-lower_bound)*self.total_length)
            index_sample = int(min(self.lengths[dataset_id]* sample_prob + random.randint(0,omit_internal), self.lengths[dataset_id]-1))
        else:
            index_sample = int(sample_prob * self.lengths[dataset_id])
        data = self.datasets[dataset_id][index_sample]
        data['subject_ids'][data['subject_ids']!=-1] += self.ID_num_list[dataset_id]
        return data
    
    def resampling_video_clips(self):
        for inds in range(len(self.datasets)):
            self.datasets[inds].prepare_video_clips()

    def __len__(self):
        return self.total_length

class SingleDataset(Dataset):
    def __init__(self, dataset=None, loading_mode=None,**kwargs):
        assert dataset in dataset_dict, print('dataset {} not found while creating data loader!'.format(dataset))
        if loading_mode is None:
            self.dataset = dataset_dict[dataset]()(**kwargs)
        else:
            self.dataset = dataset_dict[dataset](loading_mode)(**kwargs)
        self.length = len(self.dataset)            

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return self.length

class SingleVideoDataset(Dataset):
    def __init__(self, datasets=None,**kwargs):
        assert datasets in dataset_dict, print('datasets {} not found while creating data loader!'.format(datasets))
        self.datasets_used = [datasets]
        self.datasets = dataset_dict[datasets](**kwargs)
        self.sequence_ids = [self.datasets.sequence_ids]
        
        self.length = len(self.datasets) 
        self.cum_length = [self.length]           

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return self.length

def test_imgpath(ind, data_imgpaths):
    imgpaths = [os.path.basename(path) for path in data_imgpaths]
    if imgpaths[-1] == imgpaths[-2]:
        print(ind, len(data_imgpaths), imgpaths)
    elif imgpaths[0] == imgpaths[1]:
        print(ind, len(data_imgpaths), imgpaths)
    else:
        print(ind, imgpaths)