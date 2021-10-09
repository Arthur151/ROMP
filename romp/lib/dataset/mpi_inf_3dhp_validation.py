import sys, os

from dataset.mpi_inf_3dhp import MPI_INF_3DHP
from dataset.image_base import *


class MPI_INF_3DHP_VALIDATION(MPI_INF_3DHP):
    def __init__(self,train_flag=False, validation=True, **kwargs):
        super(MPI_INF_3DHP_VALIDATION,self).__init__(train_flag=train_flag, validation=validation)

if __name__ == '__main__':
    dataset=MPI_INF_3DHP_VALIDATION()
    test_dataset(dataset,with_smpl=True)
    print('Done')
