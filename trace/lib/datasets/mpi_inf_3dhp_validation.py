from datasets.mpi_inf_3dhp import MPI_INF_3DHP
from datasets.image_base import *
from datasets.base import Base_Classes, Test_Funcs

default_mode = args().video_loading_mode if args().video else args().image_loading_mode

def MPI_INF_3DHP_VALIDATION(base_class=default_mode):
    class MPI_INF_3DHP_VALIDATION(MPI_INF_3DHP(Base_Classes[base_class])):
        def __init__(self,train_flag=False, validation=True, **kwargs):
            super(MPI_INF_3DHP_VALIDATION,self).__init__(train_flag=train_flag, validation=validation)
    return MPI_INF_3DHP_VALIDATION
if __name__ == '__main__':
    datasets=MPI_INF_3DHP_VALIDATION(base_class=default_mode)()
    Test_Funcs[default_mode](datasets,with_smpl=True)
    print('Done')
