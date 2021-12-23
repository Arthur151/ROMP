## Train

1. Please first prepare your datasets follow [dataset.md](dataset.md) and finish the installation follow [installation.md](installation.md).

2. Run the script:
You can train ROMP via
```
# go into the path/to/ROMP
cd ROMP
# To train the ROMP with HRNet-32 as backbone, please run
sh scripts/V1_train.sh
# To train the ROMP with ResNet-50 as backbone, please run
sh scripts/V1_train_resnet.sh
```
To re-implement the results in Tab.3 of our paper, you can fine-tune the trained model on the 3DPW dataset via
```
# To fine-tune the ROMP with HRNet-32 as backbone, please run
sh scripts/V1_hrnet_3dpwft.sh
# To fine-tune the ROMP with ResNet-50 as backbone, please run
sh scripts/V1_resnet_3dpwft.sh
```
You can change the configurations (e.g. used GPUs, batch size) in the corresponding configs (.yml) in ROMP/configs.

The training logs wiil be saved in ROMP/log. 

The intermediate training/validation results will be visualized and saved in result_image_train/result_images.

## Training guidance

To properly train ROMP, here we provide a guidance.

1. Training datasets:

Step 1) To set the training datasets, please edit the configuration files (.yml in configs).  
All registered datasets are listed at [romp/lib/dataset/mixed_dataset.py](https://github.com/Arthur151/ROMP/blob/b8d3ac8889c5cd3736182864b43779ee70be8741/romp/lib/dataset/mixed_dataset.py#L29), such as 
```
dataset_dict = {'h36m': H36M, 'mpii': MPII, 'coco': COCO14, 'posetrack':Posetrack, 'aich':AICH, 'pw3d':PW3D, 'up':UP, 'crowdpose':Crowdpose, 'crowdhuman':CrowdHuman,\
 'lsp':LSP, 'mpiinf':MPI_INF_3DHP,'mpiinf_val':MPI_INF_3DHP_VALIDATION,'mpiinf_test':MPI_INF_3DHP_TEST, 'muco':MuCo, 'mupots':MuPoTS, \
 'cmup':CMU_Panoptic_eval,'internet':Internet, }
```
You change the `dataset` in configuration files by listing the key of `dataset_dict`.  
For instance, if we want to use Human3.6M, MPI-INF-3DHP, COCO, MPII, LSP, MuCo, Crowdpose, please set `dataset` as 
```
dataset: 'h36m,mpiinf,coco,mpii,lsp,muco,crowdpose'
```
Please note that different datasets are splited via `,` and no space shall be put.

Step 2) Setting the sampling rate of different datasets.  
Please check the `sample_prob` in configuration files to ensure that the sum of the sampling rates is 1.
For instance, in `configs/v1.yml`, we set the sampling rates as 
```
sample_prob:
 h36m: 0.2
 mpiinf: 0.16
 coco: 0.2
 lsp: 0.06
 mpii: 0.1
 muco: 0.14
 crowdpose: 0.14
```
If we want to remove LSP, then we have to adjust the sampling rates, like 
```
sample_prob:
 h36m: 0.2
 mpiinf: 0.16
 coco: 0.2
 mpii: 0.1
 muco: 0.17
 crowdpose: 0.17
```
We recommend setting the sampling rate of a data set to less than 0.2 for better generalization.

2. Training with your own datasets:

It is convenient to fine-tune ROMP with you own datasets. You just have to load the images / annoations follow the template and then register the dataset.

Step 1) Please create dataloader to load the images / annoations follow the template.  
We recommend to use [MPI-INF-3DHP](https://github.com/Arthur151/ROMP/blob/master/romp/lib/dataset/mpi_inf_3dhp_test.py) as template for 3D pose dataset.  
We recommend to use [Crowdpose](https://github.com/Arthur151/ROMP/blob/master/romp/lib/dataset/crowdpose.py) as template for 2D pose dataset.  

Firstly, please copy the template file to create a new dataloader for your own dataset, like romp/lib/datasets/mydataset.py.  
Replace the class name (like MPI_INF_3DHP_TEST in mpi_inf_3dhp_test.py) to your own dataset name (like MyDataset).

Secondly, please re-write the `pack_data` function in the class to pack the annotations into a .npz file in convenience of data loading.  
The format is a dictionary with image name as key. The annotations for each image are sub-dictionaries / list as value.  
Meanwile, please set the `self.data_folder` to the path of your dataset.

Thirdly, please re-write the function `get_image_info` to load the image / annotations.  
Please properly set the information in `img_info` dict. For the missed annotations, please just set `None`. 
Especailly, please properly set the `vmask_2d` and `vmask_3d`.  
Each bool value in `vmask_3d` means whether we have corresponding annotation.   
Each bool value in `vmask_2d` is explained below.
```
# vmask_2d | 0: the 'kp2ds' is kp2d (True) or bbox (False) | 1: whether we have track ids | 2: whether the annotations label all people in image, True for yes.
# vmask_3d | 0: kp3d | 1: smpl global orient | 2: smpl body pose | 3: smpl body shape
```

Finally, please test the correctness via running
```
cd ROMP
python -m romp.lib.dataset.mydataset
```
Annotations will be drawed on the input image. The test results will be saved in ROMP/test/.

Step 2) Please register the new dataset.

To register the new dataset, please import the new dataloader in [romp/lib/dataset/mixed_dataset.py](https://github.com/Arthur151/ROMP/blob/b8d3ac8889c5cd3736182864b43779ee70be8741/romp/lib/dataset/mixed_dataset.py) and add a new item in [dataset_dict](https://github.com/Arthur151/ROMP/blob/b8d3ac8889c5cd3736182864b43779ee70be8741/romp/lib/dataset/mixed_dataset.py#L29).  
For instance, if your dataloader is class MyDataset in datasets/mydataset.py, in `mixed_dataset.py` please add
```
from .mydataset import MyDataset
dataset_dict = {'mydataset': MyDataset, 'h36m': H36M, 'mpii': MPII ... }
``` 