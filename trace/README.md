# TRACE's code for training

This is the code for training TRACE. Please refer to [this instruction.](../simple_romp/trace2/README.md)

## Installation

1.Preparing data. 

Please download related `project_data` from [here](https://pan.baidu.com/s/1MLAobb39GdmnH5fjbQppxQ?pwd=8jr5), unzip it and put it into `ROMP/`, such that `ROMP/project_data/trace_data`.

Prepare SMPL model files refer to [this instrcution](../simple_romp/trace2/README.md), and move all generated SMPL files from `~/.romp` to `ROMP/project_data/trace_data/parameters/`. 
We are supposed to have 
```
|-- parameters
|   |-- smpl
|   |   |-- SMPL_FEMALE.pkl
|   |   |-- smpl_kid_template.npy
|   |   |-- SMPL_MALE.pkl
|   |   |-- SMPL_NEUTRAL.pkl
|   |-- smil
|   |   |-- smil_web.pkl
|   |-- SMPLA_NEUTRAL.pth
|   |-- SMPL_NEUTRAL.pth
|   |-- smil_packed_info.pth
```

2.Please install some basic python libs, including 
```
Pytorch with CUDA support
torchvision
Pytorch3D (please install it from source to match the installed pytorch version)
opencv-python
PIL Image
```

3.Except for the basic python libs, please install the dependences via
```
cd trace
sh install.sh
```

## Datasets

Please prepare datasets for training using following links:  
[DynaCam](https://github.com/Arthur151/DynaCam) 
[MPI-INF-3DHP](https://pan.baidu.com/s/17L0TZB1uC2FOkWfU8BIRmQ?pwd=w3j4)  
Human3.6M: [annotations](https://pan.baidu.com/s/1xGXeXgBUwvINz4I5c0hweQ?pwd=ek92)
3DPW: [images](https://virtualhumans.mpi-inf.mpg.de/3DPW/license.html) [annotations](https://pan.baidu.com/s/11Xv-rFKMIFMwMejPaVtu0A?pwd=q6md)  
PennAction: [images](https://github.com/dreamdragon/PennAction) [annotations](https://pan.baidu.com/s/1YKmEYhv8XM21jPoKs1Y7RQ?pwd=ac9s)

Please follow the directory structure to organize them.
```
|-- dataset
|   |-- h36m
|   |   |-- images
|   |   |-- annots_smplkps.npz
|   |-- mpi-inf-3dhp
|   |   |-- video_frames
|   |   |-- annots_video.npz
|   |-- DynaCam
|   |   |-- video_frames
|   |   |-- annotations
|   |-- 3DPW
|   |   |-- imageFiles
|   |   |-- annots.npz
|   |   |-- camera_annots.npz
|   |-- Penn_Action
|   |   |-- frames
|   |   |-- annots.npz
```
Finally, pleaset set the dataset root path:  
If you put all datasets in one folder, then you just need to change dataset_rootdir in `ROMP/trace/lib/config.py` to the path of your dataset folder, like:
```
dataset_group.add_argument('--dataset_rootdir',type=str, default='/path/to/your/dataset/folder', help= 'root dir of all datasets')
```
If you put different dataset at different path, then you have to set them separately. For instance, to set the path of Human3.6M dataset, please change `self.data_folder` in `ROMP/trace/lib/datasets/h36m.py` to the path where you put Human3.6M, like
```
self.data_folder = /path/to/your/h36m/
```

## Train
Please edit the settings in `ROMP/trace/configs/trace.yml` and then run
```
cd ROMP/trace
sh train.sh
```