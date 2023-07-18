# TRACE's code for training

This is the code for training TRACE. For inference & evaluation, please refer to [this instruction](../simple_romp/trace2/README.md). 

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
DynaCam: [data](https://github.com/Arthur151/DynaCam)   
MPI-INF-3DHP: [data](https://pan.baidu.com/s/17L0TZB1uC2FOkWfU8BIRmQ?pwd=w3j4)  
Human3.6M: [annotations](https://pan.baidu.com/s/1xGXeXgBUwvINz4I5c0hweQ?pwd=ek92)
3DPW: [images](https://virtualhumans.mpi-inf.mpg.de/3DPW/license.html) [annotations](https://pan.baidu.com/s/11Xv-rFKMIFMwMejPaVtu0A?pwd=q6md)  
PennAction: [images](https://github.com/dreamdragon/PennAction) [annotations](https://pan.baidu.com/s/1YKmEYhv8XM21jPoKs1Y7RQ?pwd=ac9s)

Please follow the directory structure to organize them.
```
|-- datasets
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
dataset_group.add_argument('--dataset_rootdir',type=str, default='/path/to/your/datasets/folder', help= 'root dir of all datasets')
```
If you put different dataset at different path, then you have to set them separately. For instance, to set the path of Human3.6M dataset, please change `self.data_folder` in `ROMP/trace/lib/datasets/h36m.py` to the path where you put Human3.6M, like
```
self.data_folder = /path/to/your/h36m/
```

## Metadata

1. Installing simple-romp:
```
pip install --upgrade setuptools numpy cython lap
pip install simple_romp==1.1.3
```

2. Preparing SMPL model files in our format:

Firstly, please register and download:  
a. Meta data from [this link](https://github.com/Arthur151/ROMP/releases/download/V2.0/smpl_model_data.zip). Please unzip it, then we get a folder named "smpl_model_data"
b. SMPL model file (SMPL_NEUTRAL.pkl) from "Download version 1.1.0 for Python 2.7 (female/male/neutral, 300 shape PCs)" in [official website](https://smpl.is.tue.mpg.de/). Please unzip it and move the SMPL_NEUTRAL.pkl from extracted folder into the "smpl_model_data" folder.      
c. Please also download SMIL model file (DOWNLOAD SMIL) from [official website](https://www.iosb.fraunhofer.de/en/competences/image-exploitation/object-recognition/sensor-networks/motion-analysis.html). Please unzip and put it into the "smpl_model_data" folder, so we have "smpl_model_data/smil/smil_web.pkl".   
Then we can get a folder in structure like this:  
```
|-- smpl_model_data
|   |-- SMPL_NEUTRAL.pkl
|   |-- J_regressor_extra.npy
|   |-- J_regressor_h36m.npy
|   |-- smpl_kid_template.npy
|   |-- smil
|   |-- |-- smil_web.pkl
```

Secondly, please convert the SMPL model files to our format via  
```
# please provide the absolute path of the "smpl_model_data" folder to the source_dir 
romp.prepare_smpl -source_dir=/path/to/smpl_model_data
romp.prepare_smpl -source_dir=/path/to/smpl_model_data --gender=female
romp.prepare_smpl -source_dir=/path/to/smpl_model_data --gender=male

bev.prepare_smil -source_dir=/path/to/smpl_model_data
cp ~/.romp/smil_packed_info.pth ~/.romp/SMIL_NEUTRAL.pth 
```
The converted file would be save to "~/.romp/" in defualt. 

Please don't worry if there are bugs reporting during "Preparing SMPL model files". It is fine as long as we get things like this in "~/.romp/".
```
|-- .romp
|   |-- SMPL_NEUTRAL.pth
|   |-- SMPL_FEMALE.pth
|   |-- SMPL_MALE.pth
|   |-- SMPLA_NEUTRAL.pth
|   |-- SMPLA_FEMALE.pth
|   |-- SMPLA_MALE.pth
|   |-- smil_packed_info.pth
|   |-- SMIL_NEUTRAL.pth
```

## Train
Please edit the settings in `ROMP/trace/configs/trace.yml` and then run
```
cd ROMP/trace
sh train.sh
```