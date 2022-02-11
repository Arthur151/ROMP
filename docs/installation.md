## Installation

We have tested the code on Ubuntu 18.04/20.04 and Centos 7. 

### Download models

##### Release:

Directly download the full-packed released package from Github:
1. [ROMP v1.1](https://github.com/Arthur151/ROMP/releases/tag/v1.1) with all features.
2. [ROMP v1.0](https://github.com/Arthur151/ROMP/releases/download/v1.0/ROMP_v1.0.zip) with some basic features to process images/videos/webcam.

##### Up-to-date:

Clone the repo:
```bash
git clone -b master --single-branch https://github.com/Arthur151/ROMP
```

Then download the [model_data.zip](https://github.com/Arthur151/ROMP/releases/download/v1.1/model_data.zip) and [demo_data.zip](https://github.com/Arthur151/ROMP/releases/download/v1.1/demo_videos.zip) from Github release, [Google drive](https://drive.google.com/drive/folders/1YdsHh62KGuQMowRjKM9Vzj_7pflb51BB?usp=sharing). 

If you want a fast try of ROMP, please download [trained_models_try.zip](https://github.com/Arthur151/ROMP/releases/download/v1.1/trained_models_try.zip).

If you want to re-implement all results presented in our paper, please download all [trained_models.zip](https://github.com/Arthur151/ROMP/releases/download/v1.1/trained_models.zip).

The layout would be
```bash
ROMP
  - configs
  - model_data
  - romp
  - trained_models
  - scripts
  - docs
  - demo
  - active_configs
```

#### Set up environments

[Pytorch 1.10.0](https://pytorch.org/)  
[Pytorch3d 0.6.1](https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md)  

**Windows only:** To build some package wheels, 'Visual Studio Build Tools' and 'Visual C++ build tools workload' are required.
You may install them with the [Chocolatey](https://chocolatey.org) package manager: `choco install visualstudio2019buildtools visualstudio2019-workload-vctools`.

Firstly, please decide whether you want to install via conda env with python 3.7 or python 3.8 or pip.  
We recommend installing via conda so that ROMP env is clean and will not affect other repo.  

Option 1) to install conda env with python 3.7, please run

```
conda create -n ROMP python==3.7.6  
conda activate ROMP  
conda install -n ROMP pytorch==1.10.0 torchvision==0.11.1 cudatoolkit=10.2 -c pytorch  
pip install https://github.com/Arthur151/ROMP/releases/download/v1.1/pytorch3d-0.6.1-cp37-cp37m-linux_x86_64.whl
cd ROMP  
pip install -r requirements.txt  
```

Option 2) to install conda env with python 3.8, please run
```
conda create -n ROMP python==3.8.8  
conda activate ROMP  
conda install -n ROMP pytorch==1.10.0 torchvision==0.11.1 cudatoolkit=10.2 -c pytorch  
pip install https://github.com/Arthur151/ROMP/releases/download/v1.1/pytorch3d-0.6.1-cp38-cp38-linux_x86_64.whl
cd ROMP  
pip install -r requirements.txt  
```

Option 3) To directly install via pip, you need to install CUDA 10.2 first (For Ubuntu, run`sudo apt-get install cuda-10-2`) and then install via:
```
pip install pytorch==1.10.0 torchvision==0.11.1
# if you use Python3.8, please install pytorch3d via
pip install https://github.com/Arthur151/ROMP/releases/download/v1.1/pytorch3d-0.6.1-cp38-cp38-linux_x86_64.whl
# if you use Python3.7, please install pytorch3d via
pip install https://github.com/Arthur151/ROMP/releases/download/v1.1/pytorch3d-0.6.1-cp37-cp37m-linux_x86_64.whl
cd ROMP  
pip install -r requirements.txt  
```