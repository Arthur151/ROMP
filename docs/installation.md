## Installation

We have tested the code on Ubuntu 18.04/20.04, Centos 7 and Windows. 

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

 * [Pytorch](https://pytorch.org/)  
 * [Pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md) (optional)

**Windows warning:** To install on Windows, please insall in cmd instead of Powershell, in convenience of activating the conda env.
To build some package wheels, 'Visual Studio Build Tools' and 'Visual C++ build tools workload' are required.
To install them with the Chocolatey package manager, please [install the Chocolatey](https://docs.chocolatey.org/en-us/choco/setup#more-install-options) first and then run `choco install visualstudio2019buildtools visualstudio2019-workload-vctools`.

1. Please decide whether you want to install the Pytorch via [pip](https://pip.pypa.io/en/stable) or [conda](https://docs.conda.io/en/latest/miniconda.html) env and Python 3.9, 3.8 or 3.7.  We recommend installing via conda so that ROMP env is clean and will not affect other repo.  

Option 1) to install conda env with python 3.9, please run
```
conda create -n ROMP python=3.9
conda activate ROMP  
conda install -n ROMP pytorch==1.10.0 torchvision==0.11.1 cudatoolkit=10.2 -c pytorch  
```
Option 2) to install conda env with python 3.8, please run
```
conda create -n ROMP python==3.8.8  
conda activate ROMP  
conda install -n ROMP pytorch==1.10.0 torchvision==0.11.1 cudatoolkit=10.2 -c pytorch  
```

Option 3) to install conda env with python 3.7, please run
```
conda create -n ROMP python==3.7.6  
conda activate ROMP  
conda install -n ROMP pytorch==1.10.0 torchvision==0.11.1 cudatoolkit=10.2 -c pytorch  
```

Option 4) To directly install via pip, you need to install CUDA 10.2 first (For Ubuntu, run`sudo apt-get install cuda-10-2`) and then install via:
```
pip install torch==1.10.0+cu102 torchvision==0.11.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html
```

(Optional) Install Pytorch3D for rendering, otherwise please refer to [this instruction](https://github.com/Arthur151/ROMP/blob/master/docs/config_guide.md#renderer-str) to set `renderer: pyrender`.  'pyrender' can be only used on desktop. To train ROMP or run it on server without visualization hardware, please install 'pytorch3d' and set renderer to 'pytorch3d'.
On Linux, please install via
```
# if you use Python3.9 (Option 1 or Option 4 with python3.9), please install pytorch3d via
pip install https://github.com/Arthur151/ROMP/releases/download/v1.1/pytorch3d-0.6.1-cp39-cp39-linux_x86_64.whl
# if you use Python3.8 (Option 2 or Option 4 with python3.8), please install pytorch3d via
pip install https://github.com/Arthur151/ROMP/releases/download/v1.1/pytorch3d-0.6.1-cp38-cp38-linux_x86_64.whl
# if you use Python3.7 (Option 3 or Option 4 with python3.7), please install pytorch3d via
pip install https://github.com/Arthur151/ROMP/releases/download/v1.1/pytorch3d-0.6.1-cp37-cp37m-linux_x86_64.whl
```
On Mac or Windows, please follow [the official instruction](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) to install.

2. Please install the python libs via
```
cd ROMP  
pip install -r requirements.txt  
```
