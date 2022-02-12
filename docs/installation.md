## Installation

We have tested the code on Ubuntu 18.04/20.04, Centos 7 and Windows 10. 

### First Step: fetching code & data

#### 1. Fetching code

Please decide whether to fetch the release (Option 1) or the up-to-date (Option 2).

##### (Option 1) Release:

Directly download the full-packed released package from Github:  
(1) [ROMP v1.1](https://github.com/Arthur151/ROMP/releases/tag/v1.1) with all features.  
(2) [ROMP v1.0](https://github.com/Arthur151/ROMP/releases/download/v1.0/ROMP_v1.0.zip) with some basic features to process images/videos/webcam.  

##### (Option 2) Up-to-date:

```bash
git clone -b master --single-branch https://github.com/Arthur151/ROMP
```

#### 2. Fetching data

Please download the essential data (model_data.zip, demo_data.zip) and pre-trained model (trained_models) from :   
(Option 1) Github release: [model_data.zip](https://github.com/Arthur151/ROMP/releases/download/v1.1/model_data.zip), [demo_data.zip](https://github.com/Arthur151/ROMP/releases/download/v1.1/demo_videos.zip), [trained_models_try.zip](https://github.com/Arthur151/ROMP/releases/download/v1.1/trained_models_try.zip) or [trained_models.zip](https://github.com/Arthur151/ROMP/releases/download/v1.1/trained_models.zip)  
(Option 2) [Google drive](https://drive.google.com/drive/folders/1YdsHh62KGuQMowRjKM9Vzj_7pflb51BB?usp=sharing).   

Please note that trained_models_try.zip is enough for processing image/video/webcam, trained_models.zip is just to re-implemente all results in our paper.

After the first step, the layout of ROMP folder should be
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

### Second step: seting up environments

#### 1. Install [Pytorch](https://pytorch.org/).
Please choose one of the following 4 options to install Pytorch via [conda](https://docs.conda.io/en/latest/miniconda.html) or [pip](https://pip.pypa.io/en/stable). 
Here, we support to install with Python 3.9, 3.8 or 3.7. 
We recommend installing via conda (Option 1-3) so that ROMP env is clean and will not affect other repo.  

##### Option 1) to install conda env with python 3.9, please run
```
conda create -n ROMP python=3.9
conda activate ROMP  
conda install -n ROMP pytorch==1.10.0 torchvision==0.11.1 cudatoolkit=10.2 -c pytorch  
```
##### Option 2) to install conda env with python 3.8, please run
```
conda create -n ROMP python==3.8.8  
conda activate ROMP  
conda install -n ROMP pytorch==1.10.0 torchvision==0.11.1 cudatoolkit=10.2 -c pytorch  
```

##### Option 3) to install conda env with python 3.7, please run
```
conda create -n ROMP python==3.7.6  
conda activate ROMP  
conda install -n ROMP pytorch==1.10.0 torchvision==0.11.1 cudatoolkit=10.2 -c pytorch  
```

##### Option 4) install via pip
To directly install via pip, you need to install CUDA 10.2 first (For Ubuntu, run`sudo apt-get install cuda-10-2`).  
Then install pytorch via:  
```
pip install torch==1.10.0+cu102 torchvision==0.11.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html
```

#### 2. (Optional) Install [Pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md) for rendering, otherwise please refer to [this instruction](https://github.com/Arthur151/ROMP/blob/master/docs/config_guide.md#renderer-str) to use pyrender via seting `renderer: pyrender`.
Please note that 'pyrender' can be only used on desktop. To train ROMP or run it on server without visualization hardware, please install 'pytorch3d' and set `renderer: pytorch3d` in configs.  
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

#### 3. Install the python dependencies.
```
cd ROMP  
pip install -r requirements.txt  
```
**To deal with bugs when installing on Windows.** To build some package wheels, 'Visual Studio Build Tools' and 'Visual C++ build tools workload' are required.
To install them with the Chocolatey on Windows, please install in cmd as administrator instead of Powershell, meanwhile, please ensure your network can access to google.
Please [install the Chocolatey](https://docs.chocolatey.org/en-us/choco/setup#more-install-options) first and then run `choco install visualstudio2019buildtools visualstudio2019-workload-vctools`.

