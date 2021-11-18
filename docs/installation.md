## Installation

We have tested the code on Ubuntu 18.04 and Centos 7. 

### Download models

##### Release:

Directly download the full-packed released package from Github:
1. [ROMP v1.1](https://github.com/Arthur151/ROMP/releases/tag/v1.1) with all functions.
2. [ROMP v1.0](https://github.com/Arthur151/ROMP/releases/download/v1.0/ROMP_v1.0.zip) with the basic functions to process images/videos/webcam.

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

Please notice that our training code only support CUDA >= 10.2 + Pytorch >= 1.9

1.Please install the Pytorch 1.9 from [the official website](https://pytorch.org/). Alternatively, install using conda:
`conda install -n env_name pytorch==1.9.0 torchvision torchaudio cudatoolkit=10.2 -c pytorch`

2.Installation of the Pytorch3d follow [this website](https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md):

```bash
# For Python3.8 + CUDA 10.2+pytorch 1.9.0, install via
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu102_pyt190/download.html
# For Python3.7 + CUDA 10.2+pytorch 1.9.0, install via
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py37_cu102_pyt190/download.html
```

Install packages:
```bash
cd ROMP/src
pip install -r requirements.txt
```
