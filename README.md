<h2 align="center"> Monocular, One-stage, Regression of Multiple 3D People </h2>

| <div align=center><img src="../assets/demo/ROMP_logo.png" width="30%" /></div> | <div align=center><img src="../assets/demo/BEV_logo.png" width="30%" /></div> |
| :---: | :---: |
| ROMP is a **one-stage** method for monocular multi-person 3D mesh recovery in **real time**. | BEV further explores multi-person **depth relationships** and supports **all age groups**.  |
| **[[Paper]](https://arxiv.org/abs/2008.12272) [[Video]](https://www.youtube.com/watch?v=hunBPJxnyBU)** | **[[Project Page]](https://arthur151.github.io/BEV/BEV.html) [[Paper]](https://arxiv.org/abs/2112.08274) [[Video]](https://youtu.be/Q62fj_6AxRI) [[RH Dataset]](https://github.com/Arthur151/Relative_Human)** |
<img src="../assets/demo/animation/blender_character_driven-min.gif" alt="drawing" width="500"/> | <img src="../assets/demo/images_results/BEV_tennis_results.png" alt="drawing" width="340"/>

We provide **cross-platform API** (installed via pip) to run ROMP & BEV on Linux / Windows / Mac. 


## Table of contents
- [Table of contents](#table-of-contents)
- [News](#news)
- [Getting started](#getting-started)
  - [Installation](#installation)
  - [Try on Google Colab](#try-on-google-colab)
- [How to use it](#how-to-use-it)
    - [Please refer to this guidance for inference & export (fbx/glb/bvh).](#please-refer-to-this-guidance-for-inference--export-fbxglbbvh)
  - [Train](#train)
  - [Evaluation](#evaluation)
  - [Docker usage](#docker-usage)
  - [Bugs report](#bugs-report)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

## News
*2022/06/21: Training & evaluation code of BEV is released. Please update the [model_data](https://github.com/Arthur151/ROMP/releases/download/v1.1/model_data.zip).*   
*2022/05/16: simple-romp v1.0 is released to support tracking, calling in python, exporting bvh, and etc.*   
*2022/04/14: Inference code of BEV has been released in simple-romp v0.1.0.*   
*2022/04/10: Adding onnx support, with faster inference speed on CPU/GPU.*   
[Old logs](docs/updates.md)

## Getting started

Please use simple-romp for inference, the rest code is just for training.

### Installation
```
pip install --upgrade setuptools numpy cython
pip install --upgrade simple-romp
```
For more details, please refer to [install.md](https://github.com/Arthur151/ROMP/blob/master/simple_romp/README.md).

## How to use it

#### Please refer to [this guidance](https://github.com/Arthur151/ROMP/blob/master/simple_romp/README.md) for inference & export (fbx/glb/bvh).

### Train
For training, please refer to [installation.md](docs/installation.md) for full installation.
Please prepare the training datasets following [dataset.md](docs/dataset.md), and then refer to [train.md](docs/train.md) for training. 

### Evaluation

Please refer to [romp_evaluation.md](docs/romp_evaluation.md) and [bev_evaluation.md](docs/bev_evaluation.md) for evaluation on benchmarks.

### Extensions

[[Blender addon]](https://github.com/yanchxx/CDBA): [Yan Chuanhang](https://github.com/yanchxx) created a Blender-addon to drive a 3D character in Blender using ROMP from image, video or webcam input.

[[VMC protocol]](https://codeberg.org/vivi90/vmcps): [Vivien Richter](https://github.com/vivi90) implemented a VMC (Virtual Motion Capture) protocol support for different Motion Capture solutions with ROMP. 

### Docker usage

Please refer to [docker.md](docs/docker.md)

### Bugs report

Welcome to submit issues for the bugs.

## Contributors

This repository is currently maintained by [Yu Sun](https://github.com/Arthur151).  

We thank [Peng Cheng](https://github.com/CPFLAME) for his constructive comments on Center map training.  

ROMP has also benefited from many developers, including   
 - [Marco Musy](https://github.com/marcomusy) : help in [the textured SMPL visualization](https://github.com/marcomusy/vedo/issues/371).  
 - [Gavin Gray](https://github.com/gngdb) : adding support for an elegant context manager to run code in a notebook.  
 - [VLT Media](https://github.com/vltmedia) and [Vivien Richter](https://github.com/vivi90) : adding support for running on Windows & batch_videos.py.  
 - [Chuanhang Yan](https://github.com/yanch2116) : developing an [addon for driving character in Blender](https://github.com/yanch2116/Blender-addons-for-SMPL).  
 - [Tian Jin](https://github.com/jinfagang): help in simplified smpl and fast rendering ([realrender](https://pypi.org/project/realrender/)).
 - [ZhengdiYu](https://github.com/ZhengdiYu) : helpful discussion on optimizing the implementation details.
 - [Ali Yaghoubian](https://github.com/AliYqb) : add Docker file for simple-romp.

## Citation
```bibtex
@InProceedings{BEV,
author = {Sun, Yu and Liu, Wu and Bao, Qian and Fu, Yili and Mei, Tao and Black, Michael J},
title = {Putting People in their Place: Monocular Regression of 3D People in Depth},
booktitle = {CVPR},
year = {2022}}
@InProceedings{ROMP,
author = {Sun, Yu and Bao, Qian and Liu, Wu and Fu, Yili and Michael J., Black and Mei, Tao},
title = {Monocular, One-stage, Regression of Multiple 3D People},
booktitle = {ICCV},
year = {2021}}
```

## Acknowledgement

We thank all [contributors](docs/contributor.md) for their help!  
This work was supported by the National Key R&D Program of China under Grand No. 2020AAA0103800.  
**Disclosure**: MJB has received research funds from Adobe, Intel, Nvidia, Facebook, and Amazon and has financial interests in Amazon, Datagen Technologies, and Meshcapade GmbH. While he was part-time at Amazon during this project, his research was performed solely at Max Planck. 
