<h1 align="center"> 
  <img src="../assets/demo/ROMP_logo.png" width="20%" />
</h1>
<h2 align="center"> Monocular, One-stage, Regression of Multiple 3D People </h2>

[![Google Colab demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1oz9E6uIbj4udOPZvA1Zi9pFx0SWH_UXg)
[![arXiv](https://img.shields.io/badge/arXiv-2008.12272-00ff00.svg)](https://arxiv.org/abs/2008.12272)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/centerhmr-a-bottom-up-single-shot-method-for/3d-human-pose-estimation-on-3d-poses-in-the)](https://paperswithcode.com/sota/3d-human-pose-estimation-on-3d-poses-in-the?p=centerhmr-a-bottom-up-single-shot-method-for)

[ROMP]((https://arxiv.org/abs/2008.12272)) is a concise one-stage network for multi-person 3D mesh recovery from a single image. It can achieve real-time inference speed on a 1070Ti GPU.

[BEV](https://arxiv.org/abs/2112.08274) is built on ROMP to further explore multi-person depth relationships and support all age groups. 

We provide cross-platform API to run on Linux / Windows / Mac. 

<p float="center">
  <img src="../assets/demo/animation/blender_character_driven-min.gif" width="66%" />
</p>

*Please use simple-romp for inference, the rest code is just for training.*

## Table of contents
- [Table of contents](#table-of-contents)
- [News](#news)
- [Getting started](#getting-started)
  - [Installation](#installation)
  - [Try on Google Colab](#try-on-google-colab)
- [How to use it](#how-to-use-it)
  - [Inference](#inference)
  - [Export](#export)
  - [Train](#train)
  - [Evaluation](#evaluation)
  - [Bugs report](#bugs-report)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

## News
*2022/04/10:simple-romp v0.0.4 has been released. Adding onnx support, with faster inference speed on CPU/GPU.*  
*2022/03/27:[Relative Human dataset](https://github.com/Arthur151/Relative_Human) has been released.*  
*2022/03/18: Simple version of ROMP for all platform. See the [guidance](https://github.com/Arthur151/ROMP/blob/master/simple_romp/README.md) for details*  
[Old logs](docs/updates.md)

## Getting started

### Installation
```
pip install --upgrade setuptools numpy cython
pip install simple-romp
```
To run in real time, please refer to [install.md](docs/basic_installation.md) for installation.

### Try on Google Colab

It allows you to run the project in the cloud, free of charge. [Google Colab demo](https://colab.research.google.com/drive/1oz9E6uIbj4udOPZvA1Zi9pFx0SWH_UXg).  

## How to use it

### Inference
Please refer to the [guidance](https://github.com/Arthur151/ROMP/blob/master/simple_romp/README.md).

### Export

Please refer to [expert.md](docs/export.md) to export the results to fbx files for Blender usage. 

### Train
For training, please refer to [installation.md](docs/installation.md) for full installation.
Please prepare the training datasets following [dataset.md](docs/dataset.md), and then refer to [train.md](docs/train.md) for training. 

### Evaluation

Please refer to [evaluation.md](docs/evaluation.md) for evaluation on benchmarks.

### Bugs report

Please refer to [bug.md](docs/bugs.md) for solutions. Welcome to submit the issues for related bugs. I will solve them as soon as possible.

## Citation
```bibtex
@InProceedings{BEV,
author = {Sun, Yu and Liu, Wu and Bao, Qian and Fu, Yili and Mei, Tao and Black, Michael J},
title = {Putting People in their Place: Monocular Regression of 3D People in Depth},
booktitle = {CVPR},
year = {2022}
}

@InProceedings{ROMP,
author = {Sun, Yu and Bao, Qian and Liu, Wu and Fu, Yili and Michael J., Black and Mei, Tao},
title = {Monocular, One-stage, Regression of Multiple 3D People},
booktitle = {ICCV},
year = {2021}
}
```

## Acknowledgement

We thank all [contributors](docs/contributor.md) for their help!

This work was supported by the National Key R&D Program of China under Grand No. 2020AAA0103800. 

**Disclosure**: MJB has received research funds from Adobe, Intel, Nvidia, Facebook, and Amazon and has financial interests in Amazon, Datagen Technologies, and Meshcapade GmbH. While he was part-time at Amazon during this project, his research was performed solely at Max Planck. 