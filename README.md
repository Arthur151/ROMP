<h2 align="center"> Monocular, One-stage, Regression of Multiple 3D People </h2>

| [ROMP](https://arxiv.org/abs/2008.12272) is a concise one-stage method for monocular multi-person 3D mesh recovery in real time. | [BEV](https://arxiv.org/abs/2112.08274) further explores multi-person depth relationships and supports all age groups.  |
| --- | --- |
<p float="center">
    <img src="../assets/demo/animation/blender_character_driven-min.gif" width="40%" />
    <img  width="2%" />
    <img src="../assets/demo/animation/Solvay_conference_1927_all_people.png" width="26%" />
    <img src="../assets/demo/animation/conference_mesh_rotating.gif" width="30%" />
</p>

We provide **cross-platform API** (installed via pip) to run on Linux / Windows / Mac. 

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
*2022/04/14: Inference code of BEV has been released in simple-romp.*
*2022/04/10: simple-romp v0.0.4 has been released. Adding onnx support, with faster inference speed on CPU/GPU.*  
*2022/03/27: [Relative Human dataset](https://github.com/Arthur151/Relative_Human) has been released.*  
*2022/03/18: Simple version of ROMP for all platform. See the [guidance](https://github.com/Arthur151/ROMP/blob/master/simple_romp/README.md) for details*  
[Old logs](docs/updates.md)

## Getting started

Please use simple-romp for inference, the rest code is just for training.

### Installation
```
pip install --upgrade setuptools numpy cython
pip install --upgrade simple-romp
```
For more details, please refer to [install.md]([docs/basic_installation.md](https://github.com/Arthur151/ROMP/blob/master/simple_romp/README.md)).

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
