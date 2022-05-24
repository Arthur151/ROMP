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
  - [Bugs report](#bugs-report)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

## News
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

### Try on Google Colab

It allows you to run the project in the cloud, free of charge. [Google Colab demo](https://colab.research.google.com/drive/1oz9E6uIbj4udOPZvA1Zi9pFx0SWH_UXg).  

## How to use it

#### Please refer to [this guidance](https://github.com/Arthur151/ROMP/blob/master/simple_romp/README.md) for inference & export (fbx/glb/bvh).

### Train
For training, please refer to [installation.md](docs/installation.md) for full installation.
Please prepare the training datasets following [dataset.md](docs/dataset.md), and then refer to [train.md](docs/train.md) for training. 

### Evaluation

Please refer to [evaluation.md](docs/evaluation.md) for evaluation on benchmarks.

### Docker usage
  ```
  # Build
  docker build --rm -t romp .
  # Inference
  docker run --privileged --rm -it --gpus 0 --ipc=host -p 8888:8888 -v /tmp/.X11-unix:/tmp/.X11-unix -v $(pwd):/workspace/results --device /dev/video0 -e DISPLAY=$DISPLAY romp --mode=webcam
  ```

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
