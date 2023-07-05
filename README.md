| <h2 align="center"> ROMP </h2> | <h2 align="center"> BEV </h2> | <h2 align="center"> TRACE </h2> |
| :---: | :---: | :---: |
| Monocular, One-stage, Regression of Multiple 3D People (ICCV21) | Putting People in their Place: Monocular Regression of 3D People in Depth (CVPR2022) | TRACE: 5D Temporal Regression of Avatars with Dynamic Cameras in 3D Environments (CVPR2023) |
| ROMP is a **one-stage** method for monocular multi-person 3D mesh recovery in **real time**. | BEV further explores multi-person **depth relationships** and supports **all age groups**. | TRACE further **tracks specific subjects** and recover their **global 3D trajectory with dynamic cameras**. |
| **[[Paper]](https://arxiv.org/abs/2008.12272) [[Video]](https://www.youtube.com/watch?v=hunBPJxnyBU)** | **[[Project Page]](https://arthur151.github.io/BEV/BEV.html) [[Paper]](https://arxiv.org/abs/2112.08274) [[Video]](https://youtu.be/Q62fj_6AxRI)** |  **[[Project Page]](https://arthur151.github.io/TRACE/TRACE.html) [[Paper]](http://arxiv.org/abs/2306.02850) [[Video]](https://www.youtube.com/watch?v=l8aLHDXWQRw)** |
| | **[[RelativeHuman Dataset]](https://github.com/Arthur151/Relative_Human)** | **[[DynaCam Dataset]](https://github.com/Arthur151/DynaCam)** |
| <img src="../assets/demo/animation/blender_character_driven-min.gif" alt="drawing" height="230"/> | <img src="../assets/demo/images_results/BEV_tennis_results.png" alt="drawing" height="230"/> | <img src="https://www.yusun.work/TRACE/images/demo.gif" alt="drawing" height="230"/> |

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
*2023/06/17: Release of TRACE's code. Please refer to this [instructions](simple_romp/trace2/README.md) for inference.*   
*2022/06/21: Training & evaluation code of BEV is released. Please update the [model_data](https://github.com/Arthur151/ROMP/releases/download/v1.1/model_data.zip).*   
*2022/05/16: simple-romp v1.0 is released to support tracking, calling in python, exporting bvh, and etc.*   
*2022/04/14: Inference code of BEV has been released in simple-romp v0.1.0.*   
*2022/04/10: Adding onnx support, with faster inference speed on CPU/GPU.*   
[Old logs](docs/updates.md)

## Getting started

Please use simple-romp for inference, the rest code is just for training.

## How to use it

## ROMP & BEV
#### For inference & export (fbx/glb/bvh), please refer to [this guidance](https://github.com/Arthur151/ROMP/blob/master/simple_romp/README.md).
#### For training, please refer to [installation.md](docs/installation.md) for full installation, [dataset.md](docs/dataset.md) for data preparation, [train.md](docs/train.md) for training.
#### For evaluation on benchmarks, please refer to [romp_evaluation](docs/romp_evaluation.md), [bev_evaluation](docs/bev_evaluation.md).

## TRACE
#### For inference, please refer to [this instrcution](simple_romp/trace2/README.md).
#### For evaluation on benchmarks, please refer to [trace_evaluation](simple_romp/trace2/README.md).
#### For training, please refer to [trace_train](trace/README.md).

### Extensions

[[Blender addon]](https://github.com/yanchxx/CDBA): [Yan Chuanhang](https://github.com/yanchxx) created a Blender-addon to drive a 3D character in Blender using ROMP from image, video or webcam input.

[[VMC protocol]](https://codeberg.org/vivi90/vmcps): [Vivien Richter](https://github.com/vivi90) implemented a VMC (Virtual Motion Capture) protocol support for different Motion Capture solutions with ROMP. 

### Docker usage

Please refer to [docker.md](docs/docker.md)

### Bugs report

Welcome to submit issues for the bugs.

## Contributors

This repository is maintained by [Yu Sun](https://www.yusun.work/).  

ROMP has also benefited from many developers, including   
 - [Peng Cheng](https://github.com/CPFLAME) : constructive discussion on Center map training.  
 - [Marco Musy](https://github.com/marcomusy) : help in [the textured SMPL visualization](https://github.com/marcomusy/vedo/issues/371).  
 - [Gavin Gray](https://github.com/gngdb) : adding support for an elegant context manager to run code in a notebook.  
 - [VLT Media](https://github.com/vltmedia) and [Vivien Richter](https://github.com/vivi90) : adding support for running on Windows & batch_videos.py.  
 - [Chuanhang Yan](https://github.com/yanch2116) : developing an [addon for driving character in Blender](https://github.com/yanch2116/Blender-addons-for-SMPL).  
 - [Tian Jin](https://github.com/jinfagang): help in simplified smpl and fast rendering ([realrender](https://pypi.org/project/realrender/)).
 - [ZhengdiYu](https://github.com/ZhengdiYu) : helpful discussion on optimizing the implementation details.
 - [Ali Yaghoubian](https://github.com/AliYqb) : add Docker file for simple-romp.

## Citation
```bibtex
@InProceedings{TRACE,
    author = {Sun, Yu and Bao, Qian and Liu, Wu and Mei, Tao and Black, Michael J.},
    title = {{TRACE: 5D Temporal Regression of Avatars with Dynamic Cameras in 3D Environments}}, 
    booktitle = {CVPR}, 
    year = {2023}}
@InProceedings{BEV,
    author = {Sun, Yu and Liu, Wu and Bao, Qian and Fu, Yili and Mei, Tao and Black, Michael J},
    title = {{Putting People in their Place: Monocular Regression of 3D People in Depth}},
    booktitle = {CVPR},
    year = {2022}}
@InProceedings{ROMP,
    author = {Sun, Yu and Bao, Qian and Liu, Wu and Fu, Yili and Michael J., Black and Mei, Tao},
    title = {{Monocular, One-stage, Regression of Multiple 3D People}},
    booktitle = {ICCV},
    year = {2021}}
```

## Acknowledgement
This work was supported by the National Key R&D Program of China under Grand No. 2020AAA0103800.  
**MJB Disclosure**: [https://files.is.tue.mpg.de/black/CoI_CVPR_2023.txt](https://files.is.tue.mpg.de/black/CoI_CVPR_2023.txt)
