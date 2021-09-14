<h1 align="center"> ROMP </h1>
<h2 align="center"> Monocular, One-stage, Regression of Multiple 3D People </h2>
[![Google Colab demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1oz9E6uIbj4udOPZvA1Zi9pFx0SWH_UXg)
[![arXiv](https://img.shields.io/badge/arXiv-2008.12272v3-00ff00.svg)](https://arxiv.org/pdf/2008.12272v3.pdf)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/centerhmr-a-bottom-up-single-shot-method-for/3d-human-pose-estimation-on-3dpw)](https://paperswithcode.com/sota/3d-human-pose-estimation-on-3dpw?p=centerhmr-a-bottom-up-single-shot-method-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/centerhmr-a-bottom-up-single-shot-method-for/3d-human-pose-estimation-on-3d-poses-in-the)](https://paperswithcode.com/sota/3d-human-pose-estimation-on-3d-poses-in-the?p=centerhmr-a-bottom-up-single-shot-method-for)

ROMP is a concise one-stage network for multi-person 3D mesh recovery from a single image.

- **Simple.** Concise one-stage framework for simultaneous person detection and 3D body mesh recovery.

- **Fast.** ROMP can run over *30* FPS on a 1070Ti GPU.

- **Strong** ROMP achieves superior performance on multiple challenging multi-person/occlusion benchmarks.

- **Easy to use.** We provide user friendly testing API and webcam demos. 

Contact: [yusun@stu.hit.edu.cn](mailto:yusun@stu.hit.edu.cn). Feel free to contact me for related questions or discussions! 

<p float="center">
  <img src="../assets/demo/animation/live_demo_guangboticao.gif" width="48%" />
  <img src="../assets/demo/animation/live_demo_sit.gif" width="48%" />
</p>

<p float="center">
  <img src="../assets/demo/animation/c1_results_compressed.gif" width="32%" />
  <img src="../assets/demo/animation/c4_results_compressed.gif" width="32%" />
  <img src="../assets/demo/animation/c0_results_compressed.gif" width="32%" />
</p>

## Table of contents
* [Features](#features)
* [News](#news)
* [Getting Started](#getting-started)
    * [Try on Google Colab](#try-on-google-colab)
    * [Installation](#installation)
    * [Inference](#inference)
        * [Processing images](#processing-images)
        * [Processing videos](#processing-videos)
        * [Webcam](#Webcam)
    * [Export](#export)
        * [Export to Blender FBX](#export-to-blender-fbx)
        * [Blender Addons](#blender-addons)
    * [Train](#train)
    * [Evaluation](#evaluation)
    * [Bugs report](#bugs-report)
* [Citation](#citation)
* [Contributor](#contributor)
* [Acknowledgement](#acknowledgement)

> [**Monocular, One-stage, Regression of Multiple 3D People**](https://arxiv.org/abs/2008.12272),            
> Yu Sun, Qian Bao, Wu Liu, Yili Fu, Michael J. Black, Tao Mei,        
> *arXiv paper ([arXiv 2008.12272](https://arxiv.org/abs/2008.12272))*

## Features
 - Running the examples on [Google Colab](https://colab.research.google.com/drive/1oz9E6uIbj4udOPZvA1Zi9pFx0SWH_UXg).  
 - Real-time online webcam demo for driving textured SMPL model with single-person motion. We also provide a wardrobe for changing clothes.  
 - Batch processing images/videos via command line / jupyter notebook / calling ROMP as a python lib.  
 - Exporting the captured single-person motion to FBX file for Blender/Unity usage.  
 - Training and evaluation for re-implementing our results presented in paper.  
 - Convenient API for 2D / 3D visualization, parsed datasets.  

## News
*2021/9/13: Low FPS / args parsing bugs are fixed. Support calling as a python lib.*   
*2021/9/10: Training code release. API optimization.*    
[Old logs](docs/updates.md)

## Getting started

### Try on Google Colab

It allows you to run the project in the cloud, free of charge.  
Let's give the prepared [Google Colab demo](https://colab.research.google.com/drive/1oz9E6uIbj4udOPZvA1Zi9pFx0SWH_UXg) a try.  

### Installation

Please refer to [install.md](docs/installation.md) for installation.

### Inference

Currently, we support processing images, video or real-time webcam.    
Pelease refer to [config_guide.md](docs/config_guide.md) for configurations.   

ROMP can be called as a python lib inside the python code, jupyter notebook, or from command line / scripts, please refer to [Google Colab demo](https://colab.research.google.com/drive/1oz9E6uIbj4udOPZvA1Zi9pFx0SWH_UXg) for examples.

#### Processing images

To re-implement the demo results, please run
```bash
cd ROMP
# change the `inputs` in configs/image.yml to /path/to/your/image folder, then run 
sh scripts/image.sh
# or run the command like
python -m romp.predict.image --inputs=demo/images --output_dir=demo/image_results
```
Please refer to [config_guide.md](docs/config_guide.md) for **saving the estimated mesh/Center maps/parameters dict**.

Here, we show an example of calling ROMP as a python lib.
```bash
# set the absolute path to ROMP
path_to_romp = '/path/to/ROMP'
import os,sys
sys.path.append(path_to_romp)
# set the detailed configurations
from romp.lib.config import ConfigContext, parse_args, args
ConfigContext.parsed_args = parse_args(["--configs_yml=configs/image.yml",'--inputs=/path/to/images_folder', '--output_dir=/path/to/save/image_results', '--save_centermap', False]) # Be caution that setting the bool configs needs two elements, ['--config', True/False]
# import the ROMP image processor
from romp.predict.image import Image_processor
processor = Image_processor(args_set=args())
results_dict = processor.run(args().inputs) # you can change the args().inputs to other /path/to/images_folder
````

#### Processing videos

```bash
cd ROMP
# change the `inputs` in configs/video.yml to /path/to/your/video file or a folder containing video frames, then run 
sh scripts/video.sh
# or run the command like
python -u -m romp.predict.video --inputs=demo/videos/sample_video.mp4 --output_dir=demo/sample_video_results
```

Here, we show an example of calling ROMP as a python lib.
```bash
# set the absolute path to ROMP
path_to_romp = '/path/to/ROMP'
import os,sys
sys.path.append(path_to_romp)
# set the detailed configurations
from romp.lib.config import ConfigContext, parse_args, args
ConfigContext.parsed_args = parse_args(["--configs_yml=configs/video.yml",'--inputs=/path/to/video', '--output_dir=/path/to/save/video_results', '--save_visualization_on_img',False]) # Be caution that setting the bool configs needs two elements, ['--config', True/False]
# import the ROMP image processor
from romp.predict.video import Video_processor
processor = Video_processor(args_set=args())
results_dict = processor.run(args().inputs) # you can change the args().inputs to other /path/to/video
````

#### Webcam

To do this you just need to run:
```bash
cd ROMP
sh scripts/webcam.sh
```
Currently, limited by the visualization pipeline, the real-time webcam demo only visualize the results of the largest person in the frames.

### Export

<p float="center">
  <img src="../assets/demo/animation/fbx_animation.gif" width="56%" />
  <img src="https://github.com/vltmedia/QuickMocap-BlenderAddon/raw/master/images/QuickMocap_v0.3.0.png" width="18%" />
</p>

##### Export to Blender FBX

Please refer to [expert.md](docs/export.md) to export the results to fbx files for Blender usage. Currently, this function only support the single-person video cases. Therefore, please test it with `demo/videos/sample_video2_results/sample_video2.mp4`, whose results would be saved to `demo/videos/sample_video2_results`.

##### Blender Addons

[VLT Media](https://github.com/vltmedia) creates a [QuickMocap-BlenderAddon](https://github.com/vltmedia/QuickMocap-BlenderAddon) to  read the .npz file created by ROMP. Clean & smooth the resulting keyframes.

### Train

Please prepare the training datasets following [dataset.md](docs/dataset.md), and then refer to [train.md](docs/train.md) for training.

### Evaluation

Please refer to [evaluation.md](docs/evaluation.md) for evaluation on benchmarks.

### Bugs report

Please refer to [bug.md](docs/bugs.md) for solutions. Welcome to submit the issues for related bugs. I will solve them as soon as possible.

## Citation
Please considering citing 
```bibtex
@InProceedings{ROMP,
author = {Sun, Yu and Bao, Qian and Liu, Wu and Fu, Yili and Michael J., Black and Mei, Tao},
title = {Monocular, One-stage, Regression of Multiple 3D People},
booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
month = {October},
year = {2021}
}
```

## Contributor

This repository is currently maintained by [Yu Sun](https://github.com/Arthur151). 

ROMP has also benefited from many developers, including 
 - [Marco Musy](https://github.com/marcomusy) : help in [the textured SMPL visualization](https://github.com/marcomusy/vedo/issues/371).
 - [Gavin Gray](https://github.com/gngdb) : adding support for an elegant context manager to run code in a notebook.
 - [VLT Media](https://github.com/vltmedia) : adding support for running on Windows & batch_videos.py.

## Acknowledgement

We thank [Peng Cheng](https://github.com/CPFLAME) for his constructive comments on Center map training.  

Here are some great resources we benefit:

- SMPL models and layer is borrowed from MPII [SMPL-X model](https://github.com/vchoutas/smplx).
- Some functions are borrowed from [HMR-pytorch](https://github.com/MandyMo/pytorch_HMR) and [SPIN](https://github.com/nkolot/SPIN).
- The evaluation code and GT annotations of 3DPW dataset is brought from [3dpw-eval](https://github.com/aymenmir1/3dpw-eval) and [VIBE](https://github.com/mkocabas/VIBE).
- 3D mesh visualization is supported by [vedo](https://github.com/marcomusy/vedo), [EasyMocap](https://github.com/zju3dv/EasyMocap), [minimal-hand](https://github.com/CalciferZh/minimal-hand) and [Open3D]( https://github.com/intel-isl/Open3D).

Please consider citing their papers.
