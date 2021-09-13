# Monocular, One-stage, Regression of Multiple 3D People
[![Google Colab demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1oz9E6uIbj4udOPZvA1Zi9pFx0SWH_UXg)
[![arXiv](https://img.shields.io/badge/arXiv-2008.12272v3-00ff00.svg)](https://arxiv.org/pdf/2008.12272v3.pdf)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/centerhmr-a-bottom-up-single-shot-method-for/3d-human-pose-estimation-on-3dpw)](https://paperswithcode.com/sota/3d-human-pose-estimation-on-3dpw?p=centerhmr-a-bottom-up-single-shot-method-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/centerhmr-a-bottom-up-single-shot-method-for/3d-human-pose-estimation-on-3d-poses-in-the)](https://paperswithcode.com/sota/3d-human-pose-estimation-on-3d-poses-in-the?p=centerhmr-a-bottom-up-single-shot-method-for)

> [**Monocular, One-stage, Regression of Multiple 3D People**](https://arxiv.org/abs/2008.12272),            
> Yu Sun, Qian Bao, Wu Liu, Yili Fu, Michael J. Black, Tao Mei,        
> *arXiv paper ([arXiv 2008.12272](https://arxiv.org/abs/2008.12272))*

ROMP is a one-stage network for multi-person 3D mesh recovery from a single image.

- **Simple:** Concise one-stage framework for simultaneous person detection and 3D body mesh recovery.

- **Fast:** ROMP can run over *30* FPS on a 1070Ti GPU.

- **Strong**: ROMP achieves superior performance on multiple challenging multi-person/occlusion benchmarks.

- **Easy to use:** We provide user friendly testing API and webcam demos. 

Contact: [yusun@stu.hit.edu.cn](mailto:yusun@stu.hit.edu.cn). Feel free to contact me for related questions or discussions! 

### News
*2021/9/13: Low FPS / args parsing bugs are fixed.* Please refer to [this issue](https://github.com/Arthur151/ROMP/issues/52#issuecomment-917988564) for the details or just upgrade to the latest version. 
*2021/9/10: Training code release. API optimization.*  
[Old logs](docs/updates.md)

<p float="center">
  <img src="../assets/demo/animation/live_demo_guangboticao.gif" width="48%" />
  <img src="../assets/demo/animation/live_demo_sit.gif" width="48%" />
</p>

<p float="center">
  <img src="../assets/demo/animation/c1_results_compressed.gif" width="32%" />
  <img src="../assets/demo/animation/c4_results_compressed.gif" width="32%" />
  <img src="../assets/demo/animation/c0_results_compressed.gif" width="32%" />
</p>

### Try on Google Colab
Before installation, you can take a few minutes to try the prepared [Google Colab demo](https://colab.research.google.com/drive/1oz9E6uIbj4udOPZvA1Zi9pFx0SWH_UXg) a try.  
It allows you to run the project in the cloud, free of charge. 

Please refer to the [bug.md](docs/bugs.md) for unpleasant bugs. Welcome to submit the issues for related bugs.

### Installation

Please refer to [install.md](docs/installation.md) for installation.

### Inference

Currently, we support processing images, video or real-time webcam.    
Pelease refer to [config_guide.md](docs/config_guide.md) for configurations.  

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

#### Processing videos

```bash
cd ROMP
# change the `inputs` in configs/video.yml to /path/to/your/video file or a folder containing video frames, then run 
sh scripts/video.sh
# or run the command like
python -u -m romp.predict.video --inputs=demo/videos/sample_video.mp4 --output_dir=demo/sample_video_results
```

#### Webcam

To do this you just need to run:
```bash
cd ROMP
sh scripts/webcam.sh
```
Currently, limited by the visualization pipeline, the real-time webcam demo only visualize the results of the largest person in the frames.

### Train

Please prepare the training datasets following [dataset.md](docs/dataset.md), and then refer to [train.md](docs/train.md) for training.

### Evaluation

Please refer to [evaluation.md](docs/evaluation.md) for evaluation on benchmarks.

### Export

<p float="center">
  <img src="../assets/demo/animation/fbx_animation.gif" width="56%" />
  <img src="https://github.com/vltmedia/QuickMocap-BlenderAddon/raw/master/images/QuickMocap_v0.3.0.png" width="18%" />
</p>

##### Export to Blender FBX 

Please refer to [expert.md](docs/export.md) to export the results to fbx files for Blender usage. Currently, this function only support the single-person video cases. Therefore, please test it with `demo/videos/sample_video2_results/sample_video2.mp4`, whose results would be saved to `demo/videos/sample_video2_results`.

##### Blender Addons

[VLT Media](https://github.com/vltmedia) creates a [QuickMocap-BlenderAddon](https://github.com/vltmedia/QuickMocap-BlenderAddon) to  read the .npz file created by ROMP. Clean & smooth the resulting keyframes.


## TODO LIST

The code will be gradually open sourced according to:
- [ ] the schedule
  - [x] demo code for internet images / videos / webcam
  - [x] runtime optimization
  - [x] benchmark evaluation
  - [x] training
  - [ ] virtual character animation

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
