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
*2021/9/10: Training code release. API optimization. *
*2021/7/15: Adding support for an elegant context manager to run code in a notebook.*  See [Colab demo](https://colab.research.google.com/drive/1oz9E6uIbj4udOPZvA1Zi9pFx0SWH_UXg) for the details.  
*2021/4/19: Adding support for textured SMPL mesh using [vedo](https://github.com/marcomusy/vedo).* See [visualization.md](docs/visualization.md) for the details.  
*2021/3/30: 1.0 version.* Rebuilding the code. Release the ResNet-50 version and evaluation on 3DPW.   
*2020/11/26: Optimization for person-person occlusion.* Small changes for video support.   
*2020/9/11: Real-time webcam demo using local/remote server.* 
*2020/9/4: Google Colab demo.* Saving a npy file per imag. 

<p float="center">
  <img src="assets/demo/animation/live_demo_guangboticao.gif" width="48%" />
  <img src="assets/demo/animation/live_demo_sit.gif" width="48%" />
</p>

<p float="center">
  <img src="assets/demo/animation/c1_results_compressed.gif" width="32%" />
  <img src="assets/demo/animation/c4_results_compressed.gif" width="32%" />
  <img src="assets/demo/animation/c0_results_compressed.gif" width="32%" />
</p>

### Try on Google Colab
Before installation, you can take a few minutes to try the prepared [Google Colab demo](https://colab.research.google.com/drive/1oz9E6uIbj4udOPZvA1Zi9pFx0SWH_UXg) a try.  
It allows you to run the project in the cloud, free of charge. 

Please refer to the [bug.md](docs/bugs.md) for unpleasant bugs. Welcome to submit the issues for related bugs.

### Installation

Please refer to [install.md](docs/installation.md) for installation.

### Processing images

To re-implement the demo results, please run
```bash
cd ROMP
sh scripts/image.sh
# if there are any bugs about shell script, please consider run the following command instead:
python -u -m romp.predict.image --configs_yml='configs/image.yml'
```
Results will be saved in ROMP/demo/images_results. You can also run the code on other images via putting the images under ROMP/demo/images or passing the path of image folder via
```bash
python -u -m romp.predict.image --inputs=/path/to/image_folder --output_dir='demo/image_results'
```
Please refer to [config_guide.md](docs/config_guide.md) for **saving the estimated mesh/Center maps/parameters dict**.

#### Processing videos

To process videos, you can change the `inputs` in configs/video.yml to /path/to/your/video, then run 
```bash
cd ROMP
sh scripts/video.sh
```
or simply run the command like
```bash
python -u -m romp.predict.video --inputs=demo/videos/sample_video.mp4 --output_dir='demo/sample_video_results'
```

#### Webcam

We also provide the webcam demo code, which can run at real-time on a 1070Ti GPU / remote server.  
Currently, limited by the visualization pipeline, the webcam visualization code only support the single-person mesh.

To do this you just need to run:
```bash
cd ROMP
sh scripts/webcam.sh
```
Pelease refer to [config_guide.md](docs/config_guide.md) for configurations.

### Blender

##### Export to Blender FBX 

<p float="center">
  <img src="assets/demo/animation/fbx_animation.gif" width="50%" />
</p>

Please refer to [expert.md](docs/export.md) to export the results to fbx files for Blender usage. Currently, this function only support the single-person video cases. Therefore, please test it with `demo/videos/sample_video2_results/sample_video2.mp4`, whose results would be saved to `demo/videos/sample_video2_results`.
##### Blender Addons

- [vltmedia/QuickMocap-BlenderAddon: Use this Blender Addon to import & clean Mocap Pose data from .npz or .pkl files. These files may have been created using Numpy, ROMP, or other motion capture processes that package their files accordingly. (github.com)](https://github.com/vltmedia/QuickMocap-BlenderAddon)
  - Reads the .npz file created by ROMP. Clean & smooth the resulting keyframes.
  - ![Quick Mocap v0.3.0](https://github.com/vltmedia/QuickMocap-BlenderAddon/raw/master/images/QuickMocap_v0.3.0.png)

### Evaluation

Please refer to [evaluation.md](docs/evaluation.md) for evaluation on benchmarks.

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

## Acknowledgement

We thank [Peng Cheng](https://github.com/CPFLAME) for his constructive comments on Center map training.  

Thanks to [Marco Musy](https://github.com/marcomusy) for his help in [the textured SMPL visualization](https://github.com/marcomusy/vedo/issues/371).

Thanks to [Gavin Gray](https://github.com/gngdb) for adding support for an elegant context manager to run code in a notebook via [this pull](https://github.com/Arthur151/ROMP/pull/58).

Thanks to [VLT Media](https://github.com/vltmedia) for adding support for running on Windows & batch_videos.py.

Here are some great resources we benefit:

- SMPL models and layer is borrowed from MPII [SMPL-X model](https://github.com/vchoutas/smplx).
- Webcam pipeline is borrowed from [minimal-hand](https://github.com/CalciferZh/minimal-hand).
- Some functions are borrowed from [HMR-pytorch](https://github.com/MandyMo/pytorch_HMR).
- Some functions for data augmentation are borrowed from [SPIN](https://github.com/nkolot/SPIN).
- Synthetic occlusion is borrowed from [synthetic-occlusion](https://github.com/isarandi/synthetic-occlusion).
- The evaluation code of 3DPW dataset is brought from [3dpw-eval](https://github.com/aymenmir1/3dpw-eval).   
- For fair comparison, the GT annotations of 3DPW dataset are brought from [VIBE](https://github.com/mkocabas/VIBE).
- 3D mesh visualization is supported by [vedo](https://github.com/marcomusy/vedo), [EasyMocap](https://github.com/zju3dv/EasyMocap) and [Open3D]( https://github.com/intel-isl/Open3D).
