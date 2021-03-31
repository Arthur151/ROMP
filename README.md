# Monocular, One-stage, Regression of Multiple 3D People
[![Google Colab demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1oz9E6uIbj4udOPZvA1Zi9pFx0SWH_UXg#scrollTo=s8gFtokdcEQo)

ROMP is a one-stage network for multi-person 3D mesh recovery from a single image.
> [**Monocular, One-stage, Regression of Multiple 3D People**](https://arxiv.org/abs/2008.12272),            
> Yu Sun, Qian Bao, Wu Liu, Yili Fu, Michael J. Black, Tao Mei,        
> *arXiv paper ([arXiv 2008.12272](https://arxiv.org/abs/2008.12272))*

Contact: [yusun@stu.hit.edu.cn](mailto:yusun@stu.hit.edu.cn). Feel free to contact me for related questions or discussions! 

- **Simple:** Simultaneously predicting the body center locations and corresponding 3D body mesh parameters for all people at each pixel.

- **Fast:** ROMP ResNet-50 model runs over *30* FPS on a 1070Ti GPU.

- **Strong**: ROMP achieves superior performance on multiple challenging multi-person/occlusion benchmarks, including 3DPW, CMU Panoptic, and 3DOH50K.

- **Easy to use:** We provide user friendly testing API and webcam demos. 

### News
*2021/3/30: 1.0 version.* Rebuilding the code. Release the ResNet-50 version and evaluation on 3DPW. 
*2020/11/26: Optimization for person-person occlusion.* Small changes for video support.  
*2020/9/11: Real-time webcam demo using local/remote server.* Please refer to [config_guide.md](src/config_guide.md) for details.  
*2020/9/4: Google Colab demo.* Saving a npy file per imag. Please refer to [config_guide.md](src/config_guide.md) for details.

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
Before installation, you can take a few minutes to try the prepared [Google Colab demo](https://colab.research.google.com/drive/1oz9E6uIbj4udOPZvA1Zi9pFx0SWH_UXg#scrollTo=s8gFtokdcEQo) a try.  
It allows you to run the project in the cloud, free of charge. 

Please refer to the [bug.md](src/bugs.md) for unpleasant bugs. Welcome to submit the issues for related bugs.

<p float="center">
  <img src="../assets/demo/images_results/images-3dpw_sit_on_street.jpg" width="48%" />
  <img src="../assets/demo/images_results/images-Cristiano_Ronaldo.jpg" width="48%" />
</p>

### Installation

Please refer to [install.md](src/lib/scripts/install.md) for installation.

### Demo

Currently, the released code is used to re-implement demo results. Only 1-2G GPU memory is needed.

To do this you just need to run
```bash
cd ROMP/src
sh run.sh
# if there are any bugs about shell script, please consider run the following command instead:
CUDA_VISIBLE_DEVICES=0 python core/test.py --gpu=0 --configs_yml=configs/single_image.yml
```
Results will be saved in ROMP/demo/images_results.

#### Internet images
You can also run the code on random internet images via putting the images under ROMP/demo/images before running ```sh run.sh```.

Please refer to [config_guide.md](src/config_guide.md) for **saving the estimated mesh/Center maps/parameters dict**.

#### Internet videos

You can also run the code on random internet videos.

To do this you just need to firstly change the input_video_path in src/configs/video.yml to /path/to/your/video. For example, set

```bash
 video_or_frame: True
 input_video_path: '../demo/sample_video.mp4' # None
```
then run 

```bash
cd ROMP/src
CUDA_VISIBLE_DEVICES=0 python core/test.py --gpu=0 --configs_yml=configs/video.yml
```
Results will be saved to `../demo/sample_video_results.mp4` and `../demo/sample_video_results.npz`.

#### Webcam

We also provide the webcam demo code, which can run at real-time on a 1070Ti GPU / remote server.  
Currently, limited by the visualization pipeline, the webcam visulization code only support the single-person mesh.

To do this you just need to run
```bash
cd ROMP/src
CUDA_VISIBLE_DEVICES=0 python core/test.py --gpu=0 --configs_yml=configs/webcam.yml
# or try to use the model with ResNet-50 as backbone.
CUDA_VISIBLE_DEVICES=0 python core/test.py --gpu=0 --configs_yml=configs/webcam_resnet.yml
```
Press Up/Down to end the demo. Pelease refer to [config_guide.md](src/config_guide.md) for running webcam demo on remote server, setting mesh color or camera id.

### Evaluation

Please refer to [evaluation.md](src/lib/scripts/evaluation.md) for evaluation on benchmarks.

## TODO LIST

The code will be gradually open sourced according to:
- [ ] the schedule
  - [x] demo code for internet images / videos / webcam
  - [x] runtime optimization
  - [x] benchmark evaluation
  - [ ] training

## Citation
Please considering citing 
```bibtex
@inproceedings{ROMP,
  title = {Monocular, One-stage, Regression of Multiple 3D People},
  author = {Yu, Sun and Qian, Bao and Wu, Liu and Yili, Fu and Black, Michael J. and Tao, Mei},
  booktitle = {arxiv:2008.12272},
  month = {August},
  year = {2020}
}
```

## Acknowledgement

We thank [Peng Cheng](https://github.com/CPFLAME) for his constructive comments on Center map training.

Here are some great resources we benefit:

- SMPL models and layer is borrowed from MPII [SMPL-X model](https://github.com/vchoutas/smplx).
- Webcam pipeline is borrowed from [minimal-hand](https://github.com/CalciferZh/minimal-hand).
- Some functions are borrowed from [HMR-pytorch](https://github.com/MandyMo/pytorch_HMR).
- Some functions for data augmentation are borrowed from [SPIN](https://github.com/nkolot/SPIN).
- Synthetic occlusion is borrowed from [synthetic-occlusion](https://github.com/isarandi/synthetic-occlusion)
- The evaluation code of 3DPW dataset is brought from [3dpw-eval](https://github.com/aymenmir1/3dpw-eval).   
- For fair comparison, the GT annotations of 3DPW dataset are brought from [VIBE](https://github.com/mkocabas/VIBE)
