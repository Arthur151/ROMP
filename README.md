# CenterHMR: a bottom-up single-shot method for multi-person 3D mesh recovery from a single image

The method achieves ECCV 2020 3DPW Challenge Runner Up. Please refer to [arxiv paper](https://arxiv.org/abs/2008.12272) for the details!

<p float="center">
  <img src="../assets/demo/animation/c1_results_compressed.gif" width="32%" />
  <img src="../assets/demo/animation/c5_results_compressed.gif" width="32%" />
  <img src="../assets/demo/animation/c0_results_compressed.gif" width="32%" />
</p>

<p float="center">
  <img src="../assets/demo/animation/c4_results_compressed.gif" width="32%" />
  <img src="../assets/demo/animation/c2_results_compressed.gif" width="32%" />
  <img src="../assets/demo/animation/c3_results_compressed.gif" width="32%" />
</p>

## Installation

#### Download models

###### Option 1:

Directly download the full-packed released package [CenterHMR.zip](https://github.com/Arthur151/CenterHMR/releases/download/v0.0/CenterHMR.zip) from github, latest version v0.0.

###### Option 2:

Clone the repo:
```bash
git clone https://github.com/Arthur151/CenterHMR --depth 1
```

Then download the CenterHMR data from [Google drive](https://drive.google.com/file/d/1vAiuallhHEV3WVq36u0gy7uzbG38d5sU/view?usp=sharing) or [Baidu Drive](https://pan.baidu.com/s/13XTwBy31zhLZLerI3V-rQA) with password ```6hye```. 

Unzip the downloaded CenterHMR_data.zip under the root CenterHMR/. The layout would be
```bash
CenterHMR
  - demo
  - models
  - src
  - trained_models
```

#### Set up environments

Please intall the Pytorch 1.6 from [the official website](https://pytorch.org/). We have tested the code on Ubuntu and Centos using Pytorch 1.6 only. 

Install packages:
```bash
cd CenterHMR/src
sh scripts/setup.sh
```

Please refer to the [bug.md](src/bugs.md) for unpleasant bugs. Feel free to submit the issues for related bugs.

<p float="center">
  <img src="../assets/demo/images_results/images-3dpw_sit_on_street.jpg" width="32%" />
  <img src="../assets/demo/images_results/images-Cristiano_Ronaldo.jpg" width="32%" />
  <img src="../assets/demo/images_results/images-person_overlap.jpg" width="32%" />
</p>

### Demo

Currently, the released code is used to re-implement demo results. Only 1-2G GPU memery is needed.

To do this you just need to run
```bash
cd CenterHMR/src
sh run.sh
```
Results will be saved in CenterHMR/demo/images_results.

#### Internet images
You can also run the code on random internet images via putting the images under CenterHMR/demo/images before running sh run.sh.

Or please refer to [config_guide.md](src/config_guide.md) for detail configurations.

Please refer to [config_guide.md](src/config_guide.md) for saving the estimated mesh/Center maps.


## TODO LIST

The code will be gradually open sourced according to:
- [ ] the schedule
  - [x] demo code for internet images or videos
  - [ ] evaluation code for re-implementation the results on 3DPW Challenge (really close)
  - [ ] runtime optimization

## Citation
Please considering citing 
```bibtex
@inproceedings{CenterHMR,
  title = {CenterHMR: a Bottom-up Single-shot Method for Multi-person 3D Mesh Recovery from a Single Image},
  author = {Yu, Sun and Qian, Bao and Wu, Liu and Yili, Fu and Tao, Mei},
  booktitle = {arxiv:2008.12272},
  month = {August},
  year = {2020}
}
```

## Acknowledgement

We thank [Peng Cheng](https://github.com/CPFLAME) for his constructive comments on Center map training.

Here are some great resources we benefit:

- SMPL models and layer is from [SMPL-X model](https://github.com/vchoutas/smplx).
- Some functions are borrowed from [HMR-pytorch](https://github.com/MandyMo/pytorch_HMR).
- Some functions for data augmentation are borrowed from [SPIN](https://github.com/nkolot/SPIN).
- Synthetic occlusion is borrowed from [synthetic-occlusion](https://github.com/isarandi/synthetic-occlusion)
- The evaluation code of 3DPW dataset is brought from [3dpw-eval](https://github.com/aymenmir1/3dpw-eval).   
- For fair comparison, the GT annotations of 3DPW dataset are brought from[VIBE](https://github.com/mkocabas/VIBE)
