# Simple_ROMP

Simplified implementation of ROMP [ICCV21], BEV [CVPR22], and TRACE [CVPR23].

As shown in [the main page](https://github.com/Arthur151/ROMP), the differences between ROMP, BEV, and TRACE are:  
ROMP has a lighter head to efficiently estimate the SMPL 3D pose/shape parameters and rough 2D position/scale of people in the image.  
BEV explicitly reasons about relative depths of people and support all age groups with SMPL+A model.  
TRACE tracks specific subjects shown in the first frame and recover their 3D trajectories in global coordinates.  

Please note that these instructions are just for ROMP and BEV.   
If you want to use TRACE as well, please refer to [this instrcution](trace2/README.md). 

## Installation

1. Installing simple_romp via pip:

```
pip install --upgrade setuptools numpy cython lap
```

```
pip install simple_romp==1.1.3
```
or download the package and install it from source:
```
python setup.py install
```

For Mac users, we strongly recommand to upgrade your pytorch to the latest version to support more basic operations used in BEV. To achieve this, please run
```
pip install --upgrade torch torchvision
```

[中国人专属-百度网盘模型下载](https://pan.baidu.com/s/1OuEpHr8assuX4UHJYeppRw?pwd=jm9i)（如果在国内访问Github不便请从这里下载除SMPL和SMIL外所有模型文件，并放到~/.romp/里）

2. Preparing SMPL model files in our format:

Firstly, please register and download:  
a. Meta data from [this link](https://github.com/Arthur151/ROMP/releases/download/V2.0/smpl_model_data.zip). Please unzip it, then we get a folder named "smpl_model_data"
b. SMPL model file (SMPL_NEUTRAL.pkl) from "Download version 1.1.0 for Python 2.7 (female/male/neutral, 300 shape PCs)" in [official website](https://smpl.is.tue.mpg.de/). Please unzip it and move the SMPL_NEUTRAL.pkl from extracted folder into the "smpl_model_data" folder.      
c. (Optional) If you use BEV, please also download SMIL model file (DOWNLOAD SMIL) from [official website](https://www.iosb.fraunhofer.de/en/competences/image-exploitation/object-recognition/sensor-networks/motion-analysis.html). Please unzip and put it into the "smpl_model_data" folder, so we have "smpl_model_data/smil/smil_web.pkl".   
Then we can get a folder in structure like this:  
```
|-- smpl_model_data
|   |-- SMPL_NEUTRAL.pkl
|   |-- J_regressor_extra.npy
|   |-- J_regressor_h36m.npy
|   |-- smpl_kid_template.npy
|   |-- smil
|   |-- |-- smil_web.pkl
```

Secondly, please convert the SMPL model files to our format via  
```
# please provide the absolute path of the "smpl_model_data" folder to the source_dir 
romp.prepare_smpl -source_dir=/path/to/smpl_model_data
# (Optional) If you use BEV, please also run:
bev.prepare_smil -source_dir=/path/to/smpl_model_data
```
The converted file would be save to "~/.romp/" in defualt. 

Please don't worry if there are bugs reporting during "Preparing SMPL model files". It is fine as long as we get things like this in "~/.romp/".
```
|-- .romp
|   |-- SMPL_NEUTRAL.pth
|   |-- SMPLA_NEUTRAL.pth
|   |-- smil_packed_info.pth
```

## Usage

Webcam demo:
```
romp --mode=webcam --show 
bev --mode=webcam --show
```
For faster inference with romp on CPU, you may run `romp --mode=webcam --show --onnx` instead.  
For Mac Users, please use the original terminal instead of other terminal app (e.g. iTerm2) to avoid the bug `zsh: abort`.

<p float="center">
  <img src="../../assets/demo/animation/image_demo2-min.gif" width="32%" />
  <img src="../../assets/demo/animation/Solvay_conference_1927_all_people.png" width="32%" />
  <img src="../../assets/demo/animation/conference_mesh_rotating.gif" width="32%" /></div>
</p>
</p>

In this [folder](https://github.com/Arthur151/ROMP/tree/assets/demo/BEV_demo_images), we prepare some demo images for testing.

Processing a single image:
```
romp --mode=image --calc_smpl --render_mesh -i=/path/to/image.jpg -o=/path/to/results.jpg
bev -i /path/to/image.jpg -o /path/to/results.jpg
```

Processing a folder of images:
```
romp --mode=video --calc_smpl --render_mesh -i=/path/to/image/folder/ -o=/path/to/output/folder/
bev -m video -i /path/to/image/folder/ -o /path/to/output/folder/
```
<p float="center">
  <img src="../../assets/demo/animation/c1_results_compressed.gif" width="32%" />
  <img src="../../assets/demo/animation/c4_results_compressed.gif" width="32%" />
  <img src="../../assets/demo/animation/c0_results_compressed.gif" width="32%" />
</p>

Processing a video:
```
romp --mode=video --calc_smpl --render_mesh -i=/path/to/video.mp4 -o=/path/to/output/folder/results.mp4 --save_video
bev -m video -i /path/to/video.mp4 -o /path/to/output/folder/results.mp4 --save_video
```

Common optional functions:
```
# show the results during processing image / video, add:
--show

# items you want to visualized, including mesh,pj2d,j3d,mesh_bird_view,mesh_side_view,center_conf,rotate_mesh. Please add if you want to see more:
--show_items=mesh,mesh_bird_view
```

ROMP only optional functions:
```
# to smooth the results in webcam / video processing, add: (the smaller the smooth_coeff (sc) is, the smoother the motion would be) 
-t -sc=3.

# to use the onnx version of ROMP for faster inference, please add:
--onnx

# to show the largest person only (remove the small subjects in background), add:
--show_largest 
```
<p float="center">
<img src="../../assets/demo/animation/video_demo_nofp.gif" width="32%" />
  <img src="../../assets/demo/animation/video_demo_fp.gif" width="40%" />
</p>
More options, see `romp -h`

Note that if you are using CPU for ROMP inference, we highly recommand to add `--onnx` for much faster speed.

### Calling as python lib

Both ROMP and BEV can be called as a python lib for further development.

```
import romp
settings = romp.main.default_settings 
# settings is just a argparse Namespace. To change it, for instance, you can change mode via
# settings.mode='video'
romp_model = romp.ROMP(settings)
outputs = romp_model(cv2.imread('path/to/image.jpg')) # please note that we take the input image in BGR format (cv2.imread).

import bev
settings = bev.main.default_settings
# settings is just a argparse Namespace. To change it, for instance, you can change mode via
# settings.mode='video'
bev_model = bev.BEV(settings)
outputs = bev_model(cv2.imread('path/to/image.jpg')) # please note that we take the input image in BGR format (cv2.imread).
```

### Export motion to .fbx / .glb / .bvh

Please refer to [export.md](doc/export.md) for details.

### Convert checkpoints
To convert the trained ROMP model '.pkl' (like ROMP.pkl) to simple-romp '.pth' model, please run
```
cd /path/to/ROMP/simple_romp/
python tools/convert_checkpoints.py ROMP.pkl ROMP.pth
```

### How to load the results saved in .npz file

```
import numpy as np
results = np.load('/path/to/results.npz',allow_pickle=True)['results'][()]
```

### Joints in output .npz file

We generate 2D/3D position of 71 joints from estimated 3D body meshes.   
The 71 joints are 24 SMPL joints + 30 extra joints + 17 h36m joints:
```
SMPL_24 = {
'Pelvis_SMPL':0, 'L_Hip_SMPL':1, 'R_Hip_SMPL':2, 'Spine_SMPL': 3, 'L_Knee':4, 'R_Knee':5, 'Thorax_SMPL': 6, 'L_Ankle':7, 'R_Ankle':8,'Thorax_up_SMPL':9,
'L_Toe_SMPL':10, 'R_Toe_SMPL':11, 'Neck': 12, 'L_Collar':13, 'R_Collar':14, 'Jaw':15, 'L_Shoulder':16, 'R_Shoulder':17,
'L_Elbow':18, 'R_Elbow':19, 'L_Wrist': 20, 'R_Wrist': 21, 'L_Hand':22, 'R_Hand':23}
SMPL_EXTRA_30 = {
'Nose':24, 'R_Eye':25, 'L_Eye':26, 'R_Ear': 27, 'L_Ear':28,
'L_BigToe':29, 'L_SmallToe': 30, 'L_Heel':31, 'R_BigToe':32,'R_SmallToe':33, 'R_Heel':34,
'L_Hand_thumb':35, 'L_Hand_index': 36, 'L_Hand_middle':37, 'L_Hand_ring':38, 'L_Hand_pinky':39,
'R_Hand_thumb':40, 'R_Hand_index':41,'R_Hand_middle':42, 'R_Hand_ring':43, 'R_Hand_pinky': 44,
'R_Hip': 45, 'L_Hip':46, 'Neck_LSP':47, 'Head_top':48, 'Pelvis':49, 'Thorax_MPII':50,
'Spine_H36M':51, 'Jaw_H36M':52, 'Head':53}
```
H36m 17 joints are just regressed them for fair comparison with previous methods. I am not sure their precise joint names.


## Copyright

Codes released under MIT license. All rights reserved by Yu Sun.
