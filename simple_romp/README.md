# Simple_ROMP

Simplified implementation of ROMP: Monocular, One-stage, Regression of Multiple 3D People, ICCV21

Please refer to https://github.com/Arthur151/ROMP for more details

## Installation

```
pip install --upgrade setuptools numpy cython black
```

```
pip install simple_romp
```
or download the package and install it from source:
```
python setup.py install
```

## Usage

Webcam demo:
```
romp --mode=webcam --show
```
For Mac Users, please use the original terminal instead of other terminal app (e.g. iTerm2) to avoid the bug `zsh: abort`.

Processing a single image:
```
romp --mode=image --calc_smpl --render_mesh --input=/path/to/image.jpg --save_path=/path/to/results.jpg
```

Processing a folder of images:
```
romp --mode=video --calc_smpl --render_mesh  --input=/path/to/image/folder/ --save_path=/path/to/output/folder/
```

Processing a video:
```
romp --mode=video --calc_smpl --render_mesh  --input=/path/to/video.mp4 --save_path=/path/to/output/folder/results.mp4 --save_video
```

Optional functions:
```
# show the results during processing image / video, add:
--show

# to smooth the results in webcam / video processing, add: (the smaller the smooth_coeff, the smoother) 
--temporal_optimize --smooth_coeff=3.

# to show the largest person only (remove the small subjects in background), add:
--show_largest 
```
More options, see `romp -h`

## Copyright

Codes released under MIT license. All rights reserved by Yu Sun.
