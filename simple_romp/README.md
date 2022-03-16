# Simple_ROMP

Simplified implementation of ROMP: Monocular, One-stage, Regression of Multiple 3D People, ICCV21

Please refer to https://github.com/Arthur151/ROMP for more details

## Installation

```
pip install setuptools numpy cython
pip install simple_romp
```

Or install from source:

```
pip install setuptools numpy cython
python setup.py install
```

## Usage

Webcam demo:
```
romp --mode=webcam --show
# to smooth the results, the smaller the smooth_coeff, the smoother:
romp --mode=webcam --show --temporal_optimize --smooth_coeff=3.
# to show largest person only:
romp --mode=webcam --show --temporal_optimize --smooth_coeff=3. --show_largest 
```

Processing a single image:
```
romp --mode=image --show --input=/path/to/image.jpg --save_path=/path/to/output/folder/
```

Processing a video / a folder of images:
```
romp --mode=video --show --input=/path/to/video.mp4 --save_path=/path/to/output/folder/
romp --mode=video --show --input=/path/to/video_frames/ --save_path=/path/to/output/folder/
# to smooth the results, the smaller the smooth_coeff, the smoother:
romp --mode=video --show --input=/path/to/video_frames/ --save_path=/path/to/output/folder/ --temporal_optimize --smooth_coeff=3.
```

## Copyright

Codes released under MIT license. All rights reserved by Yu Sun.
