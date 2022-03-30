## Inference

Currently, we support processing images, video or real-time webcam.    
Pelease refer to [config_guide.md](docs/config_guide.md) for configurations.   
ROMP can be called as a python lib inside the python code, jupyter notebook, or from command line / scripts, please refer to [Google Colab demo](https://colab.research.google.com/drive/1oz9E6uIbj4udOPZvA1Zi9pFx0SWH_UXg) for examples.

### Processing images

To re-implement the demo results, please run
```bash
cd ROMP
# on Linux
python -m romp.predict.image --inputs=demo/images --output_dir=demo/image_results
# change the `inputs` in configs/image.yml to /path/to/your/image folder, then run 
sh scripts/image.sh

# on Windows
python -m romp.predict.image --inputs=demo\images --output_dir=demo\image_results
```
Please refer to [config_guide.md](docs/config_guide.md) for **saving the estimated mesh/Center maps/parameters dict**.

<p float="center">
  <img src="../assets/demo/animation/image_demo1-min.gif" width="32%" />
  <img src="../assets/demo/animation/image_demo2-min.gif" width="32%" />
  <img src="../assets/demo/animation/image_demo3-min.gif" width="32%" />
</p>

For interactive visualization, please run
```bash
# on Linux
python -m romp.predict.image --inputs=demo/images --output_dir=demo/image_results --show_mesh_stand_on_image  --interactive_vis
# on Windows
python -m romp.predict.image --inputs=demo\images --output_dir=demo\image_results --show_mesh_stand_on_image  --interactive_vis
```

**Caution**: To use `show_mesh_stand_on_image` and `interactive_vis`, you must run ROMP on a computer with visual desktop to support the rendering. Most remote servers without visual desktop is not supported. 

### Processing videos

<p float="center">
  <img src="../assets/demo/animation/c1_results_compressed.gif" width="32%" />
  <img src="../assets/demo/animation/c4_results_compressed.gif" width="32%" />
  <img src="../assets/demo/animation/c0_results_compressed.gif" width="32%" />
</p>

```bash
cd ROMP
# on Linux
python -m romp.predict.video --inputs=demo/videos/sample_video.mp4 --output_dir=demo/sample_video_results --save_dict_results
# or you can set all configurations in configs/video.yml, then run 
sh scripts/video.sh

# on Windows
python -m romp.predict.video --inputs=demo\videos\sample_video.mp4 --output_dir=demo\sample_video_results --save_dict_results
```

We notice that some users only want to extract the motion of **the formost person**, like this
<p float="center">
<img src="../assets/demo/animation/video_demo_nofp.gif" width="32%" />
  <img src="../assets/demo/animation/video_demo_fp.gif" width="40%" />
</p>
To achieve this, please run  

```bash
# on Linux
python -m romp.predict.video --inputs=demo/videos/demo_video_frames --output_dir=demo/demo_video_fp_results --show_largest_person_only --save_dict_results --show_mesh_stand_on_image 
```

All functions can be combined or work individually. Welcome to try them.

### Webcam

<p float="center">
  <img src="../assets/demo/animation/live_demo_sit.gif" width="48%" />
  <img src="../assets/demo/animation/live_demo1-min.gif" width="48%" />
</p>


To do this you just need to run:
```bash
cd ROMP
# on Linux
sh scripts/webcam.sh

# on Windows
python -u -m romp.predict.webcam --configs_yml='configs\webcam.yml'
```
To drive a character in Blender, please refer to [expert.md](docs/export.md).