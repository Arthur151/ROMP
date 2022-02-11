## Configuration

The configure yml files are under ROMP/configs.

### Basic settings

#### renderer (str)
Please choose the renderer ('pyrender' or 'pytorch3d') for visualizing the estimated mesh on input image. 'pyrender' can be only used on desktop. To train ROMP or run it on server without visualization hardware, please install 'pytorch3d' and set renderer to 'pytorch3d'.

#### GPUS (str)
GPU device number. 
To run the code on GPUs, please set it to the GPU device number, such as `GPUS: 0` or `GPUS: 0,1,2,3`.  To run the code on CPU, please set it to `GPUS: -1`.

#### inputs (str)
Path of the folder containing the input images / path of the input video file. 
For example, to run the code on the demo video frames, please change it to 
```bash
inputs: /path/to/project/ROMP/demo/videos/Messi_1
```

#### output_dir (str)
Output path of saving the estimated results. 
Without specifying the output path, the results would be saved in input_path+`_results` by default.

#### collect_subdirs (bool)
Whether to collect images from the sub-folder of the input path.   

#### save_visualization_on_img (bool)
Whether to render the estimated meshes on the input images. Rendering is not an essential part if you only need the estimated parameters. For a faster inference speed, you can set it to `False`.

#### show_mesh_stand_on_image (bool)
Whether to render the estimated meshes on the top of input images, like standing on it. 

#### interactive_vis (bool)
For interactive visualization, please set it to True. Default: False

#### soi_camera (str)
The camera mode of `show_mesh_stand_on_image`. Two options, 'close' or 'far'.  

#### save_mesh (bool)
Whether to save the mesh results in output_dir.

#### save_centermap (bool)
Whether to visualize the estimated centermap results in output_dir.  

#### save_dict_results  (bool)
Whether to save the estimated parameter results as a dict in output_dir.  

The predicted results of each image are saved in the following format:
```bash
image_path
| - subject_0
| - | - cam (3,) # 3 camera parameters of weak-perspective camera, (scale, tranlation_x, tranlation_y)
| - | - pose (72,) # 72 SMPL pose parameters.
| - | - betas (10,) # 10 SMPL shape parameters.
| - | - j3d_all54 (54, 3) # 3D keypoints coordinates regressed from the estimated body mesh.
| - | - j3d_smpl24 (24, 3) # 3D pose results in SMPL format
| - | - j3d_spin24 (24, 3) # 3D pose results in SPIN format
| - | - j3d_op25 (25, 3) # 3D pose results in Openpose format
| - | - verts (6890, 3) # 3D coordinates of 3D human mesh.
| - | - pj2d (54, 2) # 2D coordinates of 2D keypoints in padded input image.
| - | - pj2d_org (54, 2) # 2D coordinates of 2D keypoints in original input image.
| - | - trans (3,) # rough 3D translation converted from the estimated camera parameters.
| - | - center_conf (1,) # confidence value of the detected person on centermap.
| - subject_1
	...
```
You can parse the results from the saved dict (.npz) via
```bash
np.load('/path/to/*.npz',allow_pickle=True)['results'][()]
# for example:
np.load('ROMP/demo/images_results/3dpw_sit_on_street.npz',allow_pickle=True)['results'][()]
```

#### val_batch_size (int)
Batch size during inference/validation.

#### backbone (str)
To switch backbone from HRNet-32 to ResNet-50:
Set the `backbone: hrnet` to `backbone: resnet`, meanwhile change the checkout from `model_path: trained_models/ROMP_HRNet32_V1.pkl` to `model_path: trained_models/ROMP_ResNet50_V1.pkl`.

#### nw (int)
The number of dataloader workers. 

#### model_precision (str)
If you installed Pytorch 1.6+, then you can use the automatic mix precision (AMP), by setting`model_precision: fp16`


### Video setting

The video configure file is ROMP/configs/video.yml

#### show_largest_person_only (bool)
Whether to extract the results of the person shown in the largest scale in video.
Highly recommended to set it to `True`, if you only want to extract and smooth the main subject shown in the video. 

#### temporal_optimization (bool)
Whether to smooth the motion sequence.
Temporal motion smoothing is just a testing function. We need to first track the person identity through video to extract the motion sequence of each person. While tracking is prone to fail in crowded scenes. This is just a beta version. Try it if you need it.

#### fps_save (int)
The FPS of the saved video results. 


### Webcam setting

The webcam configure file is ROMP/configs/webcam.yml

#### visulize_platform (str)
Two options, 'integrated' or 'blender'. 
'integrated' means directly visulizes the results using build-in Open3D-based visualizer. 
'blender' means visulizes the results in 'blender'. Currently, we only support live animation of single character that has a similar skeleton as SMPL.

#### cam_id (int)
Web camera id, default 0. 

#### mesh_cloth (str)
The texture of the estimated mesh. 
We have a wardrobe in ROMP/model_data/wardrobe, to dress on the cloth, please find the cloth id in wardrobe (romp/lib/constants.py)
Feel free to paint the estimated mesh results in your favorite color.
Currently, we have LightCyan, ghostwhite, Azure, Cornislk, Honeydew, LavenderBlush. 
If your favorite color is not included, please add it to the mesh_color_dict (romp/lib/constants.py).

#### run_on_remote_server (bool)
Whether run webcam (captured locally) demo on remote server 

#### server_ip (str)
IP address of remote server.

#### server_port (int)
Port of remote server, default 10086.
Please change to the other port if 10086 has been occupied.
