## Configuration

The configure file is CenterHMR/src/configs/basic_test.yml

###### demo_image_folder: absoluate path of the folder containing the input images
Please change the 
```bash
demo_image_folder: None in CenterHMR/src/configs/basic_test.yml to ''demo_image_folder: absoluate path to the image folder''
```

For example, to run the code on the provided video frames (contained in CenterHMR_data.zip), please change it to 
```bash
demo_image_folder: /path/to/project/CenterHMR/demo/videos/Messi_1
```
Results would be saved at /path/to/project/CenterHMR/demo/videos/Messi_1_results.

###### save_mesh: 

if save the mesh results, please set it to True. The obj files will be saved to demo_image_folder+\_results.

###### save_centermap: 

if save the estimated Center maps, please set it to True. The visualized images will be save to demo_image_folder+\_results.

###### save_dict_results: 

if save the estimated parameters, please set it to True. They will be save to demo_image_folder+\.npz.
```bash
## load the results
np.load('/path/to/*.npz',allow_pickle=True)['results'][()]
## for example:
np.load('~/CenterHMR/demo/images_results/person_overlap.npz',allow_pickle=True)['results'][()]
```

The predicted results of each image are saved in the following format:
```bash
image_name
 - cam (3,) # 3 parameters of weak-perspective camera, (scale, tranlation_x, tranlation_y)
 - pose (72,) # 72 SMPL pose parameters.
 - betas (10,) # 10 SMPL shape parameters.
 - j3d_smpl24 (24, 3) # 3D pose results in SMPL format
 - j3d_op25 (25, 3) # 3D pose results in Openpose format
 - verts (6890, 3) # 3D coordinates of 3D human mesh.
```

###### val_batch_size: batch size 

###### nw: the number of dataloader workers. 

###### model_precision: mix precision

If you installed Pytorch 1.6 or upper verion, then you can use the automatic mix precision (AMP), by setting
```bash
model_precision: fp16
```
