### Textured SMPL visualization

1. Prepare the data:

Please make sure you have update your ROMP code with essential data for textured SMPL visualization.

The layout of smpl data is:
- models
  - smpl
    - smpl_male.vtk
    - SMPL_sampleTex_f.jpg
    - SMPL_sampleTex_m.jpg
    - uv_table.npy

2. Change the config:

Edit the config file, i.e. [configs/webcam.yml](../src/configs/webcam.yml).

```
webcam_mesh_color: 'male_tex' # 'male_tex' for using the male texture; 'female_tex' for using the female texture; 'ghostwhite'/'LightCyan' for using the single color texture.
```

3. Run the code:

```
cd src/
CUDA_VISIBLE_DEVICES=0 python core/test.py --gpu=0 --configs_yml=configs/webcam.yml
```
