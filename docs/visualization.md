### Textured SMPL visualization

1. Prepare the data:

Please make sure you have downloaded the latest released ROMP data for textured SMPL visualization.

The layout of smpl data is:
- model_data
  - parameters
  - wardrobe

2. Change the config:

Edit the config file, i.e. [configs/webcam.yml](../configs/webcam.yml).

```
 # for picking up sepcific cloth from the wardrobe in model_data, please refer to romp/lib/constants.py
 # 'ghostwhite'/'LightCyan' for using the single color texture.
 mesh_cloth: '031' # '031' # 'random'
```
