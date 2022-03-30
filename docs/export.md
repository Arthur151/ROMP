## Export

Currently, this function only support the single-person video cases. Therefore, please test it with `demo/videos/sample_video2_results/sample_video2.mp4`, whose results would be saved to `demo/videos/sample_video2_results`.

#### Blender Addons
[Chuanhang Yan](https://github.com/yanch2116) : developing an [addon for driving character in Blender](https://github.com/yanch2116/Blender-addons-for-SMPL).  
[VLT Media](https://github.com/vltmedia) creates a [QuickMocap-BlenderAddon](https://github.com/vltmedia/QuickMocap-BlenderAddon) to  read the .npz file created by ROMP. Clean & smooth the resulting keyframes.  

### Blender character animation

1. Download the [BlenderAddon](https://github.com/yanch2116/LiveMocap-BlenderAddon) and install the [Blender](https://www.blender.org/).
2. Install the Addon in Blender:
Edit -> Preferences -> Add-ons -> install -> select ROMP/romp/exports/blender_mocap.py  
Click to active the 'Real Time Mocap' add-on.  
3. Run the ROMP webcam demo code:  
```
cd ROMP
sh scripts/webcam_blender.sh
```

### Export the results to fbx

<p float="center">
  <img src="../../assets/demo/animation/fbx_animation.gif" width="40%" />
</p>

Currently, this function can only export the motion of a single person at each time. Therefore, please test it with `demo/videos/sample_video2_results/sample_video2.mp4`, whose results would be saved to `demo/videos/sample_video2_results`.

1. Prepare the data:

Please register at [this link](https://smpl.is.tue.mpg.de/) and download the SMPL_unity_v.1.0.0.zip from SMPL for Unity.

Then set the path of the downloaded files at [convert_fbx.py](../export/convert_fbx.py). For example,

```
male_model_path = '/home/yusun/Desktop/unity/SMPL_unity_v.1.0.0/smpl/Models/SMPL_m_unityDoubleBlends_lbs_10_scale5_207_v1.0.0.fbx'
female_model_path = '/home/yusun/Desktop/unity/SMPL_unity_v.1.0.0/smpl/Models/SMPL_f_unityDoubleBlends_lbs_10_scale5_207_v1.0.0.fbx'
```
You can also choose the gender of animated model via setting `gender=male` or `gender=female`

2. Install the Blender:

Generally, Blender 2.91 can be installed following [this instructions](https://github.com/TylerGubala/blenderpy) via:  
```pip install bpy && bpy_post_install```

If you use python 3.7, bpy can be easily installed via  
```
pip install https://github.com/TylerGubala/blenderpy/releases/download/v2.91a0/bpy-2.91a0-cp37-cp37m-manylinux2014_x86_64.whl && bpy_post_install
```

3. Run the code:

```
cd ROMP/
# on Linux
python export/convert_fbx.py --input=demo/videos/sample_video2_results/sample_video2_results.npz --output=demo/videos/sample_video2.fbx --gender=female
# on Windows
python export\convert_fbx.py --input=demo\videos\sample_video2_results\sample_video2_results.npz --output=demo\videos\sample_video2.fbx --gender=female
```

4.Open the fbx animation in Blender:

File -> Import -> FBX(.fbx)

Now, you can display the estimated animation in Blender via pushing the play button at the bottom.

