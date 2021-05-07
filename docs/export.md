### Export the results to fbx

1. Prepare the data:

Please register at [this link](https://smpl.is.tue.mpg.de/) and download the SMPL_unity_v.1.0.0.zip from SMPL for Unity.

Then set the path of the downloaded files at [convert_fbx.py](../src/lib/utils/convert_fbx.py). For example,

```
male_model_path = '/home/yusun/Desktop/unity/SMPL_unity_v.1.0.0/smpl/Models/SMPL_m_unityDoubleBlends_lbs_10_scale5_207_v1.0.0.fbx'
female_model_path = '/home/yusun/Desktop/unity/SMPL_unity_v.1.0.0/smpl/Models/SMPL_f_unityDoubleBlends_lbs_10_scale5_207_v1.0.0.fbx'
```
You can also choose the gender of animated model via setting `gender=male` or `gender=female`

2. Install the Blender:

I have built the Blender 2.91 on Ubuntu 18.04 LTS following [this instructions](https://github.com/TylerGubala/blenderpy) via:
```pip install bpy && bpy_post_install```

3. Run the code:

```
cd src/
python lib/utils/convert_fbx.py --input=/home/yusun/ROMP/demo/videos/sample_video2_results/sample_video2_results.npz --output=../demo/videos/sample_video2.fbx --gender=female
```

4.Open the fbx animation in Blender:

File -> Import -> FBX(.fbx)

Now, you can display the estimated animation in Blender via pushing the play button at the bottom.
