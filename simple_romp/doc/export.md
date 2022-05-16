## Export 

In this part, we introduce how to export .fbx / .glb / .bvh from simple-romp predictions.

### Installation

The blender python API requires python3.7 env, so we need to install bpy in python3.7 via
```
conda create -n romp_export python==3.7
conda activate romp_export
pip install future-fstrings mathutils==2.81.2
```
Then, please follow the instruction at https://github.com/TylerGubala/blenderpy/releases to install the bpy.
For example, for ubuntu users, please first download the [bpy .whl package](https://github.com/TylerGubala/blenderpy/releases/download/v2.91a0/bpy-2.91a0-cp37-cp37m-manylinux2014_x86_64.whl) and then install via
```
pip install /path/to/downloaded/bpy-2.91a0-cp37-cp37m-manylinux2014_x86_64.whl && bpy_post_install
```

### Usage

Please change the input / output path in simple_romp/export.sh, for instance
```
python tools/convert2fbx.py --input=/home/yusun/BEV_results/video_results.npz --output=/home/yusun/BEV_results/dance.fbx --gender=female
```
You can also assign the subject ID of the motion you want to avoid the interaction via
```
python tools/convert2fbx.py --input=/home/yusun/BEV_results/video_results.npz --output=/home/yusun/BEV_results/dance.fbx --gender=female --subject_id=1
```
