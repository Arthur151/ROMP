# TRACE's code for inference & evaluation

To run the inference & evaluation, please first install simple-romp following [the guidance](../README.md).  

Please refer to [this instruction](../../trace/README.md) for training.

## Installation

CAUTIONS: simple-romp for ROMP and BEV has been tested on Linux, Mac, and Windows, but its TRACE version only works in Linu
x environment with CUDA support. **Please make sure that you have CUDA and corresponding pytorch installed before using simple-romp for TRACE.**

1. Installing simple_romp via pip:

```
pip install --upgrade setuptools numpy cython lap
```

```
pip install --upgrade simple_romp
```
or download the package and install it from source:
```
python setup.py install
```

2. Preparing SMPL model files in our format:

Firstly, please register and download:  
a. Meta data from [this link](https://github.com/Arthur151/ROMP/releases/download/V2.0/smpl_model_data.zip). Please unzip it, then we get a folder named "smpl_model_data"
b. SMPL model file (SMPL_NEUTRAL.pkl) from "Download version 1.1.0 for Python 2.7 (female/male/neutral, 300 shape PCs)" in [official website](https://smpl.is.tue.mpg.de/). Please unzip it and move the SMPL_NEUTRAL.pkl from extracted folder into the "smpl_model_data" folder.      
c. (Optional) If you use BEV, please also download SMIL model file (DOWNLOAD SMIL) from [official website](https://www.iosb.fraunhofer.de/en/competences/image-exploitation/object-recognition/sensor-networks/motion-analysis.html). Please unzip and put it into the "smpl_model_data" folder, so we have "smpl_model_data/smil/smil_web.pkl".   
Then we can get a folder in structure like this:  
```
|-- smpl_model_data
|   |-- SMPL_NEUTRAL.pkl
|   |-- J_regressor_extra.npy
|   |-- J_regressor_h36m.npy
|   |-- smpl_kid_template.npy
|   |-- smil
|   |-- |-- smil_web.pkl
```

Secondly, please convert the SMPL model files to our format via  
```
# please provide the absolute path of the "smpl_model_data" folder to the source_dir 
romp.prepare_smpl -source_dir=/path/to/smpl_model_data
# (Optional) If you use BEV, please also run:
bev.prepare_smil -source_dir=/path/to/smpl_model_data
```
The converted file would be save to "~/.romp/" in defualt. 

## Inference

After installation, please run 
```
trace2 -i /path/to/video_sequence
```

## Visualization 

To visualize the estimated results, please download the codes and run
```
cd simple_romp
python -m trace2.show
```

## TRACE Benchmark Evaluation

The evaluation code of TRACE is integrated into `simple-romp`, `trace2/eval.py`, `trace2/evaluation` folder.

To prepare the evaluation datasets, please refer to [trace_dataset.md](../../docs/trace_dataset.md).  

Please set the path to datasets in **dataset_dir of `simple-romp/trace2/eval.py`**, and then run:

### DynaCam
For details of DynaCam dataset, please refer to [[DynaCam Dataset]](https://github.com/Arthur151/DynaCam), which directly provides [predictions](https://github.com/Arthur151/DynaCam/releases/tag/predictions), and code for evaluation and visualization. 

You may also evaluate on DynaCam via running
```
cd simple_romp/trace2
python -m eval --eval_dataset=DynaCam
```

### MuPoTS
```
cd simple_romp/trace2
python -m eval --eval_dataset=mupots
```

### Dyna3DPW
```
cd simple_romp/trace2
python -m eval --eval_dataset=Dyna3DPW
```

### 3DPW
```
cd simple_romp/trace2
python -m eval --eval_dataset=3DPW
```