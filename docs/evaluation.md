## Benchmark Evaluation

### Evaluation on 3DPW Challenge

This evaluation has been implemented in our released version 1.0, while the latest version 1.1 still need to debug.

2 steps to re-implement our results in Tab. 1 of the main paper. 

1. Set paths:  
a. change the dataset_rootdir in configs/eval_3dpw_challenge.yml to the absolute path of the folder that contains 3DPW dataset.  
b. change the model_path in configs/eval_3dpw_challenge.yml to the absolute path of model checkpoint.  
c. change the output_dir in configs/eval_3dpw_challenge.yml to the absolute path of saving the results.zip file.  

2. Run:
```bash
cd ROMP
CUDA_VISIBLE_DEVICES=0 python romp/lib/evaluation/collect_3DPW_results.py --configs_yml=configs/eval_3dpw_challenge.yml
```

3. Results of ROMP (HRNet-32) on different device:  
The model is trained in mixed precision (fp16) mode.

On Tesla P40/ V100 GPU (Centos, Pytorch 1.6.0):  

|   DS/EM   | MPJPE | PA_MPJPE |  PCK  |  AUC  | MPJAE | PA_MPJAE |  
|:---------:|:-----:|:--------:|:-----:|:-----:|:-----:|:--------:|  
| pw3d_chal |  82.7 |   60.5   |  36.5 |  59.7 |  20.5 |   18.9   |  
 
 On a GTX 1070Ti GPU (Ubuntu, Pytorch 1.6.0):  

|   DS/EM   | MPJPE | PA_MPJPE |  PCK  |  AUC  | MPJAE | PA_MPJAE |  
|:---------:|:-----:|:--------:|:-----:|:-----:|:-----:|:--------:|  
| pw3d_chal |  81.8 |   58.6   |  37.3 |  59.9 |  20.8 |   19.1   |  


### Evaluation on 3DPW test set
2 steps to re-implement our results in Tab. 2 and Tab. 3 of the main paper. 

1. Set paths:  
a. change the dataset_rootdir in configs/eval_3dpw_test.yml to the absolute path of 3DPW datasets.  
b. change the model_path in configs/eval_3dpw_test.yml to the absolute path of model checkpoint.  

2. Run:
```bash
cd ROMP
# to evaluate the model taking HRNet-32 as backbone without fine-tunning on 3DPW, please run
python -m romp.test --configs_yml=configs/eval_3dpw_test.yml
# to evaluate the model taking ResNet-50 as backbone without fine-tunning on 3DPW, please run
python -m romp.test --configs_yml=configs/eval_3dpw_test_resnet.yml
# to evaluate the model taking HRNet-32 as backbone with fine-tunning on 3DPW, please run
python -m romp.test --configs_yml=configs/eval_3dpw_test_ft.yml
# to evaluate the model taking ResNet-50 as backbone without fine-tunning on 3DPW, please run
python -m romp.test --configs_yml=configs/eval_3dpw_test_resnet_ft.yml
```

Results of ROMP (HRNet-32) on different device:

On Tesla P40/ V100 GPU (Centos, Pytorch 1.6.0):  

|   DS/EM   | MPJPE | PA_MPJPE |  PVE   |  
|:---------:|:-----:|:--------:|:------:|  
| pw3d_vibe | 85.48 |  53.14   | 103.02 |  


On a GTX 1070Ti GPU (Ubuntu, Pytorch 1.6.0):  

|   DS/EM   | MPJPE | PA_MPJPE |  
|:---------:|:-----:|:--------:|  
| pw3d_vibe | 87.10 |  53.11   |  


### Evaluation on CMU Panoptic
1 step to re-implement our results in Tab. 5 of the main paper. 
```bash
cd ROMP
python -m romp.test --configs_yml=configs/eval_cmu_panoptic.yml
```

### Evaluation on Crowdpose test/val set
2 steps to re-implement our results in Tab. 6 of the main paper. 
1. Install the official evaluation toolkit.
```bash
# if you didn't install the evaluation code of crowdpose, then install it via
cd ROMP/romp/lib/evaluation/crowdpose-api/PythonAPI
python setup.py install
```
2. Evaluation on the test/val set of crowdpose.
```bash
cd ROMP
# to generate the predictions on the test set of crowdpose. 
python -m romp.test --configs_yml=configs/eval_crowdpose_test.yml
# to generate the predictions on the val set of crowdpose. 
python -m romp.test --configs_yml=configs/eval_crowdpose_val.yml
```

### Test FPS
To re-implement our results in Tab. 7 of the main paper. Please try our webcam demo.  
On a GTX 1070Ti GPU (Ubuntu, Pytorch 1.9.0, CUDA 10.2):  

|  Backbone | HRNet-32 | ResNet-50 |  
|:---------:|:--------:|:---------:|  
|    FPS    |   23.8   |    30.9   |  


