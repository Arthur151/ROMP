## Benchmark Evaluation

### Evaluation on 3DPW Challenge

1. Set paths
a. change the dataset_rootdir in configs/eval_3dpw_challenge.yml to the absolute path of 3DPW datasets.
b. change the gmodel_path in configs/eval_3dpw_challenge.yml to the absolute path of model checkpoint.
c. change the output_dir in configs/eval_3dpw_challenge.yml to the absolute path of saving the results.zip file.

2. Run:
```bash
cd ROMP/src
CUDA_VISIBLE_DEVICES=0 python lib/evaluation/collect_3DPW_results.py --gpu=0 --configs_yml=configs/eval_3dpw_challenge.yml
```

3. Resutls of model `ROMP_hrnet32.pkl` on different device:

On Tesla P40/ V100 GPU (Centos, Pytorch 1.6.0):  
+-----------+-------+----------+-------+-------+-------+----------+  
|   DS/EM   | MPJPE | PA_MPJPE |  PCK  |  AUC  | MPJAE | PA_MPJAE | 
+-----------+-------+----------+-------+-------+-------+----------+  
| pw3d_chal |  82.7 |   60.5   |  36.5 |  59.7 |  20.5 |   18.9   |  
+-----------+-------+----------+-------+-------+-------+----------+   
 On a GTX 1070Ti GPU (Ubuntu, Pytorch 1.6.0):  
+-----------+-------+----------+-------+-------+-------+----------+  
|   DS/EM   | MPJPE | PA_MPJPE |  PCK  |  AUC  | MPJAE | PA_MPJAE |  
+-----------+-------+----------+-------+-------+-------+----------+  
| pw3d_chal |  81.8 |   58.6   |  37.3 |  59.9 |  20.8 |   19.1   |  
+-----------+-------+----------+-------+-------+-------+----------+  

### Evaluation on 3DPW test set

1. Set paths
a. change the dataset_rootdir in configs/eval_3dpw_test.yml to the absolute path of 3DPW datasets.
b. change the gmodel_path in configs/eval_3dpw_test.yml to the absolute path of model checkpoint.
c. change the annot_dir in configs/eval_3dpw_test.yml to the absolute path of [vibe_db](https://drive.google.com/file/d/1_urpBQbboQnbQ1ieuoougBXBmqTrMdK5/view?usp=sharing) where 3dpw_test_db.pt located at.

2. Run:
```bash
cd ROMP/src
CUDA_VISIBLE_DEVICES=0 python core/benchmarks_evaluation.py --gpu=0 --configs_yml=configs/eval_3dpw_test.yml
```

3. Resutls on different device:

On Tesla P40/ V100 GPU (Centos, Pytorch 1.6.0):  
+-----------+-------+----------+--------+  
|   DS/EM   | MPJPE | PA_MPJPE |  PVE   |  
+-----------+-------+----------+--------+  
| pw3d_vibe | 85.48 |  53.14   | 103.02 |  
+-----------+-------+----------+--------+  

On a GTX 1070Ti GPU (Ubuntu, Pytorch 1.6.0):  
+-----------+-------+----------+  
|   DS/EM   | MPJPE | PA_MPJPE |  
+-----------+-------+----------+  
| pw3d_vibe | 87.10 |  53.11   |  
+-----------+-------+----------+  

### Test FPS

To test FPS of ROMP on your devices, please set configs/single_image.yml as below

```bash
 save_visualization_on_img: False
 demo_image_folder: '../demo/videos/Messi_1'
```
then run 

```bash
cd ROMP/src
CUDA_VISIBLE_DEVICES=0 python core/test.py --gpu=0 --configs_yml=configs/single_image.yml
```

On a GTX 1070Ti GPU (Ubuntu, Pytorch 1.6.0):  
+-----------+----------+-----------+  
|  Backbone | HRNet-32 | ResNet-50 |  
+-----------+----------+-----------+  
|    FPS    |   20.8   |    30.9   |  
+-----------+----------+-----------+  
