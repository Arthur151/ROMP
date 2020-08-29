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

###### val_batch_size: batch size 

###### nw: the number of dataloader workers. 

###### model_precision: mix precision

If you installed Pytorch 1.6 or upper verion, then you can use the automatic mix precision (AMP), by setting
```bash
model_precision: fp16
```
