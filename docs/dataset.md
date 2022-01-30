## Dataset preparation

### Directory structure

1: You can download the parsed data from [google drive](https://drive.google.com/drive/folders/1_g4AbXumhufs7YPdTAK3kFMnTQJYs3w3?usp=sharing).  **Make sure you have signed the license agreement with the dataset publisher.**
Please follow the directory structure to organize them.
```
|-- dataset
|   |-- h36m
|   |   |-- images
|   |   |-- annots.npz
|   |   |-- cluster_results...
|   |-- mpi-inf-3dhp
|   |   |-- images
|   |   |-- annots.npz
|   |   |-- cluster_results...
|   |-- MuCo
|   |   |-- augmented_set
|   |   |-- annots_augmented.npz
|   |-- coco
|   |   |-- images
|   |   |   |-- train2014
|   |   |   |-- val2014
|   |   |   |-- test2014
|   |   |-- annots_train2014.npz
|   |   |-- annots_val2014.npz
|   |-- mpii
|   |   |-- images
|   |   |-- annot
|   |   |-- eft_annots.npz
|   |-- lsp
|   |   |-- hr-lspet
|   |   |   |-- eft_annots.npz
|   |-- crowdpose
|   |   |-- images
|   |   |-- annots_train.npz
|   |   |-- annots_val.npz
|   |   |-- annots_test.npz
|   |-- 3DPW
|   |   |-- imageFiles
|   |   |-- sequenceFiles
|   |   |-- vibe_db
|   |   |-- annots.npz
```
If you meet 'Download limit' problem from google drive, you can make a copy of the file to your personal google drive account to avoid this.

2: Download the images from the official websites, [COCO 2014 images](https://cocodataset.org/#download), [MPII](http://human-pose.mpi-inf.mpg.de/#download), [CrowdPose](https://drive.google.com/file/d/1VprytECcLtU4tKP32SYi_7oDRbw7yUTL/view), [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/license.html). Please rename the image folder to 'images'.  

(Optional) 3. If you download the original videos from the official website of [Human3.6M](http://vision.imar.ro/human3.6m/description.php), please extract the images via:
```
python ROMP/romp/lib/dataset/preprocess/h36m_extract_frames.py h36m_extract_frames.py path/to/h36m_video_folder path/to/image_save_folder
# e.g. if you have archives/S1/Videos/Directions 1.54138969.mp4, then run
python h36m_extract_frames.py archives images
```

Finally, pleaset set the dataset root path:  
If you put all datasets in one folder, then you just need to change [this config](https://github.com/Arthur151/ROMP/blob/db299277b519de0970604789b4490d9f10318764/romp/lib/config.py#L151) to the path of your dataset folder, like:
```
dataset_group.add_argument('--dataset_rootdir',type=str, default='/path/to/your/dataset/folder', help= 'root dir of all datasets')
```
If you put different dataset at different path, then you have to set them separately. For instance, to set the path of Human3.6M dataset, please change [this line](https://github.com/Arthur151/ROMP/blob/db299277b519de0970604789b4490d9f10318764/romp/lib/dataset/h36m.py#L10) to the path where you put Human3.6M, like
```
self.data_folder = /path/to/your/h36m/
```

### Test the data loading

We can test the data loading of a datasets, like lsp via 
```
cd ROMP
python -m romp.lib.dataset.lsp

```
Annotations will be drawed on the input image. The test results will be saved in ROMP/test/.
