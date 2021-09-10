## Dataset preparation

### Directory structure

1: You can download the parsed data from [google drive](https://drive.google.com/drive/folders/1Vm2Nqaz5Bon2Kblcg8DQguwa9ti1Tn6h?usp=sharing).  **Make sure you have signed the license agreement with the dataset publisher.**
Please follow the directory structure to organize them.
```
|-- dataset
|   |-- h36m
|   |   |-- images
|   |   |-- annots.npz
|   |-- mpi-inf-3dhp
|   |   |-- images
|   |   |-- annots.npz
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
|   |   |-- eft_annots.npz
|   |-- lsp
|   |   |-- hr-lspet
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

2: Download the images from the official websites, [COCO 2014 images](https://cocodataset.org/#download), [MPII](http://human-pose.mpi-inf.mpg.de/#download), [CrowdPose](https://drive.google.com/file/d/1VprytECcLtU4tKP32SYi_7oDRbw7yUTL/view)). Please rename the image folder to 'images'.  

(Optional) 3. If you download the original videos from the official website of [Human3.6M]([http://vision.imar.ro/human3.6m/description.php], please extract the images via:
```
python ROMP/romp/lib/dataset/preprocess/h36m_extract_frames.py h36m_extract_frames.py path/to/h36m_video_folder path/to/image_save_folder
# e.g. if you have archives/S1/Videos/Directions 1.54138969.mp4, then run
python h36m_extract_frames.py archives images
```

