# Evaluation Code for ECCV 2020 3DPW Workshop

This repository contains the evaluation code for the [ECCV 2020 workshop on 3D Pose Estimation in the Wild](https://virtualhumans.mpi-inf.mpg.de/3DPW_Challenge/)

### Getting Started
Please download the [3DPW dataset](https://virtualhumans.mpi-inf.mpg.de/3DPW/) and the create a new conda environment using the command 

`source ./scripts/install_prep.sh`

By executing the above script you implicitly agree to the [license agreement](https://virtualhumans.mpi-inf.mpg.de/3DPW/license.html) of the 3DPW dataset

Please download the [SMPL body model](https://smpl.is.tue.mpg.de) and place the pkl files in the directory `./input_dir/ref` 

Please place your estimated 3D Pose in the directory `./input_dir/res`. Please ensure that the data is in the same format as described below under the label **Results Directory Structure**

### Evaluation
Execute the evaluation script using 

`python evaluate.py ./input_dir ./output_dir`

### Metrics
Evaluation Criteria
The 3D human performance is evaluated according to these metrics:

1) **MPJPE** Mean Per Joint Position Error (in mm) It measures the average Euclidean distance from prediction to ground truth joint positions.  The evaluation adjusts the translation (tx,ty,tz) of the prediction to match the ground truth.

2) **MPJPE_PA**: Mean Per Joint Position Error (in mm) after procrustes analysis. (Rotation, translation and scale are adjusted).

3) **PCK**: percentage of correct joints. A joint is considered correct when it is less than 50mm away from the ground truth. The joints considered for PCK are: shoulders, elbows, wrists, hips, knees and ankles.

4) **AUC**: the total area under the PCK-threshold curve. Calculated by computing PCKs by varying from 0 to 200 mm the threshold at which a predicted joint is considered correct

5) **MPJAE**. It measures the angle in degrees between the predicted part orientation and the ground truth orientation. The orientation difference is measured as the geodesic distance in SO(3). The 9 parts considered are: left/right upper arm, left/right lower arm, left/right upper leg, left/right lower leg and root.

6) **MPJAE_PA**. It measures the angle in degrees between the predicted part orientation and the ground truth orientation after rotating all predicted orientations by the rotation matrix obtained from the procrustes matching step. 


### Results Directory Structure
The structure of the results directory `./input_dir/res` should mirror that of the ground-truth directory. The submission directory should have three sub-directories : train/validation/test. For each pickle file in the ground truth directory, there should be a pickle file in the submission directory - with exactly the same name. Each pickle file should contain a dictionary with two keys - 'jointPositions' and 'orientations'. 

'jointPositions': array of shape P x N x J x 3. This should contain the 3D joint location of each SMPL joint. The joint positions must be in meters.
'orientations':  array of shape P x N x K x 3 x 3. This array should contain the absolute rotation matrix in the global coordinate frame of K body parts. The rotation is the map from the local bone coordinate frame to the global one.
where,

P: The number of people tracked in the sequence. If there is just one tracked person, the size should be 1 x N x 24 x 3

N: The number of frames in the sequence.

J = 24 joints of SMPL --> If you do not use SMPL skeleton, you need to convert between formats (note that this is usually relatively straightforward)

K= 9 parts in the following order: root (JOINT 0) , left hip  (JOINT 1), right hip (JOINT 2), left knee (JOINT 4), right knee (JOINT 5), left shoulder (JOINT 16), right shoulder (JOINT 17), left elbow (JOINT 18), right elbow (JOINT 19).

When the number of detected 2D pose keypoints is less than six (indicated as zeros in the 3DPW data), we exclude the frame from evaluation -- you can fill that space with zeros or some other dummy number. No algorithm will be evaluated for those frames. The evaluation protocol also ignores the frames where the camera not been well aligned with the image -- these instances are labelled in valid_camposes array in the 3DPW annotated pkl files.

If you only want to evaluate jointPositions and not orientations, please make sure that your dictionaries ONLY contain the 'jointPositions' key. Similiarly please include ONLY the 'orientations' key in each dictionary if you only want to evaluate orientation.
