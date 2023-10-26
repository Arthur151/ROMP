## Detailed explanation: 

The bug was caused by [this part](https://github.com/Arthur151/ROMP/blob/e1268164f41a701e81d3d5f14ff660e44f1fe02a/simple_romp/trace2/evaluation/eval_3DPW.py#L366):

`if S1.shape[0] != 3 and S1.shape[0] != 2:  
     S1 = S1.permute(0,2,1)`

Because the code for calculating PAMPJPE requires the input matrix of shape (N, 3, 14).   
So the second line of this part, "S1 = S1.permute(0,2,1)", is to transform the input shape of S1 from (N, 14, 3) to (N, 3, 14).  
In TRACE, N is the subject number in each frame. 

But it will only be executed when the first line, the “if” judgment statement, is established.
The bug is in this “if” judgment statement:  
`if S1.shape[0] != 3 and S1.shape[0] != 2:`  
is trying to tell  
"if the 0-dim shape of matrix S1, which is N, is not equal to 3 and 2"

This wrong “if” judgment would be false when there are two subjects in the frame. 
So when there are two-person annotations, the input's shape for calculating PAMPJPE would be (N, 14, 3), instead of (N, 3, 14) as required. 
That's why sometimes the Rotation matrix calculated for PAMPJPE is of shape (14, 14), instead of (3, 3)

But when there is only a single subject in the frame, the code would be executed normally, because that "if" judgment still stands. 
So the results for single-person images are right, but results for two-person images are wrong. 
Note that 3DPW only annotates a maximum of two people in each frame. 

After correction, the results on 3DPW are PAMPJPE 50.8, MPJPE 80.3, PVE 98.1.

### Source: 
This part of the code was directly borrowed from VIBE with no change.
But in VIBE, PAMPJPE is evaluated for the whole sequence, N is the frame number. 
So as long as the frame number is not equal to 2 or 3, this would be always fine. 
But this bug would be activated during calculating per-frame PAMPJPE. 

Like what this developer commented at the source of this code, 
https://gist.github.com/mkocabas/54ea2ff3b03260e3fedf8ad22536f427?permalink_comment_id=3987499#gistcomment-3987499
The right code should be :  
`if S1.shape[1] != 3 and S1.shape[1] != 2:`  
, instead of   
`if S1.shape[0] != 3 and S1.shape[0] != 2:`  
So I fixed it in this way. But maybe we should directly change to the correct input shape.

### Influence: 
There are still many people still using the same code with this hidden bug.  
Please warn them when you find it.  
I found 100+ repositories in Github are still using the exactly same function, same error, including:
https://github.com/OpenMotionLab/MotionGPT/blob/0499f16df4ddde44dfd72a7cbd7bd615af1b1a94/mGPT/metrics/utils.py#L275
https://github.com/JimmyZou/EventHPE/blob/2af451f08205e442d92aa6bcd7c47285a59c5445/event_pose_estimation/geometry.py#L278
https://github.com/Jeff-sjtu/DnD/blob/934601bc5b9061a4fa98678e95bc42cd8acb0a61/dnd/utils/eval_utils.py#L342
https://github.com/NVlabs/GLAMR/blob/cc752f185c9f7e83b6dbf027f0b534d61791f8e7/lib/utils/torch_transform.py#L299
https://github.com/BoyanJIANG/H4D/blob/0069f64b2edaa348a7ec0af14b5cdd1f64c193d2/lib/utils/eval_utils.py#L213
https://github.com/cure-lab/DeciWatch/blob/5afe2d67005441e2785a8d374d31a91f046a0da6/lib/utils/eval_metrics.py#L16
https://github.com/stdrr/motion-latent-diffusion/blob/0ee43a47ecc7828c32b0c28f3d0c3bc2f9d5028b/mld/models/metrics/utils.py#L275

We sincerely apologize for this error.