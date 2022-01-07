import h5py
import sys
import os
import cv2
import numpy as np
import glob
import pickle
import sys

subject_list = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
cam_dict = {'54138969': 0, '55011271': 1, '58860488': 2, '60457274': 3}

def extract_imgs(subject_id, src_folder, out_folder):
    video_files = glob.glob(os.path.join(src_folder, subject_id,  'Videos', '*.mp4'))
    for video_file in video_files:
        if "_ALL" in video_file:
            continue
        print("video_file", video_file)
        video_name = os.path.basename(video_file)
        action_name, cam_str, _ = video_name.split('.')
        cam_id = cam_dict[cam_str]
        target_name = os.path.join(out_folder,'{}_{}_{}'.format(subject_id, action_name, cam_id))
        print("target_name ", target_name)
        print("video_file", video_file)
        cap = cv2.VideoCapture(video_file)
        frame_dex = -1
        dex = 0
        frame_num = 0  #
        while (1):
            frame_dex = frame_dex + 1
            ret, frame = cap.read()
            if frame_dex % 5 != 0:
                continue
            if frame_dex == 0:
                continue
            if ret:
                cv2.imwrite(target_name + '_' + str(dex) + '.jpg', frame)
                print("target_name ", target_name + '_' + str(dex) + '.jpg')
                dex = dex + 1
                if dex > 20:
                    break
            else:
                print("video_file end", video_file)
                break
        cap.release()
    return 1


def main():
    assert len(sys.argv)==3, print('plese run the code : python h36m_extract_frames.py h36m_video_path image_save_path')
    # set the path to image folder (archives) of human3.6M dataset here
    src_folder = sys.argv[2] #"archives" # archives/S1/Videos/Directions 1.54138969.mp4 .....
    out_folder = sys.argv[3] #"images"
    os.makedirs(out_folder, exist_ok=True)
    for subject_id in subject_list:
        print('Processing {}'.format(subject_id))
        extract_imgs(subject_id, src_folder, out_folder)


if __name__ == '__main__':
    main()

