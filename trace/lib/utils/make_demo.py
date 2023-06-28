import cv2
import numpy as np 
import torch
import os
import glob
import sys,os
from PIL import Image
sys.path.append(os.path.abspath(__file__).replace('utils/make_demo.py',''))
import config

shape = [1024,1024-200]

def grub_imges_demo(fold_name):
    imgs_path_demo = os.path.join('/home/yusun/datasets/demo_image','{}_results'.format(fold_name)) # ,'demo
    imgs = glob.glob(os.path.join(imgs_path_demo, '{}-image*'.format(fold_name)))
    orders = []
    for img in imgs:
        orders.append(int(os.path.basename(img).replace(fold_name+'-image','').replace('.jpg','')))
    orders = np.array(orders)
    sorted_orders = orders[np.argsort(orders)]
    imgs_sorted = []
    for idx in sorted_orders:
        imgs_sorted.append(os.path.join(imgs_path_demo, '{}-image{:03}.jpg'.format(fold_name,idx)))
    return imgs_sorted

def make_mp4_demo(images,name):
    num = len(images)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_movie = cv2.VideoWriter(name+'.mp4', fourcc, 30, (shape[0], shape[1]))
    for i in range(num):
        if i%100==0:
            print('Writing frame: ',i,'/',num)
        output_movie.write(cv2.imread(images[i])[100:-100])

def make_gif_demo(images,name):
    num = len(images)
    images_list = []
    for i in range(num):
        images_list.append(Image.fromarray(cv2.imread(images[i])[100:-100][:,:,::-1]))
    images_list[0].save(name+'.gif',save_all=True, append_images=images_list[1:], duration=40, loop=0) #,allow_mixed=False, minimize_size=False

def main():
    all_images = []
    for video_idx in [1,36,3,23,37,29]:
        #if video_idx in [12,13,14,18,19,20,28,32,33]:
        #    continue
        fold_name = 'c{}'.format(video_idx)
        imgs = grub_imges_demo(fold_name)
        make_gif_demo(imgs,os.path.join('/home/yusun/datasets/demo_image',fold_name+'results'))
        all_images+=imgs
    #all_images+= grub_imges_demo('c20')
    #make_mp4_demo(all_images,os.path.join(config.data_dir,'demo',fold_name+'results'))
    #make_gif_demo(all_images,os.path.join('/home/yusun/datasets/demo_image',fold_name+'results'))#config.data_dir,'demo'
    #make_mp4_demo(imgs,os.path.join(out_dir,action_name))


if __name__ == '__main__':
    main()

'''
imgs_path = os.path.join(config.data_dir,'demo/results_out/results_3dpw')
video_size = [1920,1080]
video_size_resize = 512./max(video_size) * video_size
shape = [1024,512]
action_name = 'outdoors_fencing_01' # 'downtown_runForBus_00-image_' # outdoors_slalom_01 downtown_runForBus_00 courtyard_arguing_00
out_dir = os.path.join(config.data_dir,'demo/results_out/results_videos')
os.makedirs(out_dir,exist_ok=True)

vibe_imgs_path = os.path.join('~','VIBE','output')
vibe_imgs_path = os.path.join('..', action_name+'_output')

def grub_imges():
    imgs = glob.glob(os.path.join(imgs_path, '{}-image_*'.format(action_name)))
    orders = []
    for img in imgs:
        orders.append(int(os.path.basename(img).replace(action_name+'-image_','').replace('.jpg','')))
    orders = np.array(orders)
    sorted_orders = np.argsort(orders)
    return sorted_orders
    imgs_sorted = []
    for idx in sorted_orders:
        imgs_sorted.append(os.path.join(imgs_path, '{}-image_{:05}.jpg'.format(action_name,idx)))
    return imgs_sorted

def grub_imges_vibe():
    imgs = glob.glob(os.path.join(vibe_imgs_path, '*.png'))
    orders = []
    for img in imgs:
        orders.append(int(os.path.basename(img).replace('image_','').replace('.png','')))
    orders = np.array(orders)
    sorted_orders = np.argsort(orders)
    return sorted_orders
    imgs_sorted = []
    for idx in sorted_orders:
        imgs_sorted.append(imgs[idx])
    return imgs_sorted

def make_mp4(images,name):
    num = len(images)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_movie = cv2.VideoWriter(name+'.mp4', fourcc, 30, (shape[0], shape[1]))
    for i in range(num):
        if i%100==0:
            print('Writing frame: ',i,'/',num)
        output_movie.write(cv2.imread(images[i]))

def make_up_frames():
    results_ids = grub_imges()
    results_vibe_ids = grub_imges_vibe()
    all_ids = set(np.concatenate([results_ids, results_vibe_ids]).tolist())
    matched_ids = []
    for idx in all_ids:
        if idx in results_ids and idx in results_vibe_ids:
            matched_ids.append(idx)
    sorted_matched_ids = np.argsort(np.array(matched_ids))
    frames = []
    for matched_id in sorted_matched_ids:
        frames = cv2.imread(os.path.join(imgs_path, '{}-image_{:05}.jpg'.format(action_name,matched_id)))
        vibe_results = cv2.resize(cv2.imread(os.path.join(vibe_imgs_path, 'image_{:05}.jpg'.format(matched_id))), ())
'''