# make mp4 video using the generated frame results.
import os
import cv2
import glob
import numpy as np

shape = [1024,1024]
root_dir = '~/ROMP/demo/videos/'
out_dir = '~/ROMP/demo/'
fold_name = 'Messi_1'
def grub_imges_demo(fold_name):
    imgs_path_demo = os.path.join(root_dir,'{}_results'.format(fold_name))
    imgs = glob.glob(os.path.join(imgs_path_demo, '{}-image*'.format(fold_name)))
    orders = []
    for img in imgs:
        orders.append(int(os.path.basename(img).replace('{}-image'.format(fold_name),'').replace('.jpg','')))
    orders = np.array(orders)
    sorted_orders = orders[np.argsort(orders)]
    imgs_sorted = []
    for idx in sorted_orders:
        imgs_sorted.append(os.path.join(imgs_path_demo, '{}-image{}.jpg'.format(fold_name,idx)))
    return imgs_sorted
def make_mp4_demo(images,name):
    num = len(images)
    print(name, 'length', num)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_movie = cv2.VideoWriter(name+'.mp4', fourcc, 30, (shape[0], shape[1]))
    for i in range(num):
        if i%100==0:
            print('Writing frame: ',i,'/',num)
        frame = cv2.imread(images[i])
        output_movie.write(frame)
imgs = grub_imges_demo(fold_name)
make_mp4_demo(imgs,os.path.join(out_dir,fold_name+'results'))
