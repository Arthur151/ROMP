import cv2
import numpy as np
import os, sys
import torch

def padding_image(image):
    h, w = image.shape[:2]
    side_length = max(h, w)
    pad_image = np.zeros((side_length, side_length, 3), dtype=np.uint8)
    top, left = int((side_length - h) // 2), int((side_length - w) // 2)
    bottom, right = int(top+h), int(left+w)
    pad_image[top:bottom, left:right] = image
    image_pad_info = torch.Tensor([top, bottom, left, right, h, w])
    return pad_image, image_pad_info
    
def img_preprocess(image, input_size=512):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pad_image, image_pad_info = padding_image(image)
    input_image = torch.from_numpy(cv2.resize(pad_image, (input_size,input_size), interpolation=cv2.INTER_CUBIC))[None].float()
    return input_image, image_pad_info