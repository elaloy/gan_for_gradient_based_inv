# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 11:54:17 2018

@author: elaloy
"""
import os
import numpy as np
from PIL import Image
from PIL.Image import FLIP_LEFT_RIGHT

def image_to_tensor(img):
#    tensor = np.array(img).transpose( (2,0,1) )
#    tensor = tensor / 128. - 1.
    i_array=np.array(img)
    if len(i_array.shape)==2:
        i_array=i_array.reshape((i_array.shape[0],i_array.shape[1],1))
    tensor = i_array.transpose( (2,0,1) )
    tensor = tensor / 128. - 1.0
    return tensor


def tensor_to_2Dimage(tensor):
    img = np.array(tensor).transpose( (1,2,0) )
    img = (img + 1.) * 128.
    return np.uint8(img)
    

def get_texture2D_iter(folder, npx=128, npy=128,batch_size=64, \
                     filter=None, mirror=False, n_channel=1):
    HW1    = npx
    HW2    = npy
    imTex = []
    files = os.listdir(folder)
    for f in files:
        name = folder + f
        try:
            img = Image.open(name)
            imTex += [image_to_tensor(img)]
            if mirror:
                img = img.transpose(FLIP_LEFT_RIGHT)
                imTex += [image_to_tensor(img)]
        except:
            print("Image ", name, " failed to load!")

    while True:
        data=np.zeros((batch_size,n_channel,npx,npx))                   
        for i in range(batch_size):
            ir = np.random.randint(len(imTex))
            imgBig = imTex[ir]
            if HW1 < imgBig.shape[1] and HW2 < imgBig.shape[2]:
                h = np.random.randint(imgBig.shape[1] - HW1)
                w = np.random.randint(imgBig.shape[2] - HW2)
                img = imgBig[:, h:h + HW1, w:w + HW2]
            else:                                               
                img = imgBig
            data[i] = img

        yield data
        

def save_tensor2D(tensor, filename):
    img = tensor_to_2Dimage(tensor)   
    if img.shape[2]==1:
        img=img.reshape((img.shape[0],img.shape[1]))
        img = Image.fromarray(img).convert('L')
    else:
        img = Image.fromarray(img)
    img.save(filename)
    

def zx_to_npx(zx, depth):
    return (zx-1)*2**depth + 1
    