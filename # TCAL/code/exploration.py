# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 19:34:40 2018

@author: Franc
"""

import os
os.chdir('C:/Users/Franc/Desktop/Dir/Competitions/# TCAL')
         
import pandas as pd
import numpy as np

from cv2 import imread
from skimage import io
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
# ====== Imread ======

#PIL.Image.open + numpy 
#scipy.misc.imread 
#scipy.ndimage.imread
#
#这些方法都是通过调用PIL.Image.open 读取图像的信息； 
#PIL.Image.open 不直接返回numpy对象，可以用numpy提供的函数进行转换，参考Image和Ndarray互相转换； 
#其他模块都直接返回numpy.ndarray对象，通道顺序为RGB，通道值得默认范围为0-255。
#
#matplot.image.imread
#从名字中可以看出这个模块是具有matlab风格的，直接返回numpy.ndarray格式通道顺序是RGB，通道值默认范围0-255。
#
#cv2.imread
#使用opencv读取图像，直接返回numpy.ndarray 对象，通道顺序为BGR ，注意是BGR，通道值默认范围0-255。
#
#skimage.io.imread: 直接返回numpy.ndarray 对象，通道顺序为RGB，通道值默认范围0-255。 
#caffe.io.load_image: 没有调用默认的skimage.io.imread，返回值为0-1的float型数据，通道顺序为RGB

filename_list = sorted(os.listdir('data/train'))
filepath_list = glob('data\\train\\*')
plt.imshow(Image.open(filepath_list[0]))
a = np.array(Image.open(filepath_list[0]))
io.imread(filepath_list[0]).shape
img
