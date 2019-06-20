'''
2D Unet is used to segment medical images.

author ： ChengZhenfeng
'''
from __future__ import division
from unet.model_Infarct import unet2dModule
import numpy as np
import pandas as pd
import cv2

def train():
    '''
    Preprocessing for dataset
    '''
    # Read  data set (Train data from CSV file)
    csvmaskdata = pd.read_csv('./Dataset/train/TrainMask.csv')
    csvimagedata = pd.read_csv('./Dataset/train/TrainImage.csv')
    maskdata = csvmaskdata.iloc[:, :].values
    imagedata = csvimagedata.iloc[:, :].values
    # shuffle imagedata and maskdata together
    perm = np.arange(len(csvimagedata))
    np.random.shuffle(perm)
    imagedata = imagedata[perm]
    maskdata = maskdata[perm]

    unet2d = unet2dModule(256, 256, channels = 2, costname = "dice coefficient")
    unet2d.train(imagedata, maskdata, "./model/Infarct7000.ckpt",
                 "./log", 0.001, 0.8, 100000, 16)


if __name__ == "__main__":
    train()
