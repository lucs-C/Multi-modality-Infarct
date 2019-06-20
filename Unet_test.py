'''
2D Unet is used to segment medical images.

author ： ChengZhenfeng
'''
from __future__ import division
from unet.model_Infarct import unet2dModule
import numpy as np
import pandas as pd
from unet.function import readmat, get_filename
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf

dispaystep = 10
image_width = 256
image_hight = 256

def prediction():

    # Get test Image path from CSV file
    csvmaskdata = pd.read_csv('./Dataset/train/TrainMask.csv')
    csvimagedata = pd.read_csv('./Dataset/train/TrainImage.csv')
    maskdata = csvmaskdata.iloc[:, :].values
    imagedata = csvimagedata.iloc[:, :].values

    #-1. 模型分割的精度评估
    unet2d = unet2dModule(256, 256, channels = 2)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    sess.run(init)
    saver.restore(sess, "./model/Infarct100000.ckpt")

    for i in range(len(imagedata)):
        #-1. 分割精度评估
        test_images = readmat(imagedata[i][0], 'multimodalitydata')
        GT_image = readmat(maskdata[i][0], 'multimodalitymask')
        imagename = get_filename(imagedata[i][0])
        test_images = np.reshape(test_images, (1, test_images.shape[0], test_images.shape[1], 2))

        pred = sess.run(unet2d.Y_pred, feed_dict={unet2d.X: test_images,
                                                  unet2d.phase: 1,
                                                  unet2d.drop_conv: 1})
        predictvalue = np.reshape(pred, (test_images.shape[1], test_images.shape[2]))
        predictvalue = predictvalue.astype(np.float32) * 255.
        predictvalue = np.clip(predictvalue, 0, 255).astype('uint8')

        cv2.imwrite("./outresult/predmask/" + imagename + "_mask.bmp", predictvalue)
        print("The " + imagename + " has already save into ./outresult/predmask")


        #-2. 分割结果显示
        figure_ID = 0
        if i % dispaystep == 0:
            plt.figure(figure_ID)
            test_images = np.reshape(test_images, (test_images.shape[1], test_images.shape[2], 2))

            # 显示ADC
            plt.subplot(2,3,1)
            ADC = test_images[:,:,0]
            plt.imshow(ADC, cmap= 'gray')
            plt.xticks([])
            plt.yticks([])
            plt.title('ADC')

            # 显示DWI
            plt.subplot(2,3,2)
            DWI = test_images[:,:,1]
            plt.imshow(DWI, cmap= 'gray')
            plt.xticks([])
            plt.yticks([])
            plt.title('DWI')

            # 显示GT在DWI上的区域
            plt.subplot(2,3,3)
            plt.imshow(GT_image, cmap='binary')
            plt.xticks([])
            plt.yticks([])
            plt.title('GT')

            # 显示prediction
            plt.subplot(2,3,4)
            plt.imshow(predictvalue,cmap='binary')
            plt.xticks([])
            plt.yticks([])
            plt.title('predict')

            # GT 与 predict 的比较
            campresult = np.zeros((image_width, image_width))
            # convert from [0:255] => [0.0:1.0]
            predictvalue[predictvalue[:] >100] = 1

            # GT 与predict 的交集
            intersection = GT_image *predictvalue

            # GT - intersection_img
            Under_segmentation = GT_image - intersection

            # pred - intersection_img
            over_segmentation  = predictvalue - intersection

            campresult = np.zeros((image_width,image_hight,3))

            # campresult = 1*intersection + 2*Under_segmentation + 3*over_segmentation

            if np.sum(np.sum(intersection)) !=0:
                campresult[intersection[:]==1, 0] = 255
                campresult[intersection[:] == 1, 1] = 0
                campresult[intersection[:] == 1, 2] = 0
            if np.sum(np.sum(Under_segmentation)) !=0:
                campresult[Under_segmentation[:]==1, 0] = 0
                campresult[Under_segmentation[:] == 1, 1] = 255
                campresult[Under_segmentation[:] == 1, 2] = 0
            if np.sum(np.sum(over_segmentation)) !=0:
                campresult[over_segmentation[:]==1, 0] = 0
                campresult[over_segmentation[:] == 1, 1] = 0
                campresult[over_segmentation[:] == 1, 2] = 255

            plt.subplot(2, 3, 5)
            plt.imshow(campresult,)
            plt.xticks([])
            plt.yticks([])
            plt.title('compare result')

            plt.savefig('./outresult/camparationresult/' + imagename + '.png')

        # 对所有结果进行显示
        # plt.show()



if __name__ == "__main__":
    prediction()
