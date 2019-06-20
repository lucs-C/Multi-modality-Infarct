import scipy.io as sio
import numpy as np
import os

# 下面是讲解python怎么读取.mat文件以及怎么处理得到的结果
def readmat(str_path, str_variable):
    load_data = sio.loadmat(str_path)
    data_out = load_data[str_variable]

    return data_out

def get_filename(file_path):
    filepath, tempfilename = os.path.split(file_path)
    filename, extension = os.path.splitext(tempfilename)

    return filename
