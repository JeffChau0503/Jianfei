# encoding=utf-8
import numpy as np

data = np.loadtxt('E:/Datafile/Data0529.txt')  # file为二维数组. 代码含义:把这些数据读取出来,存为numpy数组.该方法只适用于txt文件中是纯数字(float,int)
m = data.shape[0]  # 获取样本数（行数）.    shape = file.shape
print("shape of Data:" + str(data.shape))
print("number of examples:" + str(m))


def lbl_norm(data):  # Layer by layer normalization 逐层归一化
    layermax = data.max(axis=0)
    layermin = data.min(axis=0)
    norm_x = (data - layermin)/(layermax - layermin)
    print(layermax)
    print(layermin)
    print(norm_x.shape)

    return norm_x


print(lbl_norm(data))
