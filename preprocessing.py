# encoding=utf-8
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('E:/Datafile/Data0529.txt')  # file为二维数组. 代码含义:把这些数据读取出来,存为numpy数组.该方法只适用于txt文件中是纯数字(float,int)
m = data.shape[0]  # 获取样本数(行数).  shape = file.shape
n = data.shape[1]  # 获取特征数(列数).
print("shape of Data:" + str(data.shape))
print("number of examples:" + str(m))


def lbl_norm(data):
    """
    Layer by layer normalization 逐层归一化
    :param data:数据data为一个2维数组, 横向为光谱方向, 纵向为空间方向
    :return:归一化后的数据.
    """

    layermax = data.max(axis=0)
    layermin = data.min(axis=0)
    norm_x = (data - layermin)/(layermax - layermin)
    print(norm_x.shape)
    return norm_x


print(lbl_norm(data))

for pixel in lbl_norm(data):
    """
    对每一个像元提取光谱数据,reshape成二维矩阵,imwirte成灰度图片
    """
    r = int(np.sqrt(n))  # 确定二维矩阵的边长,开方取整
    matrix = np.resize(pixel, (r, r))  # reshape成二维矩阵
    print(matrix)
    plt.imshow(matrix, cmap='Greys_r')
    plt.show()

