# encoding=utf-8
"""
注意！！
检查每个生成文件夹下
生成的第一张照片"0_00000" 要删掉，因为该照片是由 data数据第一行（波长信息）转化而来的
"""

import numpy as np
import scipy.misc
import os
import matplotlib.pyplot as plt
import scipy.signal as spy


#  基于移动平均框(通过卷积)平滑曲线 该函数用于生成波形图
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    yhat = np.convolve(y, box, mode='same')
    return yhat


data_path = 'E:/Datafile/0716data/pyrite0717.txt'  # --------------------------------------------------更改1 文件读取路径
# data为二维数组. 代码含义:把这些数据读取出来,存为numpy数组.该方法只适用于txt文件中是纯数字(float,int)
data = np.loadtxt(data_path)


m = data.shape[0]  # 获取样本数(行数).  shape = file.shape
n = data.shape[1]  # 获取特征数(列数).
print("shape of Data: " + str(data.shape))
print("number of examples: " + str(m))

type(data)
print("Maximum and minimum: " + str(data.max()), ' ', str(data.min()))

# build dir for save pics
type_dir1 = 'E:/imageset/pyrite0717'      # -----------------------------------------------------------更改2 文件保存路径
type_dir2 = 'E:/imageset/pyrite0717pic/'  # ---------------------------------------更改3 文件保存路径 注意: 最后的"/"不能省
if os.path.exists(type_dir1) == False:
    os.makedirs(type_dir1)
if os.path.exists(type_dir2) == False:
    os.makedirs(type_dir2)

# 灰度图
for i in range(m):
    norm = (data[i, :] - data[i, :].min())/(data[i, :].max()-data[i, :].min())
    norm = 255 * norm
    norm = norm.astype(np.int32)
    r = int(np.sqrt(n))
    matrix = np.resize(norm, (r, r))
    '''
    所有的图片大小必须一致，而且必须是灰度图像，
    mnist数据集图片命名规则：0_00001.jpg：0表示对应图片的内容，即标签；
    00001表示标签为0的图片中第1张图片，00002为第2张图片，以此类推........
    '''
    save_path = os.path.join(type_dir1, "0_" + str(i) + '.png')  # -----------------------更改4 更改类即"0_"数字
    scipy.misc.imsave(save_path, matrix)

print("Grayscale End\n")

# 波形图
for i in range(m):
    x = np.linspace(369.14, 1108.84, num=1082)  # ----------------------------------------------------检查txt是否需要更改
    y = data[i, :] * 100
    plt.plot(x, smooth(y, 37), 'k', lw=0.5)
    plt.xticks([])  # 删掉刻度信息
    plt.yticks([])
    plt.axis('off')  # 删掉坐标轴
    plt.savefig(type_dir2 + "0_" + str(i) + '.png')  # -----------------------------------更改5 更改类即"0_"数字
    plt.clf()

print("Waveform End\n")
