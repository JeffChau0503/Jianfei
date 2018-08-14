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


# 基于移动平均框(通过卷积)平滑曲线 该函数用于生成波形图
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    yhat = np.convolve(y, box, mode='same')
    return yhat


# 读取数据
data_path = 'E:/imageset/txt data/Sphalerit.txt'  # --------------------------------------------------更改1 文件读取路径
# data为二维数组. 代码含义:把这些数据读取出来,存为numpy数组.该方法只适用于txt文件中是纯数字(float,int)
data_org = np.loadtxt(data_path)

m = data_org.shape[0]  # 获取样本数(行数).  shape = file.shape
n = data_org.shape[1]  # 获取特征数(列数).
print("shape of Original data: " + str(data_org.shape))
print("number of Original examples: " + str(m))

type(data_org)
print("Maximum and minimum: " + str(data_org.max()), ' ', str(data_org.min()))

# 多光谱合并(多数组首尾相连)
step = 25
data = []  # 结果数组初始化

for i in range(int(m/step)):
    data.append([])
    for j in range(step):
        data[i] = np.concatenate((data[i], data_org[i*step+j]))

print("Now shape of Data: " + str(np.shape(data)))

'''
# 查看合并后的数据
for i in range(int(m/step)):
    for j in range(len(data[i])):
        print(data[i][j], end=' ')
    print('')
'''
array_data = np.array(data)  # list 转换为 array 以便接下来的切片操作。

# build dir for save pics
type_dir1 = 'E:/imageset/Connection method/Sphalerit grayscale'  # -------------------------------------------更改2 文件保存路径
type_dir2 = 'E:/imageset/Connection method/Sphalerit graph/'  # --------------------------更改3 文件保存路径 注意: 最后的"/"不能省
if os.path.exists(type_dir1) == False:
    os.makedirs(type_dir1)
if os.path.exists(type_dir2) == False:
    os.makedirs(type_dir2)

# 灰度图
for i in range(int(m / step)):
    norm = (array_data[i, :] - array_data[i, :].min())/(array_data[i, :].max()-array_data[i, :].min())
    norm = 255 * norm
    norm = norm.astype(np.int32)
    r = int(np.sqrt(n * step))
    matrix = np.resize(norm, (r, r))
    save_path = os.path.join(type_dir1, "7_" + str(i) + '.png')  # ---------------------------更改4 更改类即"_"之前的数字
    scipy.misc.imsave(save_path, matrix)

print("Grayscale End\n")

# 波形图
for i in range(int(m/step)):
    # 原横坐标为光谱信息（波长范围369nm 至 1108nm），现延长25倍。 纵坐标单位为百分比，来自data_path
    x = np.linspace(369.14, 1108.17 * 25, num=1082 * 25)  # ------------------------------------检查txt是否需要更改num变量
    y = array_data[i, :]
    plt.plot(x, smooth(y, 37), 'k', lw=0.5)
    plt.xticks([])  # 删掉刻度信息
    plt.yticks([])
    plt.axis('off')  # 删掉坐标轴
    plt.savefig(type_dir2 + "7_" + str(i) + '.png')  # --------------------------------------------更改5 更改类即"0_"数字
    plt.clf()

print("Waveform End\n")
