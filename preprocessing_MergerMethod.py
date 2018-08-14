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

# 多光谱合并(逐元素求和后平均)
step = 25
data = np.zeros((int(m/step), n))  # 结果数组初始化

for i in range(int(m/step)):  # 控制行
    for j in range(n):  # 控制列
        for k in range(step):  # 控制压缩
            data[i][j] += data_org[i*step+k][j]  # 压缩
data = np.divide(data, step)
print("shape of Data: " + str(data.shape))

'''
# 查看合并后的数据
for i in range(int(m/step)):
    for j in range(n):
        print(data[i][j], end=' ')
    print('\n')
'''

# build dir for save pics
type_dir1 = 'E:/imageset/Merger method/Sphalerit grayscale'  # ------------------------------------------------------更改2 文件保存路径
type_dir2 = 'E:/imageset/Merger method/Sphalerit graph/'  # -------------------------------------更改3 文件保存路径 注意: 最后的"/"不能省
if os.path.exists(type_dir1) == False:
    os.makedirs(type_dir1)
if os.path.exists(type_dir2) == False:
    os.makedirs(type_dir2)

# 灰度图
for i in range(int(m/step)):
    norm = (data[i, :] - data[i, :].min())/(data[i, :].max()-data[i, :].min())
    norm = 255 * norm
    norm = norm.astype(np.int32)
    r = int(np.sqrt(n))
    matrix = np.resize(norm, (r, r))
    '''
    所有的图片大小必须一致，而且必须是灰度图像，  
    mnist数据集图片命名规则：0_00001.jpg：0表示对应图片的内容，   即标签；
    00001表示标签为0的图片中第1张图片，00002为第2张图片，以此类推........
    '''
    save_path = os.path.join(type_dir1, "7_" + str(i) + '.png')  # -----------------------更改4 更改类即"_"之前的数字
    scipy.misc.imsave(save_path, matrix)

print("Grayscale End\n")

# 波形图
for i in range(int(m/step)):
    # xy坐标归一化。原横坐标为光谱信息（波长范围369nm 至 1108nm）。 纵坐标单位为百分比，来自data_path
    x = np.linspace(369.14 / 1108.17, 1, num=1082)  # ----------------------------------------------检查txt是否需要更改
    y = data[i, :]
    plt.plot(x, smooth(y, 37), 'k', lw=0.5)
    plt.xticks([])  # 删掉刻度信息
    plt.yticks([])
    plt.axis('off')  # 删掉坐标轴
    plt.savefig(type_dir2 + "7_" + str(i) + '.png')  # -----------------------------------更改5 更改类即"0_"数字
    plt.clf()

print("Waveform End\n")
