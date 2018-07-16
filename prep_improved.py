# encoding=utf-8
# In[1]:
import numpy as np
import scipy.misc
import os
import matplotlib.pyplot as plt

data_path = 'E:/Datafile/0716data/calcite0716.txt'  # 更改1 文件读取路径
# data为二维数组. 代码含义:把这些数据读取出来,存为numpy数组.该方法只适用于txt文件中是纯数字(float,int)
data = np.loadtxt(data_path)


m = data.shape[0]  # 获取样本数(行数).  shape = file.shape
n = data.shape[1]  # 获取特征数(列数).
print("shape of Data: " + str(data.shape))
print("number of examples: " + str(m))

type(data)
print("Maximum and minimum: " + str(data.max()), ' ', str(data.min()))

# build dir for save pics
type_dir1 = 'E:/imageset/calcite0716'  # 更改2 文件保存路径
if os.path.exists(type_dir1) == False:
    os.makedirs(type_dir1)
type_dir2 = 'E:/imageset/calcite0716pic'
if os.path.exists(type_dir2) == False:
    os.makedirs(type_dir2)

for i in range(m):
    # 灰度图
    i += 1  # txt 第一行数据为wavelength数据 因此要跳过
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
    save_path = os.path.join(type_dir1, "0_" + str(i-1).zfill(5) + '.jpg')  # 更改3 文件名命名
    scipy.misc.imsave(save_path, matrix)

    # 波形图
    x = np.linspace(369.14, 1108.84, num=1082)
    y = data[i, :] * 10
    plt.plot(x, y, 'k', lw=0.5)
    plt.xticks([])  # 删掉刻度信息
    plt.yticks([])
    plt.axis('off')  # 删掉坐标轴
    save_path = os.path.join(type_dir2, "0_" + str(i - 1).zfill(5) + '.jpg')  # 更改3 文件名命名
    plt.savefig(type_dir2, format='png', dpi=300)
    plt.clf()


print("End\n")
