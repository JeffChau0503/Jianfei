# encoding=utf-8
# In[1]:
import numpy as np
import scipy.misc
import os

data_path = 'E:/Datafile/P0629.txt'  # 更改1 文件读取路径
# data为二维数组. 代码含义:把这些数据读取出来,存为numpy数组.该方法只适用于txt文件中是纯数字(float,int)
data = np.loadtxt(data_path)


m = data.shape[0]  # 获取样本数(行数).  shape = file.shape
n = data.shape[1]  # 获取特征数(列数).
print("shape of Data: " + str(data.shape))
print("number of examples: " + str(m))

type(data)
print("Maximum and minimum: " + str(data.max()), ' ', str(data.min()))

# build dir for save pics
type_dir = 'E:/imageset/test0710'  # 更改2 文件保存路径
if os.path.exists(type_dir) == False:
    os.makedirs(type_dir)

for i in range(m):
    norm = (data[i, :] - data[i, :].min())/(data[i, :].max()-data[i, :].min())
    norm = 255 * norm
    norm = norm.astype(np.int32)
    r = int(np.sqrt(n))
    matrix = np.resize(norm, (r, r))
    save_path = os.path.join(type_dir, "0_" + str(i+1).zfill(5) + '.jpg')  # 更改3 文件名命名 下划线前表示标签
    scipy.misc.imsave(save_path, matrix)

print("End\n")
