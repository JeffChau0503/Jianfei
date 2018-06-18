# encoding=utf-8
# In[1]:
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import os

"""
数据路径:
E:/Datafile/PyriteRefl0614/DataPR0614.txt

Reflectance mode with 9 Pixels:
E:/Datafile/ZielRefl0613/DataCarbonates.txt
E:/Datafile/ZielRefl0613/DataGalena.txt
E:/Datafile/ZielRefl0613/DataPyrite.txt

Absorbance mode with 9 Pixels:
E:/Datafile/ZielAbso0614/DataCA.txt
E:/Datafile/ZielAbso0614/DataGA.txt
E:/Datafile/ZielAbso0614/DataPA.txt
"""

data_path = 'E:/Datafile/PyriteRefl0614/DataPR0614.txt'
# data为二维数组. 代码含义:把这些数据读取出来,存为numpy数组.该方法只适用于txt文件中是纯数字(float,int)
data = np.loadtxt(data_path)


m = data.shape[0]  # 获取样本数(行数).  shape = file.shape
n = data.shape[1]  # 获取特征数(列数).
print("shape of Data: " + str(data.shape))
print("number of examples: " + str(m))

type(data)
print(data.max(), ' ', data.min())

# build dir for save pics
type_dir = 'E:/imageset/Type1'
if os.path.exists(type_dir) == False:
    os.makedirs(type_dir)

for i in range(m):
    norm = (data[i, :] - data[i, :].min())/(data[i, :].max()-data[i, :].min())
    norm = 255 * norm
    norm = norm.astype(np.int32)
    r = int(np.sqrt(n))
    matrix = np.resize(norm, (r, r))
    save_path = os.path.join(type_dir, str(i+1)+'.jpg')
    scipy.misc.imsave(save_path, matrix)

print("End\n")


