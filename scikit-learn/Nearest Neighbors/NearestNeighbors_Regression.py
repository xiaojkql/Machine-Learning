print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors

# 用随机的方法生成数据
np.random.seed(50)
# 产生40个[0,5]之间均匀分布的数据
X = np.sort(5 * np.random.rand(40,1),axis=0) # (40,1) axis = 0给行排序

# 测试点数据
T = np.linspace(0,5,500)[:,np.newaxis] # 在哪个维度上增加一维 [1,2,3] -> [[1],[2],[3]]
y = np.sin(X).ravel() # 扁平化

# 给目标值增加噪声
y[::5] += (0.5 - np.random.rand(8)) # 以5为间隔取数据，共有40个数据，所以取出为8个数据

n_neighbors = 5
for weights in ['uniform','distance']:
    regf = neighbors.KNeighborsRegressor(n_neighbors=n_neighbors,weights=weights)
    regf.fit(X,y)
    test = regf.predict(T)
    plt.figure()
    plt.plot(T,test,c='g',label='prediction')
    plt.scatter(X,y,c='k',label='data')
    plt.axis('tight') # 设置坐标轴，或者使用plt.xlim() plt.ylim()
    plt.legend()
    plt.title('kNeighborsRegressor (k = %i, weights = %s)'%(n_neighbors,weights))

plt.tight_layout()
plt.show()