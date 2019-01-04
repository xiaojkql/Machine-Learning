'''
对于neighbor sklearn下的neighbor下有很多子类
NearestNeighbors
四个重要的 参数：
n_neighbors 近邻数量
radius
algorithm
metrics
'''

'''
unsupervised nearestNeighbors algorithm
'''

import numpy as np
from sklearn.neighbors import NearestNeighbors

X = np.array([[0,0,2],[1,0,0],[0,0,1]])
neigh = NearestNeighbors(2,0.4).fit(X)

print(neigh.kneighbors(np.array([0,0,1.3]).reshape(1,3),2))

# 返回多个参数时即可以赋值给一个，也可以分别进行赋值返回
a,b = neigh.kneighbors([[0,0,1.4]])
print(a)

print(neigh.kneighbors_graph(X))
# 转换为矩阵的形式进行显示 toarray()
print(neigh.kneighbors_graph(X).toarray())

samples = np.array([[0,0,0],[0,.5,0],[1,1,.5]])
neigh = NearestNeighbors(radius=0.1)
neigh.fit(samples)
rng = neigh.radius_neighbors([[1,1,1]])
print(rng)
print(rng[0][0])
print(neigh.radius_neighbors_graph(samples).toarray())

print('\n\n\n')
X = [[-1,-1],[-2,-1],[-3,-2],[1,1],[2,1],[3,2]]
neigh = NearestNeighbors(n_neighbors=2,algorithm='ball_tree').fit(X)
distance,indics = neigh.kneighbors(X)
print(distance)
print(indics)

# 进的点在index相邻，所以得到的此矩阵比较稀疏，
# 且在对角线上，对有些非监督学习算法是很有用的
neigh_graph = neigh.kneighbors_graph(X)
print(neigh_graph.toarray())
