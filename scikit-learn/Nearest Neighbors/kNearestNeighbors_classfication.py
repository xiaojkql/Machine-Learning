# kNearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors,datasets

n_neighbors = 15

iris = datasets.load_iris() # 三个类别，特征为4维特征
X,y = iris.data[:,:2],iris.target # 只用两维特征方便可视化

# 网格大小
h = 0.02
cmap_light = ListedColormap(['#FFAAAA','#AAFFAA','#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

for weights in ['uniform','distance']:
    clf = neighbors.KNeighborsClassifier(n_neighbors,weights=weights)
    clf.fit(X,y)

    # 生成网格数据
    x_min,x_max = X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max = X[:,1].min()-1,X[:,1].max()+1

    xx,yy = np.meshgrid(np.arange(x_min,x_max,h),
                        np.arange(y_min,y_max,h)) # xx 与 yy 都是矩阵


    z = clf.predict(np.c_[xx.ravel(),yy.ravel()])
    print(clf.score(X,y))
    z = z.reshape(xx.shape)

    plt.figure()
    plt.pcolormesh(xx,yy,z,cmap=cmap_light)
    plt.scatter(X[:,0],X[:,1],c=y,cmap=cmap_bold,
                edgecolors='k',s=30)
    plt.xlim(xx.min(),xx.max())
    plt.ylim(yy.min(),yy.max())
    plt.title('3-Classfication (k = %i, weight = %s)'%(n_neighbors,weights))

plt.show()
