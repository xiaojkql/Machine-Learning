print(__doc__)
# The influenc of penalty C

import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

np.random.seed(0)

# np.r_[a,b] 将a和b上下重叠在一起
# np.c_[a,b] 将a和b并排在一起
# 两者都是使用的中括号
# np.random.randn(shape())
X = np.r_[np.random.randn(20,2)+[2,2],np.random.randn(20,2)-[2,2]]
print(np.c_[np.random.randn(20,2)+[2,2],np.random.randn(20,2)-[2,2],np.random.randn(20,2)])
Y = np.r_[np.ones(20),-np.ones(20)]
Y1 = [1] * 20 + [-1] * 20 # 使用列表的加法与乘法运算

fignum = 1

for name,penalty in [('unreg',1),('reg',0.05)]:
    clf = svm.SVC(kernel='linear',C=penalty)
    clf.fit(X,Y)

    Fig = plt.figure(fignum)
    Fig.clf()
    '''
    plt.scatter(X[:,0],X[:,1],c=Y,cmap=plt.cm.Paired)
    ax = plt.gca()
    xlimt = ax.get_xlim()
    ylimt = ax.get_ylim()
    xmesh = np.linspace(xlimt[0],xlimt[1],20)
    ymesh = np.linspace(ylimt[0],ylimt[1],20)
    xxmesh,yymesh = np.meshgrid(xmesh,ymesh)
    xxZ = np.array([xxmesh.ravel(),yymesh.ravel()]).T
    Z = clf.decision_function(xxZ).reshape(xxmesh.shape)
    plt.contour(xxmesh,yymesh,Z,colors='b',levels=[-1,0,1],alpha=0.5,
                linestyles=['--','-','--'])
    '''

    # 这里是直接用直线的方程绘制分类线
    xx = np.linspace(-5,5,20)
    w = clf.coef_[0] # 注意返回的系数的维数(n,m)
    b = clf.intercept_

    # 处于分类线上的点，WX + b = 0 求 X[1]
    yy = -w[0]/w[1]*xx - b/w[1]

    yy_up = -w[0]/w[1]*xx - (b-1)/w[1]
    yy_down = -w[0]/w[1]*xx - (b+1)/w[1]

    plt.scatter(X[:,0],X[:,1],c=Y,zorder=10,s=30,cmap=plt.cm.Paired,edgecolors='k')
    plt.plot(xx,yy,'r-')
    plt.plot(xx,yy_up,'r--')
    plt.plot(xx,yy_down,'r--')
    plt.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1],
                s=100,linewidths=1,zorder=10,facecolors='none',edgecolors='r')
    plt.axis('tight')

# 绘制类别分区
    ax = plt.gca()
    xlimt = ax.get_xlim()
    ylimt = ax.get_ylim()
# mgrid用[]中括号，而不是圆括号
    xMesh,yMesh = np.mgrid[xlimt[0]:xlimt[1]:200j,ylimt[0]:ylimt[1]:200j]
    Z = clf.predict(np.c_[xMesh.ravel(),yMesh.ravel()]).reshape(xMesh.shape)
    plt.pcolormesh(xMesh,yMesh,Z,cmap=plt.cm.Paired)

    # plt.xlim(value,value)
    # plt.ylim(value,value)
    plt.xticks(())
    plt.yticks(())
    # 计算处于边界线+-1上的点 WX + b = +-1 应用函数间隔为1，求几何间隔
    # z = clf.decision_function(xx)
    fignum = fignum + 1

plt.show()