# SVM kernels different kernels

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

def test():
# np.c_[[a,b,c],[d,e,f],[m,l,n]] = [[a,d,m],[b.e.l],[c,f,n]]
# np.r_[[a,b,c],[d,e,f],[m,l,n]] = [a,b,c,d,e,f,d,e,f] 直接进行行拼接组成新的行
# np.r_[[[a,b,c]],[[d,e,f]],[[m,l,n]]] = [[a,b,c],[d,e,f],[d,e,f]] 直接进行列拼接
    X = np.c_[(.4, -.7), #(.4,-0.7)就是两行
              (-1.5, -1),
              (-1.4, -.9),
              (-1.3, -1.2),
              (-1.1, -.2),
              (-1.2, -.4),
              (-.5, 1.2),
              (-1.5, 2.1),
              (1, 1),
              # --
              (1.3, .8),
              (1.2, .5),
              (.2, -2),
              (.5, -2.4),
              (.2, -2.3),
              (0, -2.7),
              (1.3, 2.1)].T
    Y = [0]*8 + [1]*8

    figNum = 1
    for name,kernel in [['Linear','linear'],['poly','poly'],['rbf','rbf']]:
        clf = svm.SVC(kernel=kernel,gamma=2)
        clf.fit(X,Y)
        fig = plt.figure(figNum)

        # 绘制原始点
        plt.scatter(X[:,0],X[:,1],c=Y,zorder=10,cmap=plt.cm.Paired,edgecolors=['k','b'],s=30)

        ax = plt.gca()
        xlimit = [-3,3]
        ylimit = [-3,3]
        xx1,xx2 = np.mgrid[xlimit[0]:xlimit[1]:100j,ylimit[0]:ylimit[1]:100j]
        xx = np.c_[xx1.ravel(),xx2.ravel()]
        predScore = clf.decision_function(xx).reshape(xx1.shape)
        predLabel = clf.predict(xx).reshape(xx1.shape)

        # 绘制分类线与+-0.5边界线
        plt.contour(xx1,xx2,predScore,levels=[-.5,0,.5],colors=['r','r','r'],
                    linestyles=['--','-','--'])
        # 用不同颜色绘制不同类的区域
        plt.pcolormesh(xx1,xx2,predLabel,cmap=plt.cm.Paired)

        # 绘制支持向量
        plt.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1],
                    linewidths=2,s=150,edgecolors='r',facecolors='none')
        plt.title(name)
        figNum = figNum + 1
    plt.show()

if __name__=='__main__':
    test()

