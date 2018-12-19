print(__doc__)  # 打印文件前面的内容
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

# 使用make_blobs 创建样本
X,y = make_blobs(40,2,2,random_state=6)

# Create svm model
clf = svm.SVC(kernel='linear',C=1000) # C denote no error
clf.fit(X,y)

# plot
fig = plt.figure()
subFig1 = fig.add_subplot(111)
subFig1.scatter(X[:,0],X[:,1],c=y,s=30,cmap=plt.cm.Paired) # cmap颜色映射函数
subFig1.set_title("Scatter")
plt.xlabel('x1')
plt.ylabel('x2')


# 从已绘制的图上获取坐标
ax = plt.gca()
xlim = ax.get_xlim() # 获取两个极限坐标值，即每个坐标轴上的两个端点
ylim = ax.get_ylim()

#
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
XX,YY = np.meshgrid(xx,yy)

# XX.ravel() -> array(900,) 以行为优先进行扁平化
# [XX,ravel(), YY.ravel()] -> [array,array]
# vstack(list,tuple,ndarray)除了stack那一维其余都需要维数严格一致
xy = np.vstack([XX.ravel(),YY.ravel()]).T
xy1 = np.array([XX.ravel(),YY.ravel()]) # 两者是一回事

Z = clf.decision_function(xy).reshape(XX.shape)
subFig1.contour(XX,YY,Z,colors='r',levels=[-1,0,1],
                alpha=0.5,linestyles=['--','-','--'])
# 注意contour与contourf两者之间的区别

subFig1.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1],
                s = 100,linewidths=1,facecolors='none',edgecolors='r')
fig.show()


