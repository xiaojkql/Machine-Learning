print(__doc__)

import  numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt


# 产生数据
n_samples,n_features = 50,200
X = np.random.randn(n_samples,n_features)
# 产生系数
coef = np.random.randn(n_features)
inds = np.arange(n_features)
np.random.shuffle(inds)
coef[inds[10:]] = 0
y = np.dot(X,coef)

#
y += 0.01*np.random.normal(size=n_samples)

# split the data
x_train,y_train = X[:n_samples//2],y[:n_samples//2]
x_test,y_test = X[n_samples//2:],y[n_samples//2:]

# 用Lasso与elastic_net分别拟合该数据并提取拟合系数
# Lasso
lassoReg = linear_model.Lasso(alpha=0.1).fit(x_train,y_train)
lassoScore = lassoReg.score(x_test,y_test)
print("Lasso: ",lassoScore)

# 用ElasticNet进行拟合
elasticReg = linear_model.ElasticNet(alpha=0.1,l1_ratio=0.7).fit(x_train,y_train)
elasticScore = elasticReg.score(x_test,y_test)
print("Elastic: ",elasticScore)

# 绘制图比较真实的coef与两种方法求出的系数
fig = plt.figure()
subFig1 = fig.add_subplot(111)
subFig1.plot(coef,'--',color='navy',label='original coefficients')
subFig1.plot(lassoReg.coef_,color='lightgreen',linewidth=2,label='lasso')
subFig1.plot(elasticReg.coef_,color='gold',linewidth=2,label='Elastic')
plt.legend(loc='best')
plt.title("Lasso R^2 score: %f, Elastic R^2 score: %f"%(lassoScore,elasticScore))
plt.show()