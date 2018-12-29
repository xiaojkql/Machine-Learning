from sklearn import linear_model,datasets
import matplotlib.pyplot as plt
import numpy as np

# simple example
reg = linear_model.Ridge(alpha=0.5) # 最基础的提供超参数alpha
reg.fit([[0,0],[0,0],[1,1]],[0,0.1,1])
print(reg.coef_,"  ",reg.intercept_)

# complex example
diabets = datasets.load_diabetes()

x_train = diabets.data[:-20]
x_test = diabets.data[-20:]

y_train = diabets.target[:-20]
y_test = diabets.target[-20:]

# 设置惩罚项系数 alpha
n_alpha = 200
alphas = np.logspace(-1,3,n_alpha) # 以10为底

# 计算每一个alpha下的拟合系数
coefs = []
for alpha in alphas:
    ridgeReg = linear_model.Ridge(alpha=alpha)
    coef = ridgeReg.fit(x_train,y_train).coef_
    coefs.append(coef)

# 绘图
fig = plt.figure()
subFig = fig.add_subplot(111)
plt.plot(alphas,coefs)
ax = plt.gca()
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1]) # 一维数组翻转的方法
plt.axis('tight')
plt.show()

#----------------------------------------------------------
alphas = np.logspace(-100,1,10)
# 嵌入交叉验证的方法选择何时的aplha值
ridgeCV = linear_model.RidgeCV(alphas=alphas).fit(x_train,y_train)
print(ridgeCV.coef_)
print(ridgeCV.alpha_)
print(ridgeCV.score(x_train,y_train))
