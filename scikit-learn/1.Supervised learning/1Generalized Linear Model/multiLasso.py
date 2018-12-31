print(__doc__)
'''
例子思路：
产生一个回归的数据，假设特征有30个，但是Y的产生仅有前5个特征产生
然后用Lasso回归求解这个回归问题中的系数，以及用MultiLasso求解
'''

'''
处理的是一个系数是依据一个时序产生的
最后对比真实系数与拟合出来的系数之间的对比
'''


import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import MultiTaskLasso,Lasso

rng = np.random.RandomState(42) # 记录一个随机数生成器，以便生成相同的随机数列

n_samples,n_features,n_tasks = 100,30,40
n_relevant_features = 5

coef = np.zeros((n_tasks,n_features))
times = np.linspace(0,2*np.pi,n_tasks)

for k in range(n_relevant_features):
    coef[:,k] = np.sin((1.+rng.randn(1))*times + 3*rng.randn(1))

X = rng.randn(n_samples,n_features)
Y = np.dot(X,coef.T) + rng.randn(n_samples,n_tasks)

coef_lasso = np.array([Lasso(alpha=0.5).fit(X,y).coef_ for y in Y.T])
coef_multi_lasso = MultiTaskLasso(alpha=1).fit(X,Y).coef_

fig = plt.figure(figsize=(8,5))
plt.subplot(1,2,1)
plt.spy(coef_lasso)
plt.xlabel('feature')
plt.ylabel('Time(or task')
plt.text(10,5,'Lasso')

plt.subplot(1,2,2)
plt.spy(coef_multi_lasso)
plt.xlabel('Feature')
plt.ylabel('Time (or task')
plt.text(10,5,'MultiTaskLassop')
fig.suptitle('Coefficient non-zero location')

# 选择其中某一个coef进行绘制
feature_to_plot = 1
plt.figure()
lw = 2
plt.plot(coef[:,feature_to_plot],color='seagreen',linewidth=lw,
         label='Groud truth')
plt.plot(coef_lasso[:,feature_to_plot],color='cornflowerblue',linewidth=lw,
         label='MultiTaskLasso')
plt.plot(coef_multi_lasso[:,feature_to_plot],color='gold',linewidth=lw,
         label='MultiTaskLasso')
plt.legend(loc='upper center')
plt.axis('tight')
plt.ylim([-1.1,1.1])


plt.show()