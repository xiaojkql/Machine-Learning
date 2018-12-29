print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

X = 1./(np.arange(1,11) + np.arange(0,10)[:,np.newaxis])
y = np.ones(10)
print(X)

n_alphas = 200

# logspace(lo,hi,n,base=10) 生成 n 个以10为底平均分(lo,hi)两百分的幂
alphas = np.logspace(-10,-2,n_alphas)

print("============")
print(alphas)

coefs = []
for a in alphas:
    ridge = linear_model.Ridge(alpha=a,fit_intercept=False)
    ridge.fit(X,y)
    coefs.append(ridge.coef_)

ax = plt.gca()

ax.plot(alphas,coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of regularization')
plt.axis('tight')
plt.show()

fig0 = plt.figure()
fig0.add_subplot(111).scatter(X[:,1])
