print(__doc__)

from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import lasso_path,enet_path
from sklearn import datasets

X,y = datasets.load_diabetes(return_X_y=True)
X /= X.std(axis=0)

eps = 5e-3

print("Computing regularization path using lasso...")
alphas_lasso,coef_lasso,_ =lasso_path(X,y,eps,fit_intercept=False)

print("Computing regularization path using positive lasso...")
alphas_lasso_p,coef_lasso_p,_=lasso_path(X,y,eps,positive=True,fit_intercept=False)

print("Computing regularization path using elastic net...")
alphas_elstic,coef_elastic,_=enet_path(X,y,eps=eps,l1_ratio=0.8,fit_intercept=False)

print("Computing regularization path using positive elastic net...")
alphas_elstic_p,coef_elastic_p,_=enet_path(X,y,eps=eps,
        l1_ratio=0.8,positive=True,fit_intercept=False)

# plot
colors = cycle(['b','r','g','c','k'])
plt.figure(1)
neg_log_alphas_lasso = -np.log10(alphas_lasso)
neg_log_alphas_elastic = -np.log10(alphas_elstic)
for coef_l,coef_e,c in zip(coef_lasso,coef_elastic,colors):
    l1 = plt.plot(neg_log_alphas_lasso,coef_l,color=c)
    l2 = plt.plot(neg_log_alphas_elastic,coef_e,linestyle='--',color=c)

plt.xlabel('-Log(alpha)')
plt.ylabel('coefficients')
plt.title('Lasso and Elastic-Net Paths')
plt.legend((l1[-1],l2[-1]),('lass0','Elastic-Net'),loc='lower left')

neg_log_alphas_lasso_p = -np.log10(alphas_lasso_p)
plt.figure(2)
for coef_p,coef,c in zip(coef_lasso_p,coef_lasso,colors):
    l1 = plt.plot(neg_log_alphas_lasso_p,coef_p,color=c)
    l2 = plt.plot(neg_log_alphas_lasso,coef,linestyle='--',color=c)

plt.xlabel('-Log(alpha)')
plt.ylabel('coefficients')
plt.title('Lasso and positive Lasso')
plt.legend((l1[-1],l2[-1]),("Positive Lasso","Lasso"),loc='lower left')
plt.axis = ('tight')

plt.show()



