from sklearn import linear_model
import numpy as np


# simple example
X = [[0,0],[1,1]]
y = [0,1]
lineReg = linear_model.LinearRegression().fit(X,y)
lassoReg = linear_model.Lasso(alpha=0.1)
lassoReg.fit(X,y)
print(lineReg.coef_,', ',lineReg.intercept_)
print(lassoReg.coef_,', ',lassoReg.intercept_)
