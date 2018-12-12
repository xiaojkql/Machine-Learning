# Generalized Linear Models

学习的模型表达式为：
$$
\hat{y}(w, x) = w_0 + w_1 x_1 + ... + w_p x_p
$$
其中$w = (w_1, ..., w_p)$表示系数向量coef_ ，$w_0$ 表示截距intercept_

## 1.Linear Regression(线性回归)

最简单的线性模型，其实也就是最小二乘问题。以残差作为损失函数，即给定一个数据级$(X, Y)$,在学习策略为最小化，损失函数为：
$$
{|| X w - y||_2}^2
$$
最后转换为以下的优化问题：
$$
\min_{w} {|| X w - y||}_2^2
$$
优点：简单容易实现；

缺点：模型从参数$w$的学习依赖于输入矩阵中各个特征变量之间的相关性。当具有很高的相关性时，矩阵$A$就很会是奇异矩阵，所以模型参数$w$很容易受到随机因素的影响。会出现所谓的[多重共线性](https://baike.baidu.com/item/%E5%A4%9A%E9%87%8D%E5%85%B1%E7%BA%BF%E6%80%A7/10201978?fr=aladdin)(multicolinearity)。

使用方法：

```python
sklearn.linear_model.LinearRegression #class
# class的构造函数参数有，所以在使用时首先要构造这样一个class
# class的属性及一些变量有：coef_ 与 intercept_
# 方法有fit(X,Y) predict(X)，score(X,Y)
```

## 2.Ridge Regression(岭回归)

由于上面的线性回归求解时，会造成过拟合的问题，所以加一个惩罚项在其损失函数上，又所谓的$L_2$正则化(Tikhonov regularization 吉洪诺夫 正则化)。所以，优化问题变为：
$$
\min_{w} {{|| X w - y||_2}^2 + \alpha {||w||_2}^2}
$$
$\alpha$权和因子，当它越大时，所学习出来的模型对$X$变量间的[共线性](https://baike.baidu.com/item/%E5%85%B1%E7%BA%BF%E6%80%A7/4021508?fr=aladdin)( collinearity)越鲁棒。

使用方法：

```python
sklearn.linear_model.Ridge(alpha = 1,...) #class 使用之前要进行构造
# class的属性，击class中存储数据的项，coef_,intercept_,n_iter
# class的方法fit(X,Y),predict(X),score(X,Y)
```

### 2.1设置正则化$\alpha$

RidgeCV   --built in Cross Validation

## 3.Lasso Regression(套索回归)

estimate sparse coefficients。学到的模型中模型仅有少部分为不为零，从而减少依赖的变量项。所以该回归方法很适合[压缩感知](https://blog.csdn.net/yq_forever/article/details/55271952)(copressed sensing)。该方法也就是带有$l_1$正则化项得最小二乘方法。所以优化问题变为：
$$
\min_{w} { \frac{1}{2n_{samples}} ||X w - y||_2 ^ 2 + \alpha ||w||_1}
$$
**学习点1**：在一定情况下，是可以恢复非零项系数的集合[例子](https://scikit-learn.org/stable/auto_examples/applications/plot_tomography_l1_reconstruction.html#sphx-glr-auto-examples-applications-plot-tomography-l1-reconstruction-py)。

**学习点2**：该模型采用的是坐标梯度下降优化算法来求解优化问题。

**学习点3**：在低水平下，采用**lasso_path**计算系数。

**学习点4**：该方法可以用来选择feature

```python

```

### 3.1设置正则化系数$\alpha$的方法

1.使用交叉验证(Cross Validation)

​	scikit-learn objects 有 LassoCV和LassoLarsCV(基于[最小角回归](https://www.baidu.com/link?url=98hML86nRQ4H0ciqmi2OARnXQZJ3oDir_itAm8kYPArjxbrc72XcUvo6tKYujJgr7ZJAwqLshGLFBK7uRWsW6SMjOz7Ts2MGwDTeJ-FX-K_&wd=&eqid=d02370f0000962f5000000065c0f159d))，对于高维问题(样本数相对较少时，使用LassoCV)

2.信息准则

LassoLarsIC基于[赤池信息量准则](https://www.baidu.com/link?url=7-hUq-22lCAX5cFkSdlLIoVDO_mXKvQaa8voYVsQtkmfR5oWgJ2Bcy2hc9mOEhRF8p36kfiqdIaO8BL_rJkUDCbViTqLWE4JMheBpUh8eW9_k-iCzqhGucUQYwNrlUvP6M031ELfXKQQA_s14vEAccL50EX8cB06ybVeAJa5DMG&wd=&eqid=871f44a0000a0182000000065c0f1623)和[贝叶斯信息准则](http://www.baidu.com/link?url=8tR7UtTUCA-9jBpohzLSJCCks_QqRhlUFdXCYIhIsH05WwioMfP5lHyuKF5aTHkE9E5N-eHXxEqnzAWZCBwCtNaCZnwrzfHjs2V7L9vBj8ecXtRd5UMMskWYOHBc8ctTYhx4m0nQh_T6jfh3X8X6FTMbsMOHGpOfKw0zHJgbKxq)

## 4.Multi-task Lasso

多任务回归问题，也就是用一个模型一次性拟合多个样本数据集对。

优点：当用Lasso时系数$w$中含有很多的零项，而当使用MultiTaskLasso时基本每一个项都不为零。

优化问题：
$$
\min_{w} { \frac{1}{2n_{samples}} ||X W - Y||_{Fro} ^ 2 + \alpha ||W||_{21}} \\
||A||_{Fro} = \sqrt{\sum_{ij} a_{ij}^2}\\
||A||_{2 1} = \sum_i \sqrt{\sum_j a_{ij}^2}
$$
使用，坐标梯度下降法作为优化算法学习模型

## 5.Elastic Net(弹性网络)

弹性网络是一种在基本回归模型中同时加入$l_1$和$l_2$正则化项得回归模型。学习结果中的系数和Lasso一样含有很多的零项，但同时保持了岭回归的正则化项。

该模型适合于有大量特征，且有些特征之间有很强的相关性，Lasso倾向于选择相关特征中的一个，而Elastic Net倾向于选择两个。

优化问题：
$$
\min_{w} { \frac{1}{2n_{samples}} ||X w - y||_2 ^ 2 + \alpha \rho ||w||_1 +
\frac{\alpha(1-\rho)}{2} ||w||_2 ^ 2}
$$

```python
sklean.linear_model.ElasticNet()  # class
sklean.linear_model.ElasticNetCV() # 调整正则化系数
```

使用坐标梯度下降优化算法求解该优化问题

## 6.Multi-task Elastic Net

用Elastic Net来解决多任务回归问题，优化问题:
$$
\min_{W} { \frac{1}{2n_{samples}} ||X W - Y||_{Fro}^2 + \alpha \rho ||W||_{2 1} +
\frac{\alpha(1-\rho)}{2} ||W||_{Fro}^2}
$$


```python
sklearn.linear_model.MultiTaskElasticNet()
sklearn.linear_model.MultiTaskElasticNetCV() # 采用交叉验证来选择正则化系数
```

## 7.Least Angle Regression(最小角回归)

## 9. Orthogonal Matching Pursuit

 [正交匹配追踪算法](https://www.baidu.com/link?url=nVcKwIiWsvGc5aHNCCXQAvx-7rPOO33OxUYOLEMH1et2GT7xeMHBw6TQ1rqvABkt-1j30GW8Ve4mEXdIrt97jItgX7Qte4wRNBEG2T-Ewr_&wd=&eqid=eea576e8000c41b8000000065c0f1da5)



