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
\min_{w} {|| X w - y||_2}^2
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

