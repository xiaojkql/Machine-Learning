### 1.前向分步算法

考虑下列的加法模型
$$
f(x)=\sum\limits_{t=1}^{T}\beta_{t}h(x;\gamma_t)
$$
其中

给定损失函数L转换为求经验最小化从而求得模型
$$
\min\limits_{\beta_t,\gamma_t}\sum\limits_{i=1}^{N}L\left(y_i,\sum\limits_{t=1}^{T}\beta_tb\left(x_i;\gamma_t\right)\right)
$$
这是一个复杂的优化问题，含有很多个$\beta_t$和$\gamma_t$，所以是不是每一步求一个，然后逐步逼近$f(x)$，当然可以，用到的算法就是前向分步算法

——————————————————————————————

输入：训练数据集$data={(x_1,y_1),(x_2,y_2)},...,(x_N,y_N)$;损失函数$L(x,f(x))$;基于函数集${b(x;\gamma)}$;

1.初始化$f_0(x)=0$

2.for $t=1,2,...,T$  do

3.求解$(\beta_t,\gamma_t)=\arg \min\limits_{\beta,\gamma}\sum\limits_{i=1}^{N}L\left(y_i,f_{t-1}(x_i)+\beta b(x_i;\gamma)\right)$

4.更新$f_t(x)=f_{t-1}(x)+\beta_tb(x;\gamma_t)$

5.end for

6.得到最后的加法模型$f(x)=\sum\limits_{t=1}^{T}\beta_tb(x;\gamma_t)$

输出：加法模型$f(x)$

——————————————————————————————————————

前向分步算法将同时求解的大型复杂优化问题简化为逐次求解各个的简化优化问题

### 2.前向分步算法与Adaboost之间的关系

Adboost算法是前向分步算法的特例。这时，模型是由基本分类器组成的加法模型，损失函数时指数函数。

### 3.Adaboost算法的推导过程

#### 3.1 Adaboost的损失函数

指数损失函数
$$
L(y,h(x))=e^{(-yh(x))}
$$
就是分类函数就是0-1损失函数的一致替代损失函数

对于一个线性加性模型：
$$
H(x)=\sum\limits_{t=1}^{T}\alpha_th_t(x)
$$
指数损失函数是：
$$
\begin{align*}
l_{exp}(H|D)&=E_{x-D(x)}\left[L(f(x),H(x))\right]\\
&=E_{x-D(x)}\left[e^{-f(x)H(x)}\right]\\
&=\int_x e^{-f(x)H(x)}P(x)
\end{align*}
$$
其中$D(x)$是给定的样本数据的分布函数，一般最开始的时候，原始数据的分布都假定为均匀分布。

要最小化此函数，变量为$H(x)$，该函数为凸函数，最小值处一定有对H(x)的偏导为零，所以对H(x)求偏导
$$
\frac{\partial l_{exp}(H|D)}{\partial H(x)}=\frac{\partial \int_x e^{-f(x)H(x)}P(x)}{\partial H(x)}
$$
对于二分类问题
$$
f(x)=
\begin{cases}
-1&P(f(x)=-1|x)\\
\;1&P(f(x)=1|x)
\end{cases}
$$
所以求得导数为
$$
\begin{align*}
\frac{\partial l_{exp}(H|D)}{\partial H(x)}&=\frac{\partial \int_x e^{-f(x)H(x)}P(x)}{\partial H(x)}\\
&=-f(x)e^{-f(x)H(x)}P(x)\\
&=\left(-(-1)e^{-(-1)H(x)}P(f(x)=-1|x)+-(1)e^{-(1)H(x)}P(f(x)=1|x)\right)P(x)\\
&=\left( e^{H(x)}P(f(x)=-1|x)-e^{-H(x)}P(f(x)=1|x)\right)P(x)
\end{align*}
$$
因为为凸函数，所以令此偏导为零
$$
\left( e^{H(x)}P(f(x)=-1|x)-e^{-H(x)}P(f(x)=1|x)\right)P(x)=0\\
 e^{H(x)}P(f(x)=-1|x)-e^{-H(x)}P(f(x)=1|x)=0
$$
所以求得的$H(x)$为
$$
H(x)=\frac{1}{2}\ln\frac{P(f(x)=1|x)}{P(f(x)=-1|x)}
$$
所以
$$
sign(H(x))=sign\left(\frac{1}{2}\ln\frac{P(f(x)=1|x)}{P(f(x)=-1|x)}\right)
$$

#### 3.2 Adboost每一步优化

求得$H_{t-1}(x)$后求$h_t(x)$能纠正其错误，所以优化目标转换为：
$$
\begin{align*}
l_{exp}\left(H_{t-1}+h_t)|D\right)&=E_{x-D}\left[e^{-f(x)(H_{t-1}+h_t(x))}\right]\\
&=E_{x-D}\left[e^{-f(x)H_{t-1}}e^{-f(x)h_t(x)}\right]
\end{align*}
$$
