print(__doc__)

import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model,datasets
from sklearn.metrics import mean_squared_error

# load data
diabetes = datasets.load_diabetes() # 从sklearn的某个数据集中导入数据的方式

# diabetes.data[:,2]是一个一维向量 diabetes.data[:,0:2]就不会降维
# numpy提取某一维的一个元素时都会降一维
x_diabetes = diabetes.data[:,np.newaxis,2]
x_train_diabetes = x_diabetes[:-20]
x_test_diabetes = x_diabetes[-20:]

y_train_diabetes = diabetes.target[:-20]
y_test_diabetes = diabetes.target[-20:]

# 创建线性模型对象
lineReg = linear_model.LinearRegression()

# 用创建的模型对象对数据集进行训练拟合
lineReg.fit(x_train_diabetes,y_train_diabetes)

# 用训练好的线性模型对象进行预测
y_predict = lineReg.predict(x_test_diabetes)

# 打印系数
print(lineReg.coef_,'\n',lineReg.intercept_)
# 打印预测残差
print(mean_squared_error(y_test_diabetes,y_predict))
# 打印 R2 score
print("R2 Score: ",lineReg.score(x_test_diabetes,y_test_diabetes))

# 绘图
# 创建一幅图画
fig = plt.figure()
# 在该图画上创建一个子图
figSub1 = fig.add_subplot(111)
figSub1.scatter(x_test_diabetes,y_test_diabetes,color='black')
figSub1.plot(x_test_diabetes,y_predict,color='red',linewidth=10)

plt.xticks(())
plt.yticks(())

plt.show()




