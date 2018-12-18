将libsvm拷贝到python目录下的 \Lib\site-package\

然后添加__init__.py文件，然后修改svmutil中的导入代码



svm_train() 

svm_problem(y,x)

训练模型

model = svm_train(y,x,[,'training options'])

y:

x: 包含样本实例的列表或者元组。特征向量是列表、元组或者字典

或者一个1 * n numpy ndarray