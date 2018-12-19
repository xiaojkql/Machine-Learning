### Support Vector Machine

**Support vector machines (SVMs)** are a set of supervised learning methods used for [classification](https://scikit-learn.org/stable/modules/svm.html#svm-classification), [regression](https://scikit-learn.org/stable/modules/svm.html#svm-regression) and [outliers detection](https://scikit-learn.org/stable/modules/svm.html#svm-outlier-detection).

Input: dense **numpy.ndarray numpy.asarray** and **sparse scipy.sparse**  Recommendation:**C-ordered `numpy.ndarray` (dense) or`scipy.sparse.csr_matrix` (sparse) with `dtype=float64`**

#### 1.Classification

[**`SVC`**](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC), [`NuSVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html#sklearn.svm.NuSVC) and [`LinearSVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC) are **classes** capable of performing multi-class classification on a dataset.

Input :an array X of size `[n_samples, n_features]`holding the training samples, and an array y of class labels (strings or integers), size `[n_samples]`

`sklearn.svm.``SVC`(*C=1.0*, *kernel=’rbf’*, *degree=3*, *gamma=’auto_deprecated’*, *coef0=0.0*, *shrinking=True*, *probability=False*, *tol=0.001*, *cache_size=200*, *class_weight=None*, *verbose=False*, *max_iter=-1*, *decision_function_shape=’ovr’*, *random_state=None*)

使用方法：先创建一个类的对象  svm.SVC(parameter set) 的类，类初始化。然后使用它的方法method进行操作。使用属性获得一些操作以后的数据

method: fit(X,y)，decision_function(X)、

The probability model is created using cross validation, so the results can be slightly different than those obtained by predict. Also, it will produce meaningless results on very small datasets.

#### 2.Multi-class classification

#### 3.kernel

The parameter `C`, common to all SVM kernels, trades off misclassification of training examples against simplicity of the decision surface. A low `C` makes the decision surface smooth, while a high `C` aims at classifying all training examples correctly. `gamma` defines how much influence a single training example has. The larger `gamma` is, the closer other examples must be to be affected

