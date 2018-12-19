from sklearn import svm
import numpy as np

x = np.array([[0,0],[1,1]])
y = [0,1]

clf = svm.SVC(gamma=0.0001)
clf.fit(x,y)
print(clf.predict([[2,2]]))

X = [[0],[1],[2],[3]]
y = [0,1,2,3]
clf = svm.SVC(decision_function_shape='ovo')
clf.fit(X,y)
dec = clf.decision_function([[0.2]])
print(dec)

rdf_svc = svm.SVC(kernel='rbf')
print(rdf_svc.kernel)