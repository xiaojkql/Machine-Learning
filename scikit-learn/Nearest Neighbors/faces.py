import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils.validation import check_random_state

data = datasets.fetch_olivetti_faces()
targets = data.target

'''
X = np.array([[[1,2,3],[4,5,6]],[[1,2,3],[4,5,6]]])
print(X.shape)
print(X.reshape((len(X),-1)))
'''
'''
plt.figure()
plt.imshow(data.images[40])
plt.show()
'''
data = data.images.reshape((len(data.images),-1)) #将图像矩阵表示为一维
train = data[targets<30]
test = data[targets>=30]

# test on a subset of people
n_faces = 5
rng = check_random_state(4)
facesId = rng.randint(test.shape[0],size=n_faces)
test = test[facesId,:]
n_pixels = data.shape[1]
X_train = train[:,:(n_pixels+1)//2]
y_train = train[:,(n_pixels)//2:]
X_test = test[:,:(n_pixels+1)//2]
y_test = test[:,(n_pixels)//2:]

regf = KNeighborsRegressor().fit(X_train,y_train)
y_test_predict = regf.predict(X_test)

image_shape = (64,64)
for i in range(n_faces):
    true_face = np.hstack((X_test[i],y_test[i])) # [1,2,3,4] + [7,8,9,10] -> [1,2,3,4,5,6,7,8,9,10]
    sub = plt.subplot(n_faces,2,i*2+1)
    sub.axis('off')
    sub.imshow(true_face.reshape(image_shape),
               cmap=plt.cm.gray,
               interpolation='nearest')
    sub = plt.subplot(n_faces,2,i*2+2)
    predicFace = np.hstack((X_test[i],y_test_predict[i]))
    sub.axis('off')
    sub.imshow(predicFace.reshape(image_shape),
               cmap=plt.cm.gray,
               interpolation='nearest')
plt.show()




