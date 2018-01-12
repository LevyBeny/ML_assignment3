import numpy as np
import Co_Training
from sklearn.utils import shuffle


data_path="./slump.csv"

data = np.genfromtxt(data_path, delimiter=',')
X, y = shuffle(data[:, :-1], data[:, -1])
X = X.astype(np.float32)


offset = int(X.shape[0] * 0.8)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

offset= int(X_train.shape[0]*(5/8))
X_labeled,y_label=X_train[:offset],y[:offset]
X_unlabeled=X_train[offset:]

v1=[0,2,4,6,8,10]#range(0,12,2)
v2=range(1,11,2)

model=Co_Training.Co_Training_Classifier(num_of_iter=3,instance_per_iter=6)
model.fit(X_labeled,y_label,X_unlabeled,v1,v2)

res=model.predict(X_test)