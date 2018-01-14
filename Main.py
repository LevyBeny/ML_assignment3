import numpy as np
import Co_Training
from sklearn.utils import shuffle
import os

directory_in_str = "./data"

def extract_data(file_path, file_name):
    if(file_name=="nba"):
        data = np.genfromtxt(file_path, delimiter=',')
        X=data[1:,1:-1]
        X = X.astype(np.float32)
        y=data[1:,-1]

        view1_features=[1,2,3]
        view2_features=[i for i in range(1,len(data[0])-1) if i not in view1_features]

    if(file_name=="news"):
        data = np.genfromtxt(file_path, delimiter=',')
        X=data[1:,1:-1]
        X = X.astype(np.float32)
        y=data[1:,-1]

        median=np.median(y)
        y[y>median] = 0
        y[y != 0] = 1

        view1_features=[1,2,3,4,5,6,7,8,9,10]

    return X, y ,view1_features, view2_features

X, y, view1_features, view2_features = extract_data(directory_in_str+"/news_train.csv","news")

# X, y = shuffle(data[:, :-1], data[:, -1])
#
#
#
# offset = int(X.shape[0] * 0.8)
# X_train, y_train = X[:offset], y[:offset]
# X_test, y_test = X[offset:], y[offset:]
#
# offset= int(X_train.shape[0]*(5/8))
# X_labeled,y_label=X_train[:offset],y[:offset]
# X_unlabeled=X_train[offset:]
#
# v1=[0,2,4,6,8,10]#range(0,12,2)
# v2=range(1,11,2)
#
# model=Co_Training.Co_Training_Classifier(num_of_iter=3,instance_per_iter=6)
# model.fit(X_labeled,y_label,X_unlabeled,v1,v2)
#
# res=model.predict(X_test)