import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

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

        view1_features=[0,1,2]
        view2_features=[i for i in range(1,len(data[0])-1) if i not in view1_features]

    if(file_name=="news"):
        data = np.genfromtxt(file_path, delimiter=',')
        X=data[1:,1:-1]
        X = X.astype(np.float32)
        y=data[1:,-1]

        median=np.median(y)
        y[y>median] = 0
        y[y != 0] = 1

        view1_features=[0,1,2,3,4,5,6,7,10,18,19,20,21,22,23,24,25,26,27,28,29]
        view2_features = [i for i in range(1, len(data[0]) - 1) if i not in view1_features]

    if(file_name=="breast_cancer"):
        data = np.genfromtxt(file_path, delimiter=',')
        X=data[:,1:-1]
        X = X.astype(np.float32)
        y=data[:,-1]

        view1_features=[0,1,2,3]
        view2_features = [i for i in range(1, len(data[0]) - 1) if i not in view1_features]

    if(file_name=="mushrooms"):
        data = pd.read_csv(file_path)
        encoder = LabelEncoder()
        for column_name, column_type in data.dtypes.iteritems():
            data[column_name]=encoder.fit_transform(data[column_name])
        data=np.array(data)
        X=data[1:,1:]
        y=data[1:,0]

        view1_features=[i for i in range(0,len(X[0]),2)]
        view2_features = [i for i in range(len(X[0])) if i not in view1_features]

    if (file_name == "income"):
        data = pd.read_csv(file_path)
        encoder = LabelEncoder()
        onehotencoder = OneHotEncoder()
        for column_name, column_type in data.dtypes.iteritems():
            labels=data[column_name].unique()
            new_col = encoder.fit_transform(data[column_name])
            if(column_name not in ['age', 'fnlwgt','capital.gain','capital.loss','education.num','hours.per.week','income']):
                new_col = onehotencoder.fit_transform(new_col.reshape(-1,1))
                labels[...]=column_name+'_'+labels[...]
                new_col = pd.DataFrame(new_col.toarray(),columns=labels)
                data= pd.concat([new_col,data],axis=1)
                del data[column_name]
            else:
                data[column_name]=new_col

        y=data['income']
        del data['income']
        data = np.array(data)

        X = data[1:, :]
        view1_features = [i for i in range(0, len(X[0]), 2)]
        view2_features = [i for i in range(len(X[0])) if i not in view1_features]


    return X, y, view1_features, view2_features

X, y, view1_features, view2_features = extract_data(directory_in_str+"/income_train.csv","income")
