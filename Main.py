import re
import numpy as np
import os
import Co_Training
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from datetime import datetime
from math import ceil


def split_data(X, y, train_test_split=0.7, labeled_unlabeled_split=0.6):
    offset = int(X.shape[0] * train_test_split)
    X_train, y_train = X[:offset], y[:offset]
    X_test, y_test = X[offset:], y[offset:]

    offset = int(X_train.shape[0] * labeled_unlabeled_split)
    X_labeled, y_labeled = X_train[:offset], y[:offset]
    X_unlabeled = X_train[offset:]

    return X_labeled, X_unlabeled, y_labeled, X_test, y_test


def extract_data(data_path):
    return None, None, None, None


def calc_G_and_k(unlabeled_size):
    threshold = 0.5
    num_to_label = ceil((1 - threshold) * unlabeled_size)
    K = 10
    G = ceil(num_to_label / K)
    return K, G


def evaluate_co_model(X_labeled, X_unlabeled, y_labeled, X_test, y_test, view1, view2, isSVM=True):
    K, G = calc_G_and_k(X_unlabeled.shape[0])
    if isSVM:
        co_model = Co_Training.Co_Training_Classifier(K=K, G=G)
    else:
        co_model = Co_Training.Co_Training_Classifier(base_model=RandomForestClassifier(), K=K, G=G)

    # train the co-training model
    start = datetime.now()
    co_model.fit(X_labeled, y_labeled, X_unlabeled, view1, view2)
    end = datetime.now()
    co_fit_time = (end - start).total_seconds()

    # predict with co-training model
    start = datetime.now()
    co_y_pred = co_model.predict(X_test)
    end = datetime.now()
    co_predict_time = (end - start).total_seconds()

    # calculate accuracy and f1
    co_f1 = f1_score(y_test, co_y_pred)
    co_acc = accuracy_score(y_test, co_y_pred)

    return co_fit_time, co_predict_time, co_f1, co_acc


def evaluate_regular_model(X_labeled, y_labeled, X_test, y_test, isSVM=True):
    if isSVM:
        model = SVC()
    else:
        model = RandomForestClassifier()

    # train
    start = datetime.now()
    model.fit(X_labeled, y_labeled)
    end = datetime.now()
    fit_time = (end - start).total_seconds()

    # predict
    start = datetime.now()
    y_pred = model.predict(X_test)
    end = datetime.now()
    predict_time = (end - start).total_seconds()

    # calculate accuracy and f1
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    return fit_time, predict_time, f1, acc


def cross_validation(data_path, cv=10, isSVM=True):
    X, y, view1, view2 = extract_data(data_path)
    avg_co_fit_time, avg_co_predict_time, avg_co_f1, avg_co_acc = 0, 0, 0, 0
    avg_reg_fit_time, avg_reg_predict_time, avg_reg_f1, avg_reg_acc = 0, 0, 0, 0

    for i in range(cv):
        X, y = shuffle(X, y)
        X_labeled, X_unlabeled, y_labeled, X_test, y_test = split_data(X, y, train_test_split=0.8,
                                                                       labeled_unlabeled_split=5 / 8)

        co_fit_time, co_predict_time, co_f1, co_acc = evaluate_co_model(X_labeled, X_unlabeled,
                                                                        y_labeled, X_test,
                                                                        y_test, view1,
                                                                        view2, isSVM)

        reg_fit_time, reg_predict_time, reg_f1, reg_acc = evaluate_regular_model(X_labeled, y_labeled,
                                                                                 X_test, y_test, isSVM)

        avg_co_fit_time = avg_co_fit_time + (1 / cv) * co_fit_time
        avg_co_predict_time = avg_co_predict_time + (1 / cv) * co_predict_time
        avg_co_f1 = avg_co_f1 + (1 / cv) * co_f1
        avg_co_acc = avg_co_acc + (1 / cv) * co_acc

        avg_reg_fit_time = avg_reg_fit_time + (1 / cv) * reg_fit_time
        avg_reg_predict_time = avg_reg_predict_time + (1 / cv) * reg_predict_time
        avg_reg_f1 = avg_reg_f1 + (1 / cv) * reg_f1
        avg_reg_acc = avg_reg_acc + (1 / cv) * reg_acc

    if isSVM:
        res_path = re.sub(r".csv", "_SVM.txt", data_path)
    else:
        res_path = re.sub(r".csv", "_RF.txt", data_path)
    with open("./results/" + res_path, 'w') as f:
        f.write("Co-training results:\n")
        f.write("Fit Time: " + str(avg_co_fit_time) + "\n")
        f.write("Predict Time: " + str(avg_co_predict_time) + "\n")
        f.write("Accuracy Score: " + str(avg_co_acc) + "\n")
        f.write("F1 Score: " + str(avg_co_f1) + "\n")

        f.write("Regular results:\n")
        f.write("Fit Time: " + str(avg_reg_fit_time) + "\n")
        f.write("Predict Time: " + str(avg_reg_predict_time) + "\n")
        f.write("Accuracy Score: " + str(avg_reg_acc) + "\n")
        f.write("F1 Score: " + str(avg_reg_f1) + "\n")


directory_in_str = "./data"
directory = os.fsencode(directory_in_str)

# loop on all the files
for file in os.listdir(directory):
    file_name = os.fsdecode(file)
    cross_validation(file_name, isSVM=True)
    cross_validation(file_name, isSVM=False)
