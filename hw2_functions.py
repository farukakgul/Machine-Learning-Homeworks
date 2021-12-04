import scipy.io
import numpy as np
from random import randrange
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time

q1_dataset = scipy.io.loadmat('q1_dataset.mat')
q2_dataset = scipy.io.loadmat('q2_dataset.mat')

q2_data = np.array(q2_dataset["data"])
q2_data_flattened = q2_data.reshape(150, -1)

inception_features_train = np.array(q1_dataset["inception_features_train"])
inception_features_test = np.array(q1_dataset["inception_features_test"])
hog_features_train = np.array(q1_dataset["hog_features_train"])
hog_features_test = np.array(q1_dataset["hog_features_test"])
superclass_labels_train = np.array(q1_dataset["superclass_labels_train"])
superclass_labels_test = np.array(q1_dataset["superclass_labels_test"])
subclass_labels_train = np.array(q1_dataset["subclass_labels_train"])
subclass_labels_test = np.array(q1_dataset["subclass_labels_test"])

def pca_with_svd( pca_data):
    mean = pca_data.mean(axis=0)
    pca_data = pca_data - mean

    U, s, VT = np.linalg.svd(pca_data)
    print(U.shape)
    print(s.shape)
    print(VT.shape)
    smat = np.zeros((150, 10625))
    smat[:150, :150] = np.diag(s)

    T = np.matmul(np.matmul(U,smat),VT) + mean

    mse = pca_data - T
    mse = mse ** 2
    mse = np.sum(mse) / np.size(T)

    return T, mse


def pca(pca_data):
    means = np.mean(pca_data, axis=0)
    pca_data = pca_data - means
    cov_matrix = np.cov(pca_data.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    last_version = np.matmul( np.matmul(pca_data, eigenvectors), eigenvectors.T)
    original_version = last_version + means

    mse = pca_data - original_version
    mse = mse ** 2
    mse = np.sum(mse) / np.size(pca_data)

    return original_version, mse
#


class LogisticRegression:
    def __init__(self, learning_rate = 0.095, iteration_number = 1000, fit_intercept = True, verbose = True):
        self.lr = learning_rate
        self.it_num = iteration_number
        self.fit_intercept = fit_intercept
        self.verbose = verbose

    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __loss(self, h, y):
        return -(y * np.log(h) + (1 - y) * np.log(1 - h)).mean()

    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        # weight initialization
        self.theta = np.random.normal( 0, 0.01, X.shape[1])
        y = y.reshape(1, 2000)[0]

        for i in range(self.it_num):
            z = np.dot( X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.transpose(), (y - h)) / y.size
            self.theta = self.theta + self.lr * gradient

            if self.verbose == True and i % 100 == 0:
                print("thetas")
                print(self.theta)
                # z = np.dot(X, self.theta)
                # h = self.__sigmoid(z)
                # print(f'loss: {self.__loss(h, y)} \t')

    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        return self.__sigmoid(np.dot(X, self.theta))

    def predict(self, X, threshold):
        return self.predict_prob(X) >= threshold

    def get_thetas(self):
        return self.theta



class StochasticLogisticRegression(LogisticRegression):
    def __init__(self, learning_rate=0.000095, iteration_number=1000, fit_intercept=True, verbose=True):
        self.lr = learning_rate
        self.it_num = iteration_number
        self.fit_intercept = fit_intercept
        self.verbose = verbose

    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __loss(self, h, y):
        return -(y * np.log(h) + (1 - y) * np.log(1 - h)).mean()

    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        # weight initialization
        self.theta = np.random.normal(0, 0.01, X.shape[1])
        y = y.reshape(1, 2000)[0]

        for i in range(self.it_num):
            for j in range(X.shape[0]):
                z = np.dot(X[j], self.theta)
                h = self.__sigmoid(z)
                gradient = np.dot( X[j].T, (y[j] - h))
                self.theta = self.theta + self.lr * gradient

            if self.verbose == True and i % 100 == 0:
                z = np.dot(X, self.theta)
                h = self.__sigmoid(z)
                print(f'loss: {self.__loss(h, y)} \t')

    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        return self.__sigmoid(np.dot(X, self.theta))

    def predict(self, X, threshold):
        return self.predict_prob(X) >= threshold



class MiniBatchLogisticRegression(LogisticRegression):
    def __init__(self, learning_rate=0.000095, iteration_number=1000, fit_intercept=True, verbose=True):
        self.lr = learning_rate
        self.it_num = iteration_number
        self.fit_intercept = fit_intercept
        self.verbose = verbose

    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __loss(self, h, y):
        return -(y * np.log(h) + (1 - y) * np.log(1 - h)).mean()

    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        # weight initialization
        self.theta = np.random.normal(0, 0.01, X.shape[1])
        y = y.reshape(1, 2000)[0]

        for i in range(self.it_num):
            for j in range(0,X.shape[0],25):
                z = np.dot(X[j:j+25], self.theta)
                h = self.__sigmoid(z)
                gradient = np.dot( X[j:j+25].T, (y[j:j+25] - h))
                self.theta = self.theta + self.lr * gradient

            if self.verbose == True and i % 100 == 0:
                z = np.dot(X, self.theta)
                h = self.__sigmoid(z)
                print(f'loss: {self.__loss(h, y)} \t')

    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        return self.__sigmoid(np.dot(X, self.theta))

    def predict(self, X, threshold):
        return self.predict_prob(X) >= threshold



def stratified_k_fold(features, labels, folds):
    labels = labels.reshape(1,-1)
    fold_size = int(features.shape[0] / folds)
    dataset_split = np.zeros((folds, fold_size, features.shape[1]))
    split_labels = np.zeros((folds, fold_size, 1))
    dataset_zeros = features[(labels == 0)[0]]
    dataset_ones = features[(labels == 1)[0]]

    for i in range(folds):
        fold = np.zeros((fold_size, features.shape[1]))
        boolean = True
        index = 0
        while index < fold_size:
            if boolean:
                random_index = randrange( dataset_ones.shape[0])
                fold[index] = dataset_ones[random_index]
                dataset_ones = np.delete(dataset_ones, random_index, 0)
                split_labels[i][index] = 1
            else :
                random_index = randrange( dataset_zeros.shape[0])
                fold[index] = dataset_zeros[random_index]
                dataset_zeros = np.delete(dataset_zeros, random_index, 0)
                split_labels[i][index] = 0
            index += 1
            boolean = not boolean
        dataset_split[i] = fold
    return dataset_split, split_labels

def stratified_k_fold_multilabel(features, labels, folds):
    labels = labels.reshape(1,-1)
    fold_size = int(features.shape[0] / folds)
    dataset_split = np.zeros((folds, fold_size, features.shape[1]))
    split_labels = np.zeros((folds, fold_size, 1))
    dataset_zeros = features[(labels == 0)[0]]
    dataset_ones = features[(labels == 1)[0]]
    dataset_twos = features[(labels == 2)[0]]
    dataset_threes = features[(labels == 3)[0]]
    dataset_fours = features[(labels == 4)[0]]
    dataset_fives = features[(labels == 5)[0]]
    dataset_sixes = features[(labels == 6)[0]]
    dataset_sevens = features[(labels == 7)[0]]
    dataset_eights = features[(labels == 8)[0]]
    dataset_nines = features[(labels == 9)[0]]
    for i in range(folds):
        fold = np.zeros((fold_size, features.shape[1]))
        boolean = 0
        index = 0
        while index < fold_size:
            if boolean == 0:
                random_index = randrange( dataset_zeros.shape[0])
                fold[index] = dataset_zeros[random_index]
                dataset_zeros = np.delete(dataset_zeros, random_index, 0)
                split_labels[i][index] = 0
            elif boolean == 1 :
                random_index = randrange( dataset_ones.shape[0])
                fold[index] = dataset_ones[random_index]
                dataset_ones = np.delete(dataset_ones, random_index, 0)
                split_labels[i][index] = 1
            elif boolean == 2 :
                random_index = randrange( dataset_twos.shape[0])
                fold[index] = dataset_twos[random_index]
                dataset_twos = np.delete(dataset_twos, random_index, 0)
                split_labels[i][index] = 2
            elif boolean == 3:
                random_index = randrange( dataset_threes.shape[0])
                fold[index] = dataset_threes[random_index]
                dataset_threes = np.delete(dataset_threes, random_index, 0)
                split_labels[i][index] = 3
            elif boolean == 4 :
                random_index = randrange( dataset_fours.shape[0])
                fold[index] = dataset_fours[random_index]
                dataset_fours = np.delete(dataset_fours, random_index, 0)
                split_labels[i][index] = 4
            elif boolean == 5 :
                random_index = randrange( dataset_fives.shape[0])
                fold[index] = dataset_fives[random_index]
                dataset_fives = np.delete(dataset_fives, random_index, 0)
                split_labels[i][index] = 5
            elif boolean == 6 :
                random_index = randrange( dataset_sixes.shape[0])
                fold[index] = dataset_sixes[random_index]
                dataset_sixes = np.delete(dataset_sixes, random_index, 0)
                split_labels[i][index] = 6
            elif boolean == 7 :
                random_index = randrange( dataset_sevens.shape[0])
                fold[index] = dataset_sevens[random_index]
                dataset_sevens = np.delete(dataset_sevens, random_index, 0)
                split_labels[i][index] = 7
            elif boolean == 8 :
                random_index = randrange( dataset_eights.shape[0])
                fold[index] = dataset_eights[random_index]
                dataset_eights = np.delete(dataset_eights, random_index, 0)
                split_labels[i][index] = 8
            elif boolean == 9 :
                random_index = randrange( dataset_nines.shape[0])
                fold[index] = dataset_nines[random_index]
                dataset_nines = np.delete(dataset_nines, random_index, 0)
                split_labels[i][index] = 9
            index += 1
            boolean += 1
            if boolean == 10:
                boolean = 0
        dataset_split[i] = fold
    return dataset_split, split_labels



def soft_margin_svm(x_train, y_train, x_test, y_test, c_list, display=False):
    c = c_list
    accuracies = []
    for i in range(len(c)):
        classifier = SVC(probability=False, kernel='linear', C=c[i])
        classifier.fit(x_train, y_train)
        predicted = classifier.predict(x_test)
        tp = tn = fp = fn = 0
        for k in range(len(predicted)):
            if predicted[k] and y_test[k] == 1:
                tp += 1
            elif not(predicted[k]) and y_test[k] == 0:
                tn += 1
            elif (predicted[k]) and y_test[k] == 0:
                fp += 1
            else:
                fn += 1
        accuracies.append((tp+tn) / (tp + tn + fp + fn))
    if display:
        return [tp, fp, fn, tn]
    return accuracies
        # print("Confusion matrix for ", c[i],": \n", metrics.confusion_matrix(y_test, predicted))


def hard_margin_rbf(x_train, y_train, x_test, y_test, gammas, display=False):
    gamma = gammas
    accuracies = []
    for i in range(len(gamma)):
        classifier = SVC(probability=False, kernel='rbf', gamma=gamma[i])
        classifier.fit(x_train, y_train)
        predicted = classifier.predict(x_test)
        tp = tn = fp = fn = 0
        for k in range(len(predicted)):
            if predicted[k] and y_test[k] == 1:
                tp += 1
            elif not(predicted[k]) and y_test[k] == 0:
                tn += 1
            elif (predicted[k]) and y_test[k] == 0:
                fp += 1
            else:
                fn += 1
        accuracies.append((tp+tn) / (tp + tn + fp + fn))
        # print("Confusion matrix for ", gamma[i],": \n")
        # print(tp, "   ", fp)
        # print(fn, "   ", tn)
    if display:
        return [tp, fp, fn, tn]
    return accuracies


def soft_margin_rbf(x_train, y_train, x_test, y_test, c_list, gamma_lis, display=False):
    accuracies = np.zeros((len(c_list), len(gamma_lis)))
    c = c_list
    gamma = gamma_lis
    for i in range(len(c)):
        for n in range(len(gamma)):
            classifier = SVC(probability=False, kernel='rbf', C=c[i], gamma=gamma[n])
            classifier.fit(x_train, y_train)
            predicted = classifier.predict(x_test)
            tp = tn = fp = fn = 0
            for k in range(len(predicted)):
                if predicted[k] and y_test[k] == 1:
                    tp += 1
                elif not(predicted[k]) and y_test[k] == 0:
                    tn += 1
                elif (predicted[k]) and y_test[k] == 0:
                    fp += 1
                else:
                    fn += 1
            accuracies[i][n] = (tp + tn) / (tp + tn + fp + fn)
    if display:
        return [tp, fp, fn, tn]
    return accuracies

def one_vs_all_soft_margin(x_train, y_train, x_test, y_test, c_l, gammas, display=False):
    c = c_l
    gamma = gammas
    accuracies = np.zeros((len(c), len(gamma)))
    for i in range(len(c)):
        for n in range(len(gamma)):
            model = SVC(probability=False, decision_function_shape='ovr', kernel='rbf', C=c[i], gamma=gamma[n])
            model.fit(x_train, y_train)
            predicted = model.predict(x_test)
            accuracies[i][n] = metrics.accuracy_score(y_test, predicted)
    if display:
        print("Confusion matrix for ", c, gamma,": \n", metrics.confusion_matrix(y_test, predicted))
        print(metrics.classification_report(y_test, predicted))
        print("\n", metrics.confusion_matrix(y_test, predicted))
    return accuracies

def hard_margin_polynomial(x_train, y_train, x_test, y_test, d_list, gamma_list, display=False):
    d = d_list
    gamma = gamma_list
    accuracies = np.zeros((len(d), len(gamma)))
    for i in range(len(d)):
        for n in range(len(gamma)):
            model = SVC(probability=False, kernel='poly', degree=d[i], gamma=gamma[n])
            model.fit(x_train, y_train)
            predicted = model.predict(x_test)
            accuracies[i][n] = metrics.accuracy_score(y_test, predicted)
    if display:
        print("Confusion matrix for ", d, gamma,": \n", metrics.confusion_matrix(y_test, predicted))
        print( metrics.classification_report(y_test, predicted))
        print("Confusion Matrix\n", metrics.confusion_matrix(y_test, predicted))
    return accuracies

# 
# #

#
#
#
#



