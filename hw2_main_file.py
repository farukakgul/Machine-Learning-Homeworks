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
from odev2 import *

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

model_hog = LogisticRegression()
model_hog.fit(hog_features_train, superclass_labels_train)
predictions_hog = model_hog.predict(hog_features_test, 0.5)
tp_hog = 0
tn_hog = 0
fp_hog = 0
fn_hog = 0
for k in range(len(predictions_hog)):
    if predictions_hog[k] and superclass_labels_test[k] == 1:
        tp_hog += 1
    elif not(predictions_hog[k]) and superclass_labels_test[k] == 0:
        tn_hog += 1
    elif (predictions_hog[k]) and superclass_labels_test[k] == 0:
        fp_hog += 1
    else:
        fn_hog += 1
print()
print("Hog with Batch Gradient Ascent")
print("accuracy hog: ", (tp_hog + tn_hog)/superclass_labels_test.size)
print("precision hog: ", tp_hog / (tp_hog + fp_hog))
print("recall hog: ", tp_hog / (tp_hog + fn_hog))
print("negative predictive hog: ", tn_hog / (tn_hog + fn_hog))
print("false positive rate: ", fp_hog / (tn_hog + fp_hog))
print("false discovery rate: ", fp_hog / (tp_hog + fp_hog))
print("f1 score is: ", 2 * (tp_hog / (tp_hog + fp_hog)) * (tp_hog / (tp_hog + fn_hog)) / ((tp_hog / (tp_hog + fp_hog)) + (tp_hog / (tp_hog + fn_hog))))
print("f2 score is: ", 5 * (tp_hog / (tp_hog + fp_hog)) * (tp_hog / (tp_hog + fn_hog)) / (4 * (tp_hog / (tp_hog + fp_hog)) + (tp_hog / (tp_hog + fn_hog))))
print(tp_hog, "    ", fp_hog)
print(fn_hog, "    ", tn_hog)
print()


thetas1 = np.abs(model_hog.get_thetas())
sorted_thetas1 = thetas1.argsort()[::-1]
print(sorted_thetas1[0:10])
print(thetas1[sorted_thetas1[0:10]])

model = LogisticRegression()
model.fit(inception_features_train, superclass_labels_train)
predictions = model.predict(inception_features_test, 0.5)
tp_inc = 0
tn_inc = 0
fp_inc = 0
fn_inc = 0
for k in range(len(predictions)):
    if predictions[k] and superclass_labels_test[k] == 1:
        tp_inc += 1
    elif not(predictions[k]) and superclass_labels_test[k] == 0:
        tn_inc += 1
    elif (predictions[k]) and superclass_labels_test[k] == 0:
        fp_inc += 1
    else:
        fn_inc += 1

print()
print("Inception with Batch Gradient Ascent")
print("accuracy inception: ", (tp_inc + tn_inc)/superclass_labels_test.size)
print("precision inception: ", tp_inc / (tp_inc + fp_inc))
print("recall inception: ", tp_inc / (tp_inc + fn_inc))
print("negative predictive inception: ", tn_inc / (tn_inc + fn_inc))
print("false positive rate: ", fp_inc / (tn_inc + fp_inc))
print("false discovery rate: ", fp_inc / (tp_inc + fp_inc))
print("f1 score is: ", 2 * (tp_inc / (tp_inc + fp_inc)) * (tp_inc / (tp_inc + fn_inc)) / ((tp_inc / (tp_inc + fp_inc)) + (tp_inc / (tp_inc + fn_inc))))
print("f2 score is: ", 5 * (tp_inc / (tp_inc + fp_inc)) * (tp_inc / (tp_inc + fn_inc)) / (4 * (tp_inc / (tp_inc + fp_inc)) + (tp_inc / (tp_inc + fn_inc))))
print(tp_inc, "    ", fp_inc)
print(fn_inc, "    ", tn_inc)
print()

thetas = np.abs(model.get_thetas())
sorted_thetas = thetas.argsort()[::-1]
print(sorted_thetas[0:10])
print(thetas[sorted_thetas[0:10]])



model2 = StochasticLogisticRegression()
model2.fit(inception_features_train, superclass_labels_train)
predictions_stochastic = model2.predict(inception_features_test, 0.5)
tp_s = 0
tn_s = 0
fp_s = 0
fn_s = 0
for k in range(len(predictions_stochastic)):
    if predictions_stochastic[k] and superclass_labels_test[k] == 1:
        tp_s += 1
    elif not(predictions_stochastic[k]) and superclass_labels_test[k] == 0:
        tn_s += 1
    elif (predictions_stochastic[k]) and superclass_labels_test[k] == 0:
        fp_s += 1
    else:
        fn_s += 1

print()
print("Inception Features with Stochastic Gradient Ascent")
print("accuracy stochastic: ", (tp_s + tn_s)/superclass_labels_test.size)
print("precision stochastic: ", tp_s / (tp_s + fp_s))
print("recall stochastic: ", tp_s / (tp_s + fn_s))
print("negative predictive rate stochastic: ", tn_s / (tn_s + fn_s))
print("false positive rate: ", fp_s / (tn_s + fp_s))
print("false discovery rate: ", fp_s / (tp_s + fp_s))
print("f1 score is: ", 2 * (tp_s / (tp_s + fp_s)) * (tp_s / (tp_s + fn_s)) / ((tp_s / (tp_s + fp_s)) + (tp_s / (tp_s + fn_s))))
print("f2 score is: ", 5 * (tp_s / (tp_s + fp_s)) * (tp_s / (tp_s + fn_s)) / (4 * (tp_s / (tp_s + fp_s)) + (tp_s / (tp_s + fn_s))))
print("Performance Matrix")
print(tp_s, "    ", fp_s)
print(fn_s, "    ", tn_s)
print()

model_s_hog = StochasticLogisticRegression()
model_s_hog.fit(hog_features_train, superclass_labels_train)
predictions_stochastic_hog = model_s_hog.predict(hog_features_test, 0.5)
tp_s_hog = 0
tn_s_hog = 0
fp_s_hog = 0
fn_s_hog = 0
for k in range(len(predictions_stochastic_hog)):
    if predictions_stochastic_hog[k] and superclass_labels_test[k] == 1:
        tp_s_hog += 1
    elif not(predictions_stochastic_hog[k]) and superclass_labels_test[k] == 0:
        tn_s_hog += 1
    elif (predictions_stochastic_hog[k]) and superclass_labels_test[k] == 0:
        fp_s_hog += 1
    else:
        fn_s_hog += 1
print()
print("Hog Features with Stochastic Gradient Ascent")
print("accuracy stochastic: ", (tp_s_hog + tn_s_hog)/superclass_labels_test.size)
print("precision stochastic: ", tp_s_hog / (tp_s_hog + fp_s_hog))
print("recall stochastic: ", tp_s_hog / (tp_s_hog + fn_s_hog))
print("negative predictive rate stochastic: ", tn_s_hog / (tn_s_hog + fn_s_hog))
print("false positive rate: ", fp_s_hog / (tn_s_hog + fp_s_hog))
print("false discovery rate: ", fp_s_hog / (tp_s_hog + fp_s_hog))
print("f1 score is: ", 2 * (tp_s_hog / (tp_s_hog + fp_s_hog)) * (tp_s_hog / (tp_s_hog + fn_s_hog)) / ((tp_s_hog / (tp_s_hog + fp_s_hog)) + (tp_s_hog / (tp_s_hog + fn_s_hog))))
print("f2 score is: ", 5 * (tp_s_hog / (tp_s_hog + fp_s_hog)) * (tp_s_hog / (tp_s_hog + fn_s_hog)) / (4 * (tp_s_hog / (tp_s_hog + fp_s_hog)) + (tp_s_hog / (tp_s_hog + fn_s_hog))))
print("Performance Matrix")
print(tp_s_hog, "    ", fp_s_hog)
print(fn_s_hog, "    ", tn_s_hog)
print()


model3 = MiniBatchLogisticRegression()
model3.fit(hog_features_train, superclass_labels_train)
predictions_mini_batch_hog = model3.predict(hog_features_test, 0.5)
tp_m_hog = 0
tn_m_hog = 0
fp_m_hog = 0
fn_m_hog = 0
for k in range(len(predictions_mini_batch_hog)):
    if predictions_mini_batch_hog[k] and superclass_labels_test[k] == 1:
        tp_m_hog += 1
    elif not(predictions_stochastic_hog[k]) and superclass_labels_test[k] == 0:
        tn_m_hog += 1
    elif (predictions_stochastic_hog[k]) and superclass_labels_test[k] == 0:
        fp_m_hog += 1
    else:
        fn_m_hog += 1
print()
print("Hog Features with Mini Batch Gradient Ascent")
print("accuracy : ", (tp_m_hog + tn_m_hog)/superclass_labels_test.size)
print("precision : ", tp_m_hog / (tp_m_hog + fp_m_hog))
print("recall : ", tp_m_hog / (tp_m_hog + fn_m_hog))
print("negative predictive rate : ", tn_m_hog / (tn_m_hog + fn_m_hog))
print("false positive rate: ", fp_m_hog / (tn_m_hog + fp_m_hog))
print("false discovery rate: ", fp_m_hog / (tp_m_hog + fp_m_hog))
print("f1 score is: ", 2 * (tp_m_hog / (tp_m_hog + fp_m_hog)) * (tp_m_hog / (tp_m_hog + fn_m_hog)) / ((tp_m_hog / (tp_m_hog + fp_m_hog)) + (tp_m_hog / (tp_m_hog + fn_m_hog))))
print("f2 score is: ", 5 * (tp_m_hog / (tp_m_hog + fp_m_hog)) * (tp_m_hog / (tp_m_hog + fn_m_hog)) / (4 * (tp_m_hog / (tp_m_hog + fp_m_hog)) + (tp_m_hog / (tp_m_hog + fn_m_hog))))
print("Performance Matrix")
print(tp_m_hog, "    ", fp_m_hog)
print(fn_m_hog, "    ", tn_m_hog)
print()

model3_inc = MiniBatchLogisticRegression()
model3_inc.fit(inception_features_train, superclass_labels_train)
predictions_mini_batch = model3_inc.predict(inception_features_test, 0.5)
tp_m = 0
tn_m = 0
fp_m = 0
fn_m = 0
for k in range(len(predictions_mini_batch)):
    if predictions_mini_batch[k] and superclass_labels_test[k] == 1:
        tp_m += 1
    elif not(predictions_stochastic_hog[k]) and superclass_labels_test[k] == 0:
        tn_m += 1
    elif (predictions_stochastic_hog[k]) and superclass_labels_test[k] == 0:
        fp_m += 1
    else:
        fn_m += 1
print()
print("Inception Features with Mini Batch Gradient Ascent")
print("accuracy : ", (tp_m + tn_m)/superclass_labels_test.size)
print("precision : ", tp_m / (tp_m + fp_m))
print("recall : ", tp_m / (tp_m + fn_m))
print("negative predictive rate : ", tn_m / (tn_m + fn_m))
print("false positive rate: ", fp_m / (tn_m + fp_m))
print("false discovery rate: ", fp_m / (tp_m + fp_m))
print("f1 score is: ", 2 * (tp_m / (tp_m + fp_m)) * (tp_m / (tp_m + fn_m)) / ((tp_m / (tp_m + fp_m)) + (tp_m / (tp_m + fn_m))))
print("f2 score is: ", 5 * (tp_m / (tp_m + fp_m)) * (tp_m / (tp_m + fn_m)) / (4 * (tp_m / (tp_m + fp_m)) + (tp_m / (tp_m + fn_m))))
print("Performance Matrix")
print(tp_m, "    ", fp_m)
print(fn_m, "    ", tn_m)
print()



# ********************************************************************************

data_set_split, split_labels = stratified_k_fold(inception_features_train, superclass_labels_train, 5)

data_set_split_four = np.zeros((5, int(4/5*inception_features_train.shape[0]), inception_features_train.shape[1]))
split_labels_four = np.zeros((5, int(4/5*superclass_labels_train.shape[0]), superclass_labels_train.shape[1]))

data_set_split_four[0] = np.concatenate((data_set_split[1], data_set_split[2], data_set_split[3], data_set_split[4]))
data_set_split_four[1] = np.concatenate((data_set_split[0], data_set_split[2], data_set_split[3], data_set_split[4]))
data_set_split_four[2] = np.concatenate((data_set_split[0], data_set_split[1], data_set_split[3], data_set_split[4]))
data_set_split_four[3] = np.concatenate((data_set_split[0], data_set_split[1], data_set_split[2], data_set_split[4]))
data_set_split_four[4] = np.concatenate((data_set_split[0], data_set_split[1], data_set_split[2], data_set_split[3]))


split_labels_four[0] = np.concatenate((split_labels[1], split_labels[2], split_labels[3], split_labels[4]))
split_labels_four[1] = np.concatenate((split_labels[0], split_labels[2], split_labels[3], split_labels[4]))
split_labels_four[2] = np.concatenate((split_labels[0], split_labels[1], split_labels[3], split_labels[4]))
split_labels_four[3] = np.concatenate((split_labels[0], split_labels[1], split_labels[2], split_labels[4]))
split_labels_four[4] = np.concatenate((split_labels[0], split_labels[1], split_labels[2], split_labels[3]))

# ***********************************************************************************

data_set_split_hog, split_labels_hog = stratified_k_fold(hog_features_train, superclass_labels_train, 5)

data_set_split_four_hog = np.zeros((5, int(4/5*hog_features_train.shape[0]), hog_features_train.shape[1]))
split_labels_four_hog = np.zeros((5, int(4/5*superclass_labels_train.shape[0]), superclass_labels_train.shape[1]))

data_set_split_four_hog[0] = np.concatenate((data_set_split_hog[1], data_set_split_hog[2], data_set_split_hog[3], data_set_split_hog[4]))
data_set_split_four_hog[1] = np.concatenate((data_set_split_hog[0], data_set_split_hog[2], data_set_split_hog[3], data_set_split_hog[4]))
data_set_split_four_hog[2] = np.concatenate((data_set_split_hog[0], data_set_split_hog[1], data_set_split_hog[3], data_set_split_hog[4]))
data_set_split_four_hog[3] = np.concatenate((data_set_split_hog[0], data_set_split_hog[1], data_set_split_hog[2], data_set_split_hog[4]))
data_set_split_four_hog[4] = np.concatenate((data_set_split_hog[0], data_set_split_hog[1], data_set_split_hog[2], data_set_split_hog[3]))


split_labels_four_hog[0] = np.concatenate((split_labels_hog[1], split_labels_hog[2], split_labels_hog[3], split_labels_hog[4]))
split_labels_four_hog[1] = np.concatenate((split_labels_hog[0], split_labels_hog[2], split_labels_hog[3], split_labels_hog[4]))
split_labels_four_hog[2] = np.concatenate((split_labels_hog[0], split_labels_hog[1], split_labels_hog[3], split_labels_hog[4]))
split_labels_four_hog[3] = np.concatenate((split_labels_hog[0], split_labels_hog[1], split_labels_hog[2], split_labels_hog[4]))
split_labels_four_hog[4] = np.concatenate((split_labels_hog[0], split_labels_hog[1], split_labels_hog[2], split_labels_hog[3]))

# -----------------------------subclass part----------------------------------
data_set_split_sub, split_labels_sub = stratified_k_fold_multilabel(inception_features_train, subclass_labels_train, 5)

data_set_split_four_sub = np.zeros((5, int(4/5*inception_features_train.shape[0]), inception_features_train.shape[1]))
split_labels_four_sub = np.zeros((5, int(4/5*subclass_labels_train.shape[0]), subclass_labels_train.shape[1]))

data_set_split_four_sub[0] = np.concatenate((data_set_split_sub[1], data_set_split_sub[2], data_set_split_sub[3], data_set_split_sub[4]))
data_set_split_four_sub[1] = np.concatenate((data_set_split_sub[0], data_set_split_sub[2], data_set_split_sub[3], data_set_split_sub[4]))
data_set_split_four_sub[2] = np.concatenate((data_set_split_sub[0], data_set_split_sub[1], data_set_split_sub[3], data_set_split_sub[4]))
data_set_split_four_sub[3] = np.concatenate((data_set_split_sub[0], data_set_split_sub[1], data_set_split_sub[2], data_set_split_sub[4]))
data_set_split_four_sub[4] = np.concatenate((data_set_split_sub[0], data_set_split_sub[1], data_set_split_sub[2], data_set_split_sub[3]))


split_labels_four_sub[0] = np.concatenate((split_labels_sub[1], split_labels_sub[2], split_labels_sub[3], split_labels_sub[4]))
split_labels_four_sub[1] = np.concatenate((split_labels_sub[0], split_labels_sub[2], split_labels_sub[3], split_labels_sub[4]))
split_labels_four_sub[2] = np.concatenate((split_labels_sub[0], split_labels_sub[1], split_labels_sub[3], split_labels_sub[4]))
split_labels_four_sub[3] = np.concatenate((split_labels_sub[0], split_labels_sub[1], split_labels_sub[2], split_labels_sub[4]))
split_labels_four_sub[4] = np.concatenate((split_labels_sub[0], split_labels_sub[1], split_labels_sub[2], split_labels_sub[3]))

# ***********************************************************************************

data_set_split_hog_sub, split_labels_hog_sub = stratified_k_fold_multilabel(hog_features_train, subclass_labels_train, 5)

data_set_split_four_hog_sub = np.zeros((5, int(4/5*hog_features_train.shape[0]), hog_features_train.shape[1]))
split_labels_four_hog_sub = np.zeros((5, int(4/5*subclass_labels_train.shape[0]), subclass_labels_train.shape[1]))

data_set_split_four_hog_sub[0] = np.concatenate((data_set_split_hog_sub[1], data_set_split_hog_sub[2], data_set_split_hog_sub[3], data_set_split_hog_sub[4]))
data_set_split_four_hog_sub[1] = np.concatenate((data_set_split_hog_sub[0], data_set_split_hog_sub[2], data_set_split_hog_sub[3], data_set_split_hog_sub[4]))
data_set_split_four_hog_sub[2] = np.concatenate((data_set_split_hog_sub[0], data_set_split_hog_sub[1], data_set_split_hog_sub[3], data_set_split_hog_sub[4]))
data_set_split_four_hog_sub[3] = np.concatenate((data_set_split_hog_sub[0], data_set_split_hog_sub[1], data_set_split_hog_sub[2], data_set_split_hog_sub[4]))
data_set_split_four_hog_sub[4] = np.concatenate((data_set_split_hog_sub[0], data_set_split_hog_sub[1], data_set_split_hog_sub[2], data_set_split_hog_sub[3]))


split_labels_four_hog_sub[0] = np.concatenate((split_labels_hog_sub[1], split_labels_hog_sub[2], split_labels_hog_sub[3], split_labels_hog_sub[4]))
split_labels_four_hog_sub[1] = np.concatenate((split_labels_hog_sub[0], split_labels_hog_sub[2], split_labels_hog_sub[3], split_labels_hog_sub[4]))
split_labels_four_hog_sub[2] = np.concatenate((split_labels_hog_sub[0], split_labels_hog_sub[1], split_labels_hog_sub[3], split_labels_hog_sub[4]))
split_labels_four_hog_sub[3] = np.concatenate((split_labels_hog_sub[0], split_labels_hog_sub[1], split_labels_hog_sub[2], split_labels_hog_sub[4]))
split_labels_four_hog_sub[4] = np.concatenate((split_labels_hog_sub[0], split_labels_hog_sub[1], split_labels_hog_sub[2], split_labels_hog_sub[3]))

print("soft margin svm inception")
soft_accuracies = []
c = [0.01, 0.1, 1, 10, 100]
for j in range(5):
    soft_accuracies.append(soft_margin_svm(data_set_split_four[j], np.ravel(split_labels_four[j]), data_set_split[j], np.ravel(split_labels[j]), c))
soft_means_np = np.mean(np.array(soft_accuracies), axis=0)
print("means\n", soft_means_np)
chosen_c = [c[soft_means_np.argsort()[-1]]]
print("chosen c", chosen_c)
metric_values = soft_margin_svm(inception_features_train, np.ravel(superclass_labels_train)*1., inception_features_test, np.ravel(superclass_labels_test)*1., chosen_c, True)
print("Performance Matrices")
print(metric_values[0], "    ", metric_values[1])
print(metric_values[2], "    ", metric_values[3])
print("Accuracy: ", (metric_values[0] + metric_values[3]) / superclass_labels_test.size)
print("Precision: ", metric_values[0] / (metric_values[0] + metric_values[1]))
print("Recall: ", metric_values[0] / (metric_values[0] + metric_values[2]))

# # ***************************************************
#
print("\n\nSoft Margin SVM for Hog Features\n\n")

soft_accuracies_hog = []
c_hog = [0.01, 0.1, 1, 10, 100]
for j in range(5):
    soft_accuracies_hog.append(soft_margin_svm(data_set_split_four_hog[j], np.ravel(split_labels_four_hog[j]), data_set_split_hog[j], np.ravel(split_labels_hog[j]), c_hog))
soft_means_np_hog = np.mean(np.array(soft_accuracies_hog), axis=0)
print("means", soft_means_np_hog)
chosen_c_hog = [c[soft_means_np_hog.argsort()[-1]]]
print("chosen c", chosen_c_hog)
metric_values_hog = soft_margin_svm(hog_features_train, np.ravel(superclass_labels_train)*1., hog_features_test, np.ravel(superclass_labels_test)*1., chosen_c_hog, True)
print("Performance Matrices")
print(metric_values_hog[0], "    ", metric_values_hog[1])
print(metric_values_hog[2], "    ", metric_values_hog[3])
print("Accuracy: ", (metric_values_hog[0] + metric_values_hog[3]) / superclass_labels_test.size)
print("Precision: ", metric_values_hog[0] / (metric_values_hog[0] + metric_values_hog[1]))
print("Recall: ", metric_values_hog[0] / (metric_values_hog[0] + metric_values_hog[2]))

# *******************************************************************************************
#
print("\n\nHard Margin with RBF Inception v3\n")
hard_rbf_accuracies = []
gamma_list = [1/16, 1/8, 1/4, 1/2, 1, 2, 64]
for j in range(5):
    hard_rbf_accuracies.append(hard_margin_rbf(data_set_split_four[j], np.ravel(split_labels_four[j]), data_set_split[j], np.ravel(split_labels[j]), gamma_list))
hard_rbf_means = np.mean(np.array(hard_rbf_accuracies), axis=0)
print(hard_rbf_accuracies)
print("means\n", hard_rbf_means)
chosen_gamma = [gamma_list[hard_rbf_means.argsort()[-1]]]
print("chosen g", chosen_gamma)
metric_values_hard_rbf = hard_margin_rbf(inception_features_train, np.ravel(superclass_labels_train)*1., inception_features_test, np.ravel(superclass_labels_test)*1., chosen_gamma, True)
print("Performance Matrices")
print(metric_values_hard_rbf[0], "   ", metric_values_hard_rbf[1])
print(metric_values_hard_rbf[2], "   ", metric_values_hard_rbf[3])
print("Accuracy: ", (metric_values_hard_rbf[0] + metric_values_hard_rbf[3]) / superclass_labels_test.size)
print("Precision: ", metric_values_hard_rbf[0] / (metric_values_hard_rbf[0] + metric_values_hard_rbf[1]))
print("Recall: ", metric_values_hard_rbf[0] / (metric_values_hard_rbf[0] + metric_values_hard_rbf[2]))

# **********************************************************************************************
#
print("\n\nHard Margin with RBF HOG\n")
hard_rbf_accuracies_hog = []
gamma_list = [1/16, 1/8, 1/4, 1/2, 1, 2, 64]
for j in range(5):
    hard_rbf_accuracies_hog.append(hard_margin_rbf(data_set_split_four_hog[j], np.ravel(split_labels_four_hog[j]), data_set_split_hog[j], np.ravel(split_labels_hog[j]), gamma_list))
hard_rbf_means_hog = np.mean(np.array(hard_rbf_accuracies_hog), axis=0)
print(hard_rbf_accuracies_hog)
print("means\n", hard_rbf_means_hog)
chosen_gamma_hog = [gamma_list[hard_rbf_means_hog.argsort()[-1]]]
print(chosen_gamma_hog)
metric_values_hard_rbf_hog = hard_margin_rbf(hog_features_train, np.ravel(superclass_labels_train)*1., hog_features_test, np.ravel(superclass_labels_test)*1., chosen_gamma_hog, True)
print("Performance Matrices")
print(metric_values_hard_rbf_hog[0], "   ", metric_values_hard_rbf_hog[1])
print(metric_values_hard_rbf_hog[2], "   ", metric_values_hard_rbf_hog[3])
print("Accuracy: ", (metric_values_hard_rbf_hog[0] + metric_values_hard_rbf_hog[3]) / superclass_labels_test.size)
print("Precision: ", metric_values_hard_rbf_hog[0] / (metric_values_hard_rbf_hog[0] + metric_values_hard_rbf_hog[1]))
print("Recall: ", metric_values_hard_rbf_hog[0] / (metric_values_hard_rbf_hog[0] + metric_values_hard_rbf_hog[2]))

# **************************************************************************************************

print("\n\nSoft Margin with RBF Inception\n")
soft_rbf_accuracies = []
c_lis = [0.01, 1, 100]
gamma_l = [1/4, 2, 64]
for j in range(5):
    soft_rbf_accuracies.append(soft_margin_rbf(data_set_split_four[j], np.ravel(split_labels_four[j]), data_set_split[j], np.ravel(split_labels[j]), c_lis, gamma_l))
print(soft_rbf_accuracies)
three_three_2 = np.zeros((3,3))
for y in range(5):
    three_three_2 += soft_rbf_accuracies[y]
print(three_three_2)
chosen_location = np.argmax(three_three_2)
chosen_c = [c_lis[int (chosen_location/ 3)]]
chosen_gamma = [gamma_l[chosen_location % 3]]
print(chosen_c, "  ", chosen_gamma)
metric_values_soft_rbf = soft_margin_rbf(inception_features_train, np.ravel(superclass_labels_train)*1., inception_features_test, np.ravel(superclass_labels_test)*1., chosen_c, chosen_gamma, True)
print("Confusion matrix for ", chosen_c, " ", chosen_gamma,": \n")
print(metric_values_soft_rbf[0], "   ", metric_values_soft_rbf[1])
print(metric_values_soft_rbf[2], "   ", metric_values_soft_rbf[3])
print("Performance Matrices")
print("Accuracy: ", (metric_values_soft_rbf[0] + metric_values_soft_rbf[3]) / superclass_labels_test.size)
print("Precision: ", metric_values_soft_rbf[0] / (metric_values_soft_rbf[0] + metric_values_soft_rbf[1]))
print("Recall: ", metric_values_soft_rbf[0] / (metric_values_soft_rbf[0] + metric_values_soft_rbf[2]))

# # *******************************************************************
#
print("\n\nSoft Margin with RBF HOG\n")
soft_rbf_accuracies_hog = []
c_lis = [0.01, 1, 100]
gamma_l = [1/4, 2, 64]
for j in range(5):
    soft_rbf_accuracies_hog.append(soft_margin_rbf(data_set_split_four_hog[j], np.ravel(split_labels_four[j]), data_set_split_hog[j], np.ravel(split_labels[j]), c_lis, gamma_l))
print(soft_rbf_accuracies_hog)
three_three = np.zeros((3,3))
for y in range(5):
    three_three += soft_rbf_accuracies_hog[y]
print(three_three)
chosen_location_hog = np.argmax(three_three)
chosen_c_hog = [c_lis[int (chosen_location_hog/ 3)]]
chosen_gamma_hog = [gamma_l[chosen_location_hog % 3]]
print(chosen_c_hog, "  ", chosen_gamma_hog)
metric_values_soft_rbf_hog = soft_margin_rbf(hog_features_train, np.ravel(superclass_labels_train)*1., hog_features_test, np.ravel(superclass_labels_test)*1., chosen_c_hog, chosen_gamma_hog, True)
print("Confusion matrix for ", chosen_c_hog, " ", chosen_gamma_hog,": \n")
print(metric_values_soft_rbf_hog[0], "   ", metric_values_soft_rbf_hog[1])
print(metric_values_soft_rbf_hog[2], "   ", metric_values_soft_rbf_hog[3])
print("Performance Matrices")
print("Accuracy: ", (metric_values_soft_rbf_hog[0] + metric_values_soft_rbf_hog[3]) / superclass_labels_test.size)
print("Precision: ", metric_values_soft_rbf_hog[0] / (metric_values_soft_rbf_hog[0] + metric_values_soft_rbf_hog[1]))
print("Recall: ", metric_values_soft_rbf_hog[0] / (metric_values_soft_rbf_hog[0] + metric_values_soft_rbf_hog[2]))


print("\n\nSoft Margin with RBF One vs All - Inception\n")
soft_rbf_accuracies_ovr = []
c_lis_ovr = [0.01, 1, 100]
gamma_l_ovr = [1/4, 2, 64]
for j in range(5):
    soft_rbf_accuracies_ovr.append(one_vs_all_soft_margin(data_set_split_four_sub[j], np.ravel(split_labels_four_sub[j]), data_set_split_sub[j], np.ravel(split_labels_sub[j]), c_lis_ovr, gamma_l_ovr))
print(soft_rbf_accuracies_ovr)
three_three_ovr = np.zeros((3,3))
for y in range(5):
    three_three_ovr += soft_rbf_accuracies_ovr[y]
print(three_three_ovr/5)
chosen_location_ovr = np.argmax(three_three_ovr)
chosen_c_ovr = [c_lis_ovr[int (chosen_location_ovr/ 3)]]
chosen_gamma_ovr = [gamma_l_ovr[chosen_location_ovr % 3]]
print(chosen_c_ovr, "  ", chosen_gamma_ovr)
one_vs_all_soft_margin(inception_features_train, np.ravel(subclass_labels_train)*1., inception_features_test, np.ravel(subclass_labels_test)*1., chosen_c_ovr, chosen_gamma_ovr, True)

# *******************************************************************

print("\n\nSoft Margin with RBF One vs All - HOG\n")
soft_rbf_accuracies_ovr_hog = []
c_lis_ovr_hog = [0.01, 1, 100]
gamma_l_ovr_hog = [1/4, 2, 64]
for j in range(5):
    soft_rbf_accuracies_ovr_hog.append(one_vs_all_soft_margin(data_set_split_four_hog_sub[j], np.ravel(split_labels_four_hog_sub[j]), data_set_split_hog_sub[j], np.ravel(split_labels_hog_sub[j]), c_lis_ovr_hog, gamma_l_ovr_hog))
print(soft_rbf_accuracies_ovr_hog)
three_three_ovr_hog = np.zeros((3,3))
for y in range(5):
    three_three_ovr_hog += soft_rbf_accuracies_ovr_hog[y]
print(three_three_ovr_hog/5)
chosen_location_ovr_hog = np.argmax(three_three_ovr_hog)
chosen_c_ovr_hog = [c_lis_ovr_hog[int (chosen_location_ovr_hog/ 3)]]
chosen_gamma_ovr_hog = [gamma_l_ovr_hog[chosen_location_ovr_hog % 3]]
print(chosen_c_ovr_hog, "  ", chosen_gamma_ovr_hog)
one_vs_all_soft_margin(hog_features_train, np.ravel(subclass_labels_train)*1., hog_features_test, np.ravel(subclass_labels_test)*1., chosen_c_ovr_hog, chosen_gamma_ovr_hog, True)



# *********************************************************************
print("\n\Hard Margin with Polynomial - Inception\n")
poly_accuracies = []
d = [3, 5, 7]
gamma = [1/4, 2, 64]
for j in range(5):
    poly_accuracies.append(hard_margin_polynomial(data_set_split_four_sub[j], np.ravel(split_labels_four_sub[j]), data_set_split_sub[j], np.ravel(split_labels_sub[j]), d, gamma))
print(poly_accuracies)
three_three_poly = np.zeros((3,3))
for y in range(5):
    three_three_poly += poly_accuracies[y]
print(three_three_poly/5)
chosen_location_poly = np.argmax(three_three_poly)
chosen_d_poly = [d[int (chosen_location_poly/ 3)]]
chosen_gamma_poly = [gamma[chosen_location_poly % 3]]
print(chosen_d_poly, "  ", chosen_gamma_poly)
hard_margin_polynomial(inception_features_train, np.ravel(subclass_labels_train)*1., inception_features_test, np.ravel(subclass_labels_test)*1., chosen_d_poly, chosen_gamma_poly, True)

# # *********************************************************************

print("\n\Hard Margin with Polynomial - HOG\n")
poly_accuracies_hog = []
d_hog = [3, 5, 7]
gamma_hog = [1/4, 2, 64]
for j in range(5):
    poly_accuracies_hog.append(hard_margin_polynomial(data_set_split_four_hog_sub[j], np.ravel(split_labels_four_hog_sub[j]), data_set_split_hog_sub[j], np.ravel(split_labels_hog_sub[j]), d_hog, gamma_hog))
print(poly_accuracies_hog)
three_three_poly_hog = np.zeros((3,3))
for y in range(5):
    three_three_poly_hog += poly_accuracies_hog[y]
print(three_three_poly_hog/5)
chosen_location_poly_hog = np.argmax(three_three_poly_hog)
chosen_d_poly_hog = [d_hog[int (chosen_location_poly_hog/ 3)]]
chosen_gamma_poly_hog = [gamma_hog[chosen_location_poly_hog % 3]]
print(chosen_d_poly_hog, "  ", chosen_gamma_poly_hog)
hard_margin_polynomial(hog_features_train, np.ravel(subclass_labels_train)*1., hog_features_test, np.ravel(subclass_labels_test)*1., chosen_d_poly_hog, chosen_gamma_poly_hog, True)

# ********************************************************

start = time.time()

reconstructed, mse1 = pca_with_svd(q2_data_flattened)
print(mse1)
reconstructed = reconstructed.reshape(150, 85, 125)
print("Total time for svd is ", time.time() - start)
print("*****************************************")

# ********************************************************

start = time.time()
reconstructed_2, mse2 = pca(q2_data_flattened)
print(mse2)
reconstructed_2 = reconstructed_2.real
reconstructed_2 = reconstructed_2.reshape(150, 85, 125)
print("Total time for pca is ", time.time() - start)


plt.clf()
fig, axs = plt.subplots(3, 5)

axs[0,0].imshow(np.array(q2_data)[0], cmap=cm.gray)
axs[0,1].imshow(np.array(q2_data)[1], cmap=cm.gray)
axs[0,2].imshow(np.array(q2_data)[2], cmap=cm.gray)
axs[0,3].imshow(np.array(q2_data)[3], cmap=cm.gray)
axs[0,4].imshow(np.array(q2_data)[4], cmap=cm.gray)
axs[1,0].imshow(np.array(reconstructed)[0], cmap=cm.gray)
axs[1,1].imshow(np.array(reconstructed)[1], cmap=cm.gray)
axs[1,2].imshow(np.array(reconstructed)[2], cmap=cm.gray)
axs[1,3].imshow(np.array(reconstructed)[3], cmap=cm.gray)
axs[1,4].imshow(np.array(reconstructed)[4], cmap=cm.gray)
axs[2,0].imshow(np.array(reconstructed_2)[0], cmap=cm.gray)
axs[2,1].imshow(np.array(reconstructed_2)[1], cmap=cm.gray)
axs[2,2].imshow(np.array(reconstructed_2)[2], cmap=cm.gray)
axs[2,3].imshow(np.array(reconstructed_2)[3], cmap=cm.gray)
axs[2,4].imshow(np.array(reconstructed_2)[4], cmap=cm.gray)

plt.show()


bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(hog_features_train,superclass_labels_train)
dfscores = pd.DataFrame(fit.scores_)
sc = dfscores.values.transpose()[0]
sc_np = np.array(sc)
sc_sorted = np.argsort(sc_np)
sc_s = np.sort(sc_np)
print(sc_sorted[:10])  #print 10 best features
