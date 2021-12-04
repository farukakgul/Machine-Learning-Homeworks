#Omer Faruk Akgul
import numpy as np
import time
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.animation as animation

start = time.time()


# The function guesses the label of the given test data. For example, P(Y | GPPASUTR) is guessed by using Naive Bayes
# For this function since there is only one 1 for each 8 columns posterior is calculated by extracting the corresponding
# protein. If the expected solution is the one solved by considering each 160 column, the upcoming function written for
# mutual information will be used.
# @param: train_set: is the data that will be used to train the model.
# @param: test_set: is the data that will be used to test the created model.
# @return: result_matrix_with_one_zeros: shows the predictions of the given test set.
# @return: accuracy_rate: shows the rate that the model predicts the label correctly
def naive_bayes_classifier_laplace(train_set, test_set, laplace=0):
    train_set_data = np.delete(train_set, 160, 1)
    train_set_labels = train_set[:, -1:]  # can be converted to int by numpy.ndarry.astype(int)
    total_ones = np.sum(train_set_labels)
    total_zeros = train_set_labels.size - total_ones
    p_one = total_ones / (total_ones + total_zeros)  # prior
    p_zero = 1 - p_one
    position_ones = np.where(train_set_labels == 1)[0]
    position_zeros = np.where(train_set_labels == 0)[0]
    train_set_data_ones = train_set_data[position_ones]
    train_set_data_zeros = train_set_data[position_zeros]

    laplace_prob_one_list = []
    laplace_prob_zero_list = []
    for a in range(laplace+1):
        amino_conditional_prob_one = (train_set_data_ones.sum(axis=0)+a).reshape((1, 160)) / (train_set_data_ones.shape[0] + 2*a)
        laplace_prob_one_list.append(amino_conditional_prob_one)
        amino_conditional_prob_zero = (train_set_data_zeros.sum(axis=0)+a).reshape((1, 160)) / (train_set_data_zeros.shape[0] + 2*a)
        laplace_prob_zero_list.append(amino_conditional_prob_zero)

    test_set_data = np.delete(test_set, 160, 1)
    test_set_labels = test_set[:, -1:]

    accuracies_laplace = []
    for k in range(laplace+1):
        repeat_prob_one = np.repeat(laplace_prob_one_list[k], test_set_data.shape[0], axis=0)
        repeat_prob_zero = np.repeat(laplace_prob_zero_list[k], test_set_data.shape[0], axis=0)
        occurrence_prob_one = np.multiply(repeat_prob_one, test_set_data)
        occurrence_prob_zero = np.multiply(repeat_prob_zero, test_set_data)
        all_conditional_prob_of_test_set_ones = occurrence_prob_one[test_set_data == 1].reshape(test_set_data.shape[0], 8)
        all_conditional_prob_of_test_set_zero = occurrence_prob_zero[test_set_data == 1].reshape(test_set_data.shape[0], 8)
        try:
            log_condition_one = np.log2(all_conditional_prob_of_test_set_ones)
        except RuntimeWarning:
            print("log0 occurred and taken as it is, -inf")
        log_condition_zero = np.log2(all_conditional_prob_of_test_set_zero)
        log_condition_one_sum = log_condition_one.sum(axis=1) + np.log2(p_one)
        log_condition_zero_sum = log_condition_zero.sum(axis=1) + np.log2(p_zero)
        test_set_labels = test_set_labels.reshape(1, test_set_labels.size)[0]
        result_matrix = log_condition_one_sum > log_condition_zero_sum
        result_matrix_with_one_zeros = np.zeros(test_set_labels.size)
        for t in range(test_set_labels.size):
            if result_matrix[t]:
                result_matrix_with_one_zeros[t] = 1
        accuracy = 0
        for p in range(test_set_labels.size):
            if test_set_labels[p] == result_matrix_with_one_zeros[p]:
                accuracy += 1
        accuracies_laplace.append(accuracy / test_set_labels.size * 100)
    return result_matrix_with_one_zeros, accuracies_laplace, log_condition_one_sum, log_condition_zero_sum

# The function guesses the label of the given test data. For example, P(Y | GPPASUTR) is guessed by using Naive Bayes
# For this function since there is only one 1 for each 8 columns posterior is calculated by extracting the corresponding
# protein. If the expected solution is the one solved by considering each 160 column, the upcoming function written for
# mutual information will be used.
# @param: train_set: is the data that will be used to train the model.
# @param: test_set: is the data that will be used to test the created model.
# @return: result_matrix_with_one_zeros: shows the predictions of the given test set.
# @return: accuracies: list containing the rate that the model predicts the label correctly
def naive_bayes_classifier_for_mutual_information(train_set, test_set, laplace=0):
    train_set_data = train_set[:, :-1]
    train_set_labels = train_set[:, -1:]  # can be converted to int by numpy.ndarry.astype(int)
    total_ones = np.sum(train_set_labels)
    total_zeros = train_set_labels.size - total_ones
    p_one = total_ones / (total_ones + total_zeros)  # prior olasılığı veriyor
    p_zero = 1 - p_one
    position_ones = np.where(train_set_labels == 1)[0]
    position_zeros = np.where(train_set_labels == 0)[0]
    train_set_data_ones = train_set_data[position_ones]
    train_set_data_zeros = train_set_data[position_zeros]

    laplace_prob_one_list = []
    laplace_prob_zero_list = []
    laplace_prob_zero_given_one_list = []
    laplace_prob_zero_given_zero_list = []

    test_set_data = test_set[:, :-1]
    reverse_test_set = np.abs(test_set_data - 1)
    test_set_labels = test_set[:, -1:]
    accuracies = []

    for k in range(laplace + 1):
        amino_conditional_prob_one = (train_set_data_ones.sum(axis=0)+k).reshape((1, train_set_data.shape[1])) / (train_set_data_ones.shape[0]+2*k)
        amino_conditional_prob_zero = (train_set_data_zeros.sum(axis=0)+k).reshape((1, train_set_data.shape[1])) / (train_set_data_zeros.shape[0]+2*k)

        prob_of_being_zero_given_one = 1 - amino_conditional_prob_one
        prob_of_being_zero_given_zero = 1 - amino_conditional_prob_zero

        laplace_prob_one_list.append(amino_conditional_prob_one)
        laplace_prob_zero_list.append(amino_conditional_prob_zero)
        laplace_prob_zero_given_one_list.append(prob_of_being_zero_given_one)
        laplace_prob_zero_given_zero_list.append(prob_of_being_zero_given_zero)

        repeat_prob_one = np.repeat(laplace_prob_one_list[k], test_set_data.shape[0], axis=0)
        repeat_prob_zero = np.repeat(laplace_prob_zero_list[k], test_set_data.shape[0], axis=0)
        repeat_prob_being_zero_given_one = np.repeat(laplace_prob_zero_given_one_list[k], test_set_data.shape[0], axis=0)
        repeat_prob_of_being_zero_given_zero = np.repeat(laplace_prob_zero_given_zero_list[k], test_set_data.shape[0], axis=0)

        occurrence_prob_one = np.multiply(repeat_prob_one, test_set_data)
        occurrence_prob_zero = np.multiply(repeat_prob_zero, test_set_data)
        occurrence_prob_zero_given_one = np.multiply(repeat_prob_being_zero_given_one, reverse_test_set)
        occurrence_prob_zero_given_zero = np.multiply(repeat_prob_of_being_zero_given_zero, reverse_test_set)

        all_conditional_prob_of_test_set_ones = occurrence_prob_one + occurrence_prob_zero_given_one
        all_conditional_prob_of_test_set_zero = occurrence_prob_zero + occurrence_prob_zero_given_zero

        log_condition_one = np.log2(all_conditional_prob_of_test_set_ones)
        log_condition_zero = np.log2(all_conditional_prob_of_test_set_zero)
        log_condition_one_sum = log_condition_one.sum(axis=1) + np.log2(p_one)
        log_condition_zero_sum = log_condition_zero.sum(axis=1) + np.log2(p_zero)
        test_set_labels = test_set_labels.reshape(1, test_set_labels.size)[0]
        result_matrix = log_condition_one_sum > log_condition_zero_sum
        result_matrix_with_one_zeros = np.zeros(test_set_labels.size)
        for m in range(test_set_labels.size):
            if result_matrix[m]:
                result_matrix_with_one_zeros[m] = 1
        accuracy = 0
        for n in range(test_set_labels.size):
            if test_set_labels[n] == result_matrix_with_one_zeros[n]:
                accuracy += 1
        accuracy_rate = accuracy / test_set_labels.size * 100
        accuracies.append(accuracy_rate)
    return result_matrix_with_one_zeros, accuracies, log_condition_one_sum, log_condition_zero_sum


train_a = np.loadtxt("q2_train_set.txt", delimiter=",")
test_a = np.loadtxt("q2_test_set.txt", delimiter=",")
print("The accuracy calculated with first function is", naive_bayes_classifier_laplace(train_a, test_a)[1][0])

# part b of the homework  ***********************************************************************************

amino_acids = ["G", "P", "A", "V", "L", "I", "M", "C", "F", "Y", "W", "H", "K", "R", "Q", "N", "E", "D", "S", "T"]
test_file_2 = open("q2_gag_sequence.txt", "r")
sequence = test_file_2.readline()
eight_sequence = []
for i in range(len(sequence) - 7):
    eight_sequence.append(sequence[i:i+8].swapcase())
new_test_array = np.zeros((len(eight_sequence), 161))

index = 0
row = 0
for i in eight_sequence:
    for j in i:
        innerIndex = amino_acids.index(j)
        new_test_array[row][index + innerIndex] = 1
        index += 20
    row += 1
    index = 0

new_labels, garbage, new_ones_prob, new_zeros_prob = naive_bayes_classifier_for_mutual_information(train_a, new_test_array)
cleavage = np.where(new_labels == 1)[0] + 3

indices = []
for i in cleavage:
    temp = "" + str(i) + "-" + str(i+1)
    indices.append(temp)
print("The indices where cleavage may occur: ", indices)
max_one_index = np.where(new_ones_prob == np.amax(new_ones_prob))[0][0]
max_zero_index = np.where(new_zeros_prob == np.amin(new_zeros_prob))[0][0]
print("Model assigns class 1 with highest probability is ", eight_sequence[max_one_index])
print("Model assigns class 0 with lowest probability is", eight_sequence[max_zero_index])

# end of part c of homework1 *********************************************************************

smoothing_list = naive_bayes_classifier_for_mutual_information(train_a, test_a, 10)[1]
print("Maximum accuracy obtained with laplace: ", max(smoothing_list))
print("Laplace accuracies obtained for the alpha values from 0 to 10", np.array(smoothing_list)/100)
print("The best fitting a: ", smoothing_list.index(max(smoothing_list)))

smoothing_list_75 = naive_bayes_classifier_for_mutual_information(train_a[:75], test_a, 10)[1]
print("Maximum accuracy obtained with laplace for 75 data: ", max(smoothing_list_75))
print("Laplace accuracies obtained for the alpha values from 0 to 10", np.array(smoothing_list_75)/100)
print("The best fitting a for 75 data: ", smoothing_list_75.index(max(smoothing_list_75)))

# end of homework 1 part d ******************************************************************
mutual_set = []

for pos in range(160):
    n_1 = train_a[train_a.T[pos] == 1]
    n_11 = n_1[n_1.T[160] == 1].shape[0]
    n_10 = n_1[n_1.T[160] == 0].shape[0]

    n_0 = train_a[train_a.T[pos] == 0]
    n_01 = n_0[n_0.T[160] == 1].shape[0]
    n_00 = n_0[n_0.T[160] == 0].shape[0]

    if n_11 == 0:
        number1 = -float('inf')
    else:
        number1 = (n_11 / train_a.shape[0]) * np.log2(train_a.shape[0] * n_11 / ((n_11+n_10) * (n_11+n_01)))
    number2 = (n_01 / train_a.shape[0]) * np.log2(train_a.shape[0] * n_01 / ((n_00+n_01) * (n_11+n_01)))
    number3 = (n_10 / train_a.shape[0]) * np.log2(train_a.shape[0] * n_10 / ((n_11+n_10) * (n_10+n_00)))
    number4 = (n_00 / train_a.shape[0]) * np.log2(train_a.shape[0] * n_00 / ((n_01+n_00) * (n_10+n_00)))

    mutual_set.append((number1+number2+number3+number4))

descending_indices = np.argsort(mutual_set)[::-1]
mutual_set_ordered = np.sort(mutual_set)[::-1]

print("\nIndices carrying the most information is as follows:")
print(descending_indices)
mutual_accuracies = []

for selected in range(1, 161):
    train_mutual = train_a.T[descending_indices[:selected]].T
    train_labels = train_a[:, -1:]
    train_mutual = np.append(train_mutual, train_labels, axis=1)

    test_mutual = test_a.T[descending_indices[:selected]].T
    test_labels = test_a[:, -1:]
    test_mutual = np.append(test_mutual, test_labels, axis=1)

    mutual_accuracies.append(naive_bayes_classifier_for_mutual_information(train_mutual, test_mutual)[1][0])

mutual_accuracies_np = np.array(mutual_accuracies)
top_mutual_index = np.argsort(mutual_accuracies_np)[::-1] + 1
top_mutual_ordered = np.sort(mutual_accuracies_np)[::-1]

print("\nThe following list shows the sorted k values according to accuracies")
print(top_mutual_index)
print("\nThe following list shows the corresponding accuracies")
print(top_mutual_ordered)
print("Maximum accuracy value is : ", top_mutual_ordered[0], "and the corresponding k value is", np.argmax(mutual_accuracies_np)+1)

# end of part e and part f starts **********************************************************************


def pca_with_svd( train_a):
    pca_data = train_a[:, :-1]
    pca_data = pca_data - pca_data.mean(axis=0)

    U, s, VT = np.linalg.svd(pca_data)
    # create m x n Sigma matrix
    # populate Sigma with n x n diagonal matrix
    # select
    n_elements = 3

    s_3 = s[:n_elements]
    pve_1 = sum(s_3)/sum(s)*100
    print("Proportion of variance explained with SVD is: ", pve_1)
    print("Singular Value Decomposition numpy.linalg.svd spends nearly 8 seconds on my computer so it is the main time consumer.")
    VT = VT[:n_elements, :]
    # transform
    T = pca_data.dot(VT.T)

    return T, pve_1


print("*****************************************")


def pca(train_a):
    pca_data = train_a[:, :-1]
    cov_matrix = np.cov(pca_data.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    eigenvalues_sorted_indices = np.argsort(eigenvalues)
    eigenvalues_sorted = np.sort(eigenvalues)
    chosen_eigenvectors = eigenvectors.T[eigenvalues_sorted_indices[-3:]]

    last_version = np.matmul(pca_data, chosen_eigenvectors.T)

    pve = sum(eigenvalues_sorted[-3:]) / sum(eigenvalues)
    return pve
print("Proportion of variance explained by sorting eigenvalues is : ", pca(train_a)*100, "%(percent)")
print("Calculating eigenvalues and choosing most important eigenvectors causes some complex numbers so some errors but it is really fast.")

T = pca_with_svd(train_a)[0]

print("Total time from the beginning is ", time.time() - start)

plt.clf()
k = np.arange(1,161)
plt.plot(k, mutual_accuracies, ".-")
plt.xlabel("k")
plt.ylabel("Accuracy(%)")
plt.legend(["Data"])
plt.title("k vs Accuracy")
plt.show()

a_values = np.arange(11)
plt.clf()
plt.plot(a_values, smoothing_list, "*-")
plt.xlabel("Laplace value (a)")
plt.ylabel("Accuracy percentage")
plt.title("Laplace Smoothing Values vs Accuracy")
plt.grid(True)
plt.legend(["Accuracy"])
plt.show()

a_values_75 = np.arange(11)
plt.clf()
plt.plot(a_values, smoothing_list_75, "*-")
plt.xlabel("Laplace value (a)")
plt.ylabel("Accuracy percentage")
plt.title("Laplace Smoothing Values vs Accuracy (75 Data Points)")
plt.grid(True)
plt.legend(["Accuracy"])
plt.show()


fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot3D(T.T[0][train_a.T[160] == 1], T.T[1][train_a.T[160] == 1], T.T[2][train_a.T[160] == 1], ".")
ax.plot3D(T.T[0][train_a.T[160] == 0], T.T[1][train_a.T[160] == 0], T.T[2][train_a.T[160] == 0], "r.")
ax.set_xlabel("1st Component")
ax.set_ylabel("2nd Component")
ax.set_zlabel("3rd Component")
ax.set_title("PCA with 3 Components (6062 Data Points)")
ax.legend(["Label 1", "Label 0"])
plt.show()



