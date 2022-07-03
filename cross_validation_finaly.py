import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
import copy

TEST_SIZE = 57
K_FOLD_NUMBER = 10

# def split_dataset():
#     df = pd.read_csv('databases/wdbc_unsplit_norm.csv')
#     # df = pca(df, NUM_FEATURES)  # Reducing dimensions
#     y = pd.get_dummies(df.label).values
#     y = y[:, 1]
#     x = df.drop('label', 1)
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=90)

#     y_train = y_train.reshape(-1, 1)
#     y_test = y_test.reshape(-1, 1)

#     return x_train, x_test, y_train, y_test

def pca(x_data, num_features):
    scalar = StandardScaler()
    X = scalar.fit_transform(x_data.T)
    pca = PCA(n_components=num_features)
    principalComponents = pca.fit_transform(X)
    return principalComponents

def sigmoid(x):
    warnings.filterwarnings('ignore')
    return 1 / (1 + np.exp(-x))

def feed_forward(weights, inputs, keep_rate):
    activation_layers = [sigmoid(np.dot(inputs, weights[0]))]
    dropouts = [
        np.random.rand(activation_layers[0].shape[0], activation_layers[0].shape[1]) < keep_rate]
    activation_layers[0] = (activation_layers[0] * dropouts[0]) / keep_rate
    for i in range(1, len(weights)):
        activation_layers[i - 1] = np.hstack(
            (activation_layers[i - 1], np.ones((activation_layers[i - 1].shape[0], 1))))
        activation_layer = sigmoid(np.dot(activation_layers[i - 1], weights[i]))
        dropout = np.random.rand(activation_layer.shape[0], activation_layer.shape[1]) < keep_rate
        if i == len(weights) - 1:
            potential = (np.dot(activation_layers[i - 1], weights[i]) * dropout) / keep_rate
        activation_layer = (activation_layer * dropout) / keep_rate
        activation_layers.append(activation_layer)
        dropouts.append(dropout)
    return activation_layers, dropouts, potential

def derivative_loss_array(y_pred, potential, y_true):
    ret_arr = np.zeros(y_pred.shape)
    mask = (y_pred == 0) | (y_pred == 1)
    ret_arr[mask] = 0 * (y_pred[mask] - y_true[mask])
    ret_arr[~mask] = (1 - y_true[~mask] - sigmoid(-potential[~mask])) * (
            1 / (y_pred[~mask] * (1 - y_pred[~mask])))
    return ret_arr

def backpropagation(weights, activation_layers, potential, y_true, dropouts, keep_rate):
    diff = derivative_loss_array(activation_layers[-1], potential, y_true)
    # diff = (activation_layers[-1] - y_true)
    dot_one = activation_layers[-1] * diff
    delta_layers = [dot_one * (1 - activation_layers[-1])]
    delta_layers[0] = (delta_layers[0] * dropouts[-1]) / keep_rate

    for i in range(len(weights) - 1, 0, -1):
        a = np.dot(delta_layers[-1], weights[i][:-1].T)
        b = activation_layers[i - 1] * (1 - activation_layers[i - 1])
        delta_layers.append(a * (dropouts[-1] / keep_rate) * b[:, :-1])
    delta_layers.reverse()
    return delta_layers

def accuracy(weights, x_test, y_test, err_margin):
    activation_layers, dropout, potential = feed_forward(weights, x_test, 1)
    mask = y_test == 1
    correct = (activation_layers[-1][mask] >= 1 - err_margin).sum() + (
            activation_layers[-1][~mask] <= err_margin).sum()
    return correct / y_test.size

# TODO : Real Loss Function
def loss(y_pred, potential, y_true):
    return np.mean((y_pred - y_true)**2)

def train_model(weights, keep_rate, learning_rate, epochs, x_train, y_train):
    N = y_train.size
    for i in range(epochs):
        train_outs, dropouts, potential = feed_forward(weights, x_train, keep_rate)

        # Backpropagation
        delta_layers = backpropagation(weights, train_outs, potential, y_train, dropouts,
                                       keep_rate)

        # Update weights
        weights[0] -= learning_rate * np.dot(x_train.T, delta_layers[0]) / N
        for i in range(1, len(weights)):
            weights[i] -= learning_rate * np.dot(train_outs[i - 1].T, delta_layers[i]) / N
    return weights

def train_transfer(weights, keep_rate, learning_rate, epochs, x_train, y_train):
    N = y_train.size
    for i in range(epochs):
        train_outs, dropouts, potential = feed_forward(weights, x_train, keep_rate)

        # Backpropagation
        delta_layers = backpropagation(weights, train_outs, potential, y_train, dropouts,
                                       keep_rate)

        # Update weights
        weights[len(weights) - 1] -= learning_rate * np.dot(train_outs[len(weights) - 2].T, delta_layers[len(weights) - 1]) / N
    return weights

def accuracy(weights, x_test, y_test, err_margin):
    activation_layers, dropout, potential = feed_forward(weights, x_test, 1)
    mask = y_test == 1
    correct = (activation_layers[-1][mask] >= 1 - err_margin).sum() + (
            activation_layers[-1][~mask] <= err_margin).sum()
    return correct / y_test.size

def process(test_case, x_train, x_test, y_train, y_test, num_features):
    hidden_size_1 = test_case[0]
    hidden_size_2 = test_case[1]
    keep_rate = test_case[2]
    learning_rate = test_case[3]
    epochs = test_case[4]
    if hidden_size_2 == 0:
        W1 = np.random.normal(scale=0.1, size=(num_features + 1, hidden_size_1))
        W2 = np.random.normal(scale=0.1, size=(hidden_size_1 + 1, 1))
        weights = [W1, W2]
    else:
        W1 = np.random.normal(scale=0.1, size=(num_features + 1, hidden_size_1))
        W2 = np.random.normal(scale=0.1, size=(hidden_size_1 + 1, hidden_size_2))
        W3 = np.random.normal(scale=0.1, size=(hidden_size_2 + 1, 1))
        weights = [W1, W2, W3]
    final_weights = train_model(weights, keep_rate, learning_rate, epochs, x_train, y_train)
    final_accuracy = accuracy(final_weights, x_test, y_test, 0.1)
    return final_accuracy, test_case, num_features

def process_K_fold(test_case, x_train, x_test, y_train, y_test, num_features):
    hidden_size_1 = test_case[0]
    hidden_size_2 = test_case[1]
    keep_rate = test_case[2]
    learning_rate = test_case[3]
    epochs = test_case[4]
    if hidden_size_2 == 0:
        W1 = np.random.normal(scale=0.1, size=(num_features + 1, hidden_size_1))
        W2 = np.random.normal(scale=0.1, size=(hidden_size_1 + 1, 1))
        weights = [W1, W2]
    else:
        W1 = np.random.normal(scale=0.1, size=(num_features + 1, hidden_size_1))
        W2 = np.random.normal(scale=0.1, size=(hidden_size_1 + 1, hidden_size_2))
        W3 = np.random.normal(scale=0.1, size=(hidden_size_2 + 1, 1))
        weights = [W1, W2, W3]
    final_weights = train_model(weights, keep_rate, learning_rate, epochs, x_train, y_train)
    final_accuracy_10 = accuracy(final_weights, x_test, y_test, 0.1)
    final_accuracy_30 = accuracy(final_weights, x_test, y_test, 0.3)
    final_accuracy_50 = accuracy(final_weights, x_test, y_test, 0.5)
    return np.array([final_accuracy_10, final_accuracy_30, final_accuracy_50])

def transfer_test(x, y, test_case, num_features):

    x_train, x_test, y_train, y_test  = train_test_split(x, y, test_size=TEST_SIZE, random_state=33)

    x_first, x_second, y_first, y_second = train_test_split(x_train, y_train, test_size=TEST_SIZE*2, random_state=33)

    # initialize model
    hidden_size_1 = test_case[0]
    hidden_size_2 = test_case[1]
    keep_rate = test_case[2]
    learning_rate = test_case[3]
    epochs = test_case[4]
    if hidden_size_2 == 0:
        W1 = np.random.normal(scale=0.1, size=(num_features + 1, hidden_size_1))
        W2 = np.random.normal(scale=0.1, size=(hidden_size_1 + 1, 1))
        weights = [W1, W2]
    else:
        W1 = np.random.normal(scale=0.1, size=(num_features + 1, hidden_size_1))
        W2 = np.random.normal(scale=0.1, size=(hidden_size_1 + 1, hidden_size_2))
        W3 = np.random.normal(scale=0.1, size=(hidden_size_2 + 1, 1))
        weights = [W1, W2, W3]

    # train model    
    final_weights = train_model(weights, keep_rate, learning_rate, epochs, x_first, y_first)
    
    # reinitialize final layer
    if hidden_size_2 == 0:
        W4 = np.random.normal(scale=0.1, size=(hidden_size_1 + 1, 1))
        weights_transfer = [final_weights[0], W4]
    else:
        W4 = np.random.normal(scale=0.1, size=(hidden_size_2 + 1, 1))
        weights_transfer = [final_weights[0], final_weights[1], W4]

    # train transfer
    final_weights_transfer = train_transfer(weights_transfer, keep_rate, learning_rate, epochs, x_second, y_second)
    final_accuracy_10 = accuracy(final_weights, x_test, y_test, 0.1)
    final_accuracy_30 = accuracy(final_weights, x_test, y_test, 0.3)
    final_accuracy_50 = accuracy(final_weights, x_test, y_test, 0.5)
    final_accuracy_10_transfer = accuracy(final_weights_transfer, x_test, y_test, 0.1)
    final_accuracy_30_transfer = accuracy(final_weights_transfer, x_test, y_test, 0.3)
    final_accuracy_50_transfer = accuracy(final_weights_transfer, x_test, y_test, 0.5)

    return np.array([final_accuracy_10, final_accuracy_30, final_accuracy_50, final_accuracy_10_transfer, final_accuracy_30_transfer, final_accuracy_50_transfer])


def print_final(accuracies, test_case, num_features, results):
    print("\n**********************************\nFirst Layer: " + str(
        test_case[0]) + "\nSecond Layer: " + str(test_case[1]) + "\nKeep Rate of: " + str(
        test_case[2]) + "\nLearning Rate of: " + str(test_case[3]) + "\nEpoch: " + str(
        test_case[4]) + "\nWith " + str(num_features) + " features: \n" + "The accuracies were:\n" + str(
        round(accuracies[2], 3)) + "/" +  str(
        round(accuracies[1], 3)) + "/" + str(
        round(accuracies[0], 3)) + "\n\n" +
        "We do transfer learning on this case!" + "\n"
        + "From " + str(
        round(results[2], 3)) + "/" +  str(
        round(results[1], 3)) + "/" + str(
        round(results[0], 3)) + " to " + str(
        round(results[5], 3)) + "/" +  str(
        round(results[4], 3)) + "/" + str(
        round(results[3], 3)))

# assumes x is already PCA-ed
def cross_validation_test(x, y, test_case, num_features):
    # first check on one iteration
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=33)
    final_accuracy, test_case, num_features = process(test_case, x_train, x_test, y_train, y_test, num_features)
    # if good enough proceed to do 10-FOLD-CROSS-VALIDATION
    if final_accuracy > 0.9:
        kf = KFold(n_splits=K_FOLD_NUMBER, random_state = 22, shuffle = True)
        accuracies = np.array([0, 0, 0])
        for train_index, test_index in kf.split(x):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            accuracies = process_K_fold(test_case, x_train, x_test, y_train, y_test, num_features)/K_FOLD_NUMBER + accuracies
        results = transfer_test(x,y, test_case, num_features)

        print_final(accuracies, test_case, num_features, results)






df = pd.read_csv('databases/wdbc_unsplit_norm.csv')
y = pd.get_dummies(df.label).values
y = y[:, 1]
y = y.reshape(-1, 1)
x = df.drop('label', 1)

# comment out when running main for! 
# x = np.hstack((x, np.ones((x.shape[0], 1))))
# x_reduced = pca(x.T, 2)
# x_reduced = np.hstack((x_reduced, np.ones((x_reduced.shape[0], 1))))
# cross_validation_test(x_reduced,y, (8,18,0.95,0.3,2000), 2)
# hidden_size_1_arr = [8]
# hidden_size_2_arr = [8]
# keep_rate_arr = [0.95, 1]
# learning_rate_arr = [0.1]
# epochs_arr = [6000]
# hidden_size_1_arr = [8]
# hidden_size_2_arr = [8]
# keep_rate_arr = [0.95, 1]
# learning_rate_arr = [0.1]
# epochs_arr = [6000]


hidden_size_1_arr = [2]
hidden_size_2_arr = [0]
keep_rate_arr = [0.95, 1]
learning_rate_arr = [0.001, 0.01, 0.1]
epochs_arr = [2000, 4000, 8000, 10000]

combined = [(h1, h2, kr, lr, e) for h1 in hidden_size_1_arr for h2 in hidden_size_2_arr for kr in keep_rate_arr for lr in learning_rate_arr for e in epochs_arr]

for num_features in [2, 5, 10, 15, 20, 30]:
    x_reduced = pca(x.T, num_features)
    x_reduced = np.hstack((x_reduced, np.ones((x_reduced.shape[0], 1))))
    Parallel(n_jobs=12)(delayed(cross_validation_test)(x_reduced, y, test_case, num_features) for test_case in combined)

