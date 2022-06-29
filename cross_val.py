import warnings
import copy
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler

TEST_SIZE = 57

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

def pca(x_train, x_test, num_features):
    X = np.hstack((x_train.T, x_test.T))
    scalar = StandardScaler()
    X = scalar.fit_transform(X.T)
    pca = PCA(n_components=num_features)
    principalComponents = pca.fit_transform(X)
    finalDf = pd.DataFrame(data=principalComponents,
                           columns=[f"f{i}" for i in range(1, num_features + 1)])
    return finalDf[:-TEST_SIZE], finalDf[-TEST_SIZE:]

# x_train_raw, x_test_raw, y_train, y_test = split_dataset()

def split_cross_val():
    kf = KFold(n_splits=10, random_state = 70, shuffle = True)
    df = pd.read_csv('databases/wdbc_unsplit_norm.csv')
    # df = df.iloc[:-1 , :]
    y = pd.get_dummies(df.label).values
    y = y[:, 1]
    x = df.drop('label', 1)
    x = x.to_numpy()
    X_trains = []
    X_tests = []
    y_trains = []
    y_tests = []
    i = 0
    for train_index, test_index in kf.split(x):
        X_train, X_test = x[train_index], x[test_index]
        if i == 9:
            x_tr_temp = X_train[-1]
            print(X_train.shape)
            X_train = X_train[:-1]
            X_test = np.append(X_test, np. array([x_tr_temp]), axis = 0)
        y_train, y_test = y[train_index], y[test_index]
        if i == 9:
            y_tr_temp = y_train[-1]
            y_train = y_train[:-1]
            y_test = np.append(y_test, np. array([y_tr_temp]), axis = 0)
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        X_trains.append(X_train)
        X_tests.append(X_test)
        y_trains.append(y_train)
        y_tests.append(y_test)
        i += 1
    return X_trains, X_tests, y_trains, y_tests
        

X_trains, X_tests, y_trains, y_tests = split_cross_val()

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

# 1/e^z − 1
def derivative_loss_array(y_pred, potential, y_true):
    ret_arr = np.zeros(y_pred.shape)
    mask = (y_pred == 0) | (y_pred == 1)
    ret_arr[mask] = 100 * (y_pred[mask] - y_true[mask])
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

def accuracy(weights, x_test, y_test):
    activation_layers, dropout, potential = feed_forward(weights, x_test, 1)
    mask = y_test == 1
    correct = (activation_layers[-1][mask] >= 0.7).sum() + (
            activation_layers[-1][~mask] <= 0.3).sum()
    return correct / y_test.size

def print_results(accuracy, test_case, num_features):
    print("\n**********************************\nFirst Layer: " + str(
        test_case[0]) + "\nSecond Layer: " + str(test_case[1]) + "\nKeep Rate of: " + str(
        test_case[2]) + "\nLearning Rate of: " + str(test_case[3]) + "\nEpoch: " + str(
        test_case[4]) + "\nAn Accuracy of " + str(
        round(accuracy, 3)) + " was archieved with " + str(num_features) + " Features\n")

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
    final_accuracy = accuracy(final_weights, x_test, y_test)
    return final_accuracy, test_case, num_features

if __name__ == '__main__':
    procs1_scores = []
    procs2_scores = []
    procs3_scores = []
    procs4_scores = []
    procs5_scores = []

    for x_train_raw, x_test_raw, y_train, y_test in zip(X_trains, X_tests, y_trains, y_tests):
        x_train10, x_test10 = pca(x_train_raw, x_test_raw, 10)
        x_train10 = np.hstack((x_train10, np.ones((x_train10.shape[0], 1))))
        x_test10 = np.hstack((x_test10, np.ones((x_test10.shape[0], 1))))

        process1, tc1, nf1 = process((32,0,1,0.3,20000), x_train10, x_test10, y_train, y_test, 10)
        process2, tc2, nf2 = process((2,0,1,0.3,8000), x_train10, x_test10, y_train, y_test, 10)
        procs1_scores.append(process1)
        procs2_scores.append(process2)

        x_train20, x_test20 = pca(x_train_raw, x_test_raw, 20)
        x_train20 = np.hstack((x_train20, np.ones((x_train20.shape[0], 1))))
        x_test20 = np.hstack((x_test20, np.ones((x_test20.shape[0], 1))))

        process3, tc3, nf3 = process((8,0,0.95,0.3,2000), x_train20, x_test20, y_train, y_test, 20)
        process4, tc4, nf4 = process((16,0,1,0.01,20000), x_train20, x_test20, y_train, y_test, 20)
        process5, tc5, nf5 = process((2,0,1,0.01,20000), x_train20, x_test20, y_train, y_test, 20)
        procs3_scores.append(process3)
        procs4_scores.append(process4)
        procs5_scores.append(process5)

    proc1_final_acc = np.mean(procs1_scores)
    proc2_final_acc = np.mean(procs2_scores)
    proc3_final_acc = np.mean(procs3_scores)
    proc4_final_acc = np.mean(procs4_scores)
    proc5_final_acc = np.mean(procs5_scores)

    print_results(proc1_final_acc, (32,0,1,0.3,20000), 10)
    print_results(proc2_final_acc, (2,0,1,0.3,8000), 10)
    print_results(proc3_final_acc, (8,0,0.95,0.3,2000), 20)
    print_results(proc4_final_acc, (16,0,1,0.01,20000), 20)
    print_results(proc5_final_acc, (2,0,1,0.01,20000), 20)