import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler

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
    finalDf = pd.DataFrame(data=principalComponents,
                           columns=[f"f{i}" for i in range(1, num_features + 1)])
    return finalDf

# x_train_raw, x_test_raw, y_train, y_test = split_dataset()

def split_cross_val():

    kf = KFold(n_splits=10, random_state = 70, shuffle = True)
    df = pd.read_csv('databases/wdbc_unsplit_norm.csv')

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
    ret_arr[mask] = 0 * (y_pred[mask] - y_true[mask])
    ret_arr[~mask] = (1 - y_true[~mask] + 1 / (np.exp(potential[~mask]) - 1)) * (
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

def loss(y_pred, potential, y_true):
    return np.mean((y_pred - y_true)**2)

def train_model(weights, keep_rate, learning_rate, epochs, x_train, y_train):
    N = y_train.size
    training_loss = []
    for i in range(epochs):
        train_outs, dropouts, potential = feed_forward(weights, x_train, keep_rate)

        # Backpropagation
        delta_layers = backpropagation(weights, train_outs, potential, y_train, dropouts,
                                       keep_rate)

        # Update weights
        weights[0] -= learning_rate * np.dot(x_train.T, delta_layers[0]) / N
        for i in range(1, len(weights)):
            weights[i] -= learning_rate * np.dot(train_outs[i - 1].T, delta_layers[i]) / N
        training_loss.append(loss(train_outs[-1], potential, y_train))

    plt.plot(training_loss)
    plt.show()
    return weights

def accuracy(weights, x_test, y_test, err_margin):
    activation_layers, dropout, potential = feed_forward(weights, x_test, 1)
    mask = y_test == 1
    correct = (activation_layers[-1][mask] >= 1 - err_margin).sum() + (
            activation_layers[-1][~mask] <= err_margin).sum()
    return correct / y_test.size

def print_results(accuracies, test_case, num_features):
    print("\n**********************************\nFirst Layer: " + str(
        test_case[0]) + "\nSecond Layer: " + str(test_case[1]) + "\nKeep Rate of: " + str(
        test_case[2]) + "\nLearning Rate of: " + str(test_case[3]) + "\nEpoch: " + str(
        test_case[4]) + "\nWith " + str(num_features) + "features: \n" + "The accuracies were:\n" + str(
        round(accuracies[2], 3)) + "/" +  str(
        round(accuracies[1], 3)) + "/" + str(
        round(accuracies[0], 3)) )

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



# assumes x is already PCA-ed
def cross_validation_test(x, y, test_case, num_features):
    # first check on one iteration
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=23)
    final_accuracy, test_case, num_features = process(test_case, x_train, x_test, y_train, y_test, num_features)
    # if good enough proceed to do 10-FOLD-CROSS-VALIDATION
    print(final_accuracy)
    if final_accuracy > 0.9:
        kf = KFold(n_splits=K_FOLD_NUMBER, random_state = 70, shuffle = True)
        accuracies = np.array([0, 0, 0])
        for train_index, test_index in kf.split(x):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            accuracies = process_K_fold(test_case, x_train, x_test, y_train, y_test, num_features)/K_FOLD_NUMBER + accuracies
        print_results(accuracies, test_case, num_features)


df = pd.read_csv('databases/wdbc_unsplit_norm.csv')
y = pd.get_dummies(df.label).values
y = y[:, 1]
y = y.reshape(-1, 1)
x = df.drop('label', 1)
x = np.hstack((x, np.ones((x.shape[0], 1))))
cross_validation_test(x,y, (8,18,0.95,0.3,2000), 30)