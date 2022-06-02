import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import copy
from joblib import Parallel, delayed

TEST_SIZE = 57


def split_dataset():
    df = pd.read_csv('databases/wdbc_unsplit_norm.csv')
    # df = pca(df, NUM_FEATURES)  # Reducing dimensions
    y = pd.get_dummies(df.label).values
    y = y[:, 1]
    x = df.drop('label', 1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=23)

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    return x_train, x_test, y_train, y_test


def pca(x_train, x_test, num_features):
   
    
    X = np.hstack((x_train.to_numpy().T, x_test.to_numpy().T))
    scalar = StandardScaler()
    X = scalar.fit_transform(X.T)
    pca = PCA(n_components=num_features)
    principalComponents = pca.fit_transform(X)
    finalDf = pd.DataFrame(data=principalComponents,
                           columns=[f"f{i}" for i in range(1, num_features + 1)])
    return finalDf[:-TEST_SIZE], finalDf[-TEST_SIZE:]


x_train_raw, x_test_raw, y_train, y_test = split_dataset()


def sigmoid(x):
    warnings.filterwarnings('ignore')
    return 1 / (1 + np.exp(-x))



def feed_forward(weights, inputs, keep_rate):
    activation_layers = [sigmoid(np.dot(inputs, weights[0]))]
    dropouts = [np.random.rand(activation_layers[0].shape[0], activation_layers[0].shape[1]) < keep_rate]
    activation_layers[0] = (activation_layers[0] * dropouts[0]) / keep_rate
    for i in range(1, len(weights)):
        activation_layers[i - 1] = np.hstack((activation_layers[i - 1], np.ones((activation_layers[i - 1].shape[0], 1))))
        activation_layer = sigmoid(np.dot(activation_layers[i - 1], weights[i]))
        dropout = np.random.rand(activation_layer.shape[0], activation_layer.shape[1]) < keep_rate
        activation_layer = (activation_layer * dropout) / keep_rate
        activation_layers.append(activation_layer)
        dropouts.append(dropout)
    return activation_layers, dropouts


def backpropagation(weights, activation_layers, y_true, dropouts, keep_rate):
    diff = (activation_layers[-1] - y_true)
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
        train_outs, dropouts = feed_forward(weights, x_train, keep_rate)

        # Backpropagation
        delta_layers = backpropagation(weights, train_outs, y_train, dropouts, keep_rate)

        # Update weights
        weights[0] -= learning_rate * np.dot(x_train.T, delta_layers[0]) / N
        for i in range(1, len(weights)):
            weights[i] -= learning_rate * np.dot(train_outs[i - 1].T, delta_layers[i]) / N

    return weights

def accuracy(weights, x_test, y_test):
    activation_layers, dropout = feed_forward(weights, x_test, 1)
    hit_pred = activation_layers[-1] >= 0.5
    hit_true = y_test >= 0.5
    return (hit_pred == hit_true).sum() / y_test.size

def print_results(accuracy, test_case, num_features):
    print("\n**********************************\nFirst Layer: " + str(test_case[0]) + "\nSecond Layer: " + str(test_case[1]) + "\nKeep Rate of: " + str(test_case[2]) + "\nLearning Rate of: " + str(test_case[3]) + "\nEpoch: " + str(test_case[4]) + "\nAn Accuracy of " + str(round(accuracy, 3)) + " was archieved with " + str(num_features) + " Features\n")


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
    print_results(final_accuracy, test_case, num_features)




hidden_size_1_arr = [2, 4, 8]
hidden_size_2_arr = [0, 2, 4]
keep_rate_arr = [0.95, 1]
learning_rate_arr = [0.01, 0.1]
epochs_arr = [2000, 4000]
combined = [(h1, h2, kr, lr, e) for h1 in hidden_size_1_arr for h2 in hidden_size_2_arr for kr in keep_rate_arr for lr in learning_rate_arr for e in epochs_arr]


for num_features in [2, 5]:
    x_train, x_test = pca(x_train_raw, x_test_raw, num_features)
    x_train = np.hstack((x_train, np.ones((x_train.shape[0], 1))))
    x_test = np.hstack((x_test, np.ones((x_test.shape[0], 1))))
    Parallel(n_jobs=4)(delayed(process)(test_case, x_train, x_test, y_train, y_test, num_features) for test_case in combined)
    #for test_case in combined:
        #process(test_case, x_train, x_test, y_train, y_test, num_features) 



print("######################################")
print("######################################")
print("######################################")
print("ITS TRANSFER TIME!!!!")

# do transfer
