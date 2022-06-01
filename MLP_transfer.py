import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import copy

NUM_FEATURES = 10
KEEP_RATE = 0.95


def sigmoid(x):
    warnings.filterwarnings('ignore')
    return 1 / (1 + np.exp(-x))


def derivative_BCE_loss(y_pred, y_true):
    dL = np.zeros(y_true.shape)
    for i in range(len(y_true)):
        if y_true[i] == 0:
            if y_pred[i] == 1:
                dL[i] = 1000000
            else:
                dL[i] = (1 / (1 - y_pred[i]))
        else:
            if y_pred[i] == 0:
                dL[i] = 1000000
            else:
                dL[i] = (-1 / y_pred[i])
    return dL


def mean_squared_error(y_pred, y_true):
    return ((y_pred - y_true) ** 2).sum() / (2 * y_pred.size)


def normalized_mse(y_pred, y_true):
    return (mean_squared_error(y_pred, y_true) / y_true.var()) ** (1 / 2)


def accuracy(y_pred, y_true):
    hit_pred = y_pred >= 0.5
    hit_true = y_true >= 0.5
    return (hit_pred == hit_true).sum() / y_true.size


def split_dataset():
    df = pd.read_csv('databases/wdbc_split_norm.csv')
    y = pd.get_dummies(df.label).values
    y = y[:, 1]
    x = df.drop('label', 1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=100, random_state=23)
    x_train1, x_train2, y_train1, y_train2 = train_test_split(x_train, y_train, test_size=0.3, random_state = 23)

    y_train1 = y_train1.reshape(-1, 1)
    y_train2 = y_train2.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    return x_train1, x_test, y_train1, y_test, x_train2, y_train2


def feed_forward(weights, inputs):
    activation_layers = [sigmoid(np.dot(inputs, weights[0]))]
    for i in range(1, len(weights)):
        activation_layers[i - 1] = np.hstack(
            (activation_layers[i - 1], np.ones((activation_layers[i - 1].shape[0], 1))))
        activation_layers.append(sigmoid(np.dot(activation_layers[i - 1], weights[i])))
    return activation_layers


keep_rate = KEEP_RATE


def feed_forward_dropout(weights, inputs):
    activation_layers = [sigmoid(np.dot(inputs, weights[0]))]
    dropouts = [np.random.rand(activation_layers[0].shape[0], activation_layers[0].shape[1]) < keep_rate]
    activation_layers[0] = (activation_layers[0] * dropouts[0]) / keep_rate
    for i in range(1, len(weights)):
        activation_layers[i - 1] = np.hstack(
            (activation_layers[i - 1], np.ones((activation_layers[i - 1].shape[0], 1))))
        activation_layer = sigmoid(np.dot(activation_layers[i - 1], weights[i]))
        dropout = np.random.rand(activation_layer.shape[0], activation_layer.shape[1]) < keep_rate
        activation_layer = (activation_layer * dropout) / keep_rate
        activation_layers.append(activation_layer)
        dropouts.append(dropout)
    return activation_layers, dropouts


def backpropagation(weights, activation_layers, y_true):
    #diff = np.power(activation_layers[-1] - y_true, 5)
    diff = derivative_BCE_loss(activation_layers[-1], y_true)
    dot_one = activation_layers[-1] * diff
    delta_layers = [dot_one * (1 - activation_layers[-1])]

    for i in range(len(weights) - 1, 0, -1):
        a = np.dot(delta_layers[-1], weights[i][:-1].T)
        b = activation_layers[i - 1] * (1 - activation_layers[i - 1])
        delta_layers.append(a * b[:, :-1])
    delta_layers.reverse()
    return delta_layers


def backpropagation_dropout(weights, activation_layers, y_true, dropouts):
    #diff = np.power(activation_layers[-1] - y_true, 5)
    diff = derivative_BCE_loss(activation_layers[-1], y_true)
    dot_one = activation_layers[-1] * diff
    delta_layers = [dot_one * (1 - activation_layers[-1])]
    delta_layers[0] = (delta_layers[0] * dropouts[-1]) / keep_rate

    for i in range(len(weights) - 1, 0, -1):
        a = np.dot(delta_layers[-1], weights[i][:-1].T)
        b = activation_layers[i - 1] * (1 - activation_layers[i - 1])
        delta_layers.append(a * (dropouts[-1] / keep_rate) * b[:, :-1])
    delta_layers.reverse()
    return delta_layers


if __name__ == '__main__':

    input_size = NUM_FEATURES
    hidden_size_1 = 30
    hidden_size_2 = 30
    output_size = 1
    learning_rate = 0.1
    epochs = 10000

    x_train1, x_test, y_train1, y_test, x_train2, y_train2 = split_dataset()

    x_train1 = np.hstack((x_train1, np.ones((x_train1.shape[0], 1))))
    x_train2 = np.hstack((x_train2, np.ones((x_train2.shape[0], 1))))
    x_test = np.hstack((x_test, np.ones((x_test.shape[0], 1))))

    W1 = np.random.normal(scale=0.1, size=(input_size + 1, hidden_size_1))
    W2 = np.random.normal(scale=0.1, size=(hidden_size_1 + 1, hidden_size_2))
    W3 = np.random.normal(scale=0.1, size=(hidden_size_1 + 1, output_size))

    N = y_train1.size

    train_errors1 = []
    test_errors1 = []
    train_errors2 = []
    test_errors2 = []

    weights1 = [W1, W2, W3]

    train_outs1 = []
    train_outs2 = []
    test_outs1 = []
    test_outs2 = []

    start_time = time.time()

    for i in range(epochs):

        # Feed forward
        train_outs1, dropouts = feed_forward_dropout(weights1, x_train1)

        # Backpropagation
        delta_layers1 = backpropagation_dropout(weights1, train_outs1, y_train1, dropouts)

        # Update weights
        weights1[0] -= learning_rate * np.dot(x_train1.T, delta_layers1[0]) / N
        for i in range(1, len(weights1)):
            weights1[i] -= learning_rate * np.dot(train_outs1[i - 1].T, delta_layers1[i]) / N

        # Calculate errors
        train_errors1.append(normalized_mse(train_outs1[-1], y_train1))
        test_errors1.append(normalized_mse(feed_forward(weights1, x_test)[-1], y_test))


    end_time = time.time()

    test_outs1 = feed_forward(weights1, x_test)

    acc = accuracy(test_outs1[-1], y_test)

    print(f"Time: {round(end_time - start_time, 2)}s")
    print("Accuracy With DropOut: {}".format(acc))

    W4 = np.random.normal(scale=0.1, size=(hidden_size_1 + 1, output_size))
    weights2 = [W1, W2, W4]
    N = y_train2.size


    for i in range(epochs):

        # Feed forward
        train_outs2, dropouts = feed_forward_dropout(weights2, x_train2)

        # Backpropagation
        delta_layers2 = backpropagation_dropout(weights2, train_outs2, y_train2, dropouts)

        # Update weights
        weights2[2] -= learning_rate * np.dot(train_outs2[1].T, delta_layers2[2]) / N

        # Calculate errors
        train_errors2.append(normalized_mse(train_outs2[-1], y_train2))
        test_errors2.append(normalized_mse(feed_forward(weights2, x_test)[-1], y_test))


    test_outs2 = feed_forward(weights2, x_test)

    acc = accuracy(test_outs2[-1], y_test)

    print("Accuracy of Transfer With DropOut: {}".format(acc))



    # Density plot sns
    sns.set(style="whitegrid")
    sns.set(rc={"figure.figsize": (10, 6)})
    sns.set(font_scale=1.5)
    sns.kdeplot(test_outs1[-1].flatten(), label="Training1")
    sns.kdeplot(test_outs2[-1].flatten(), label="Training2")
    sns.kdeplot(y_test.flatten(), label="Test")
    plt.legend()
    plt.title("Density plot")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # Train Test error plot
    sns.set(style="whitegrid")
    sns.set(rc={"figure.figsize": (10, 6)})
    sns.set(font_scale=1.5)
    sns.lineplot(x=range(epochs), y=train_errors1, label='Train1')
    sns.lineplot(x=range(epochs), y=test_errors1, label='Test1')
    sns.lineplot(x=range(epochs), y=train_errors2, label='Train2')
    sns.lineplot(x=range(epochs), y=test_errors2, label='Test2')
    plt.legend()
    plt.title(f"Mean Squared Error (acc: {round(acc, 3)})")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.show()


