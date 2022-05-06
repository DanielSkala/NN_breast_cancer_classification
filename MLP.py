import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from preprocessing import pca
import time

# TODO: normalize input data - Medhat
# TODO: regularization, dropout - Bianca
# TODO: trainable bias - Mitja
# TODO: PCA - Daniel

NUM_FEATURES = 5


def sigmoid(x):
    warnings.filterwarnings('ignore')
    return 1 / (1 + np.exp(-x))


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
    df = pca(df, NUM_FEATURES)  # Reducing dimensions
    y = pd.get_dummies(df.label).values
    y = y[:, 0]
    x = df.drop('label', 1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=200, random_state=23)

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    return x_train, x_test, y_train, y_test


def feed_forward(weights, inputs):
    activation_layers = [sigmoid(np.dot(inputs, weights[0]))]
    for i in range(1, len(weights)):
        activation_layers[i - 1] = np.hstack(
            (activation_layers[i - 1], np.ones((activation_layers[i - 1].shape[0], 1))))
        activation_layers.append(sigmoid(np.dot(activation_layers[i - 1], weights[i])))
    return activation_layers


def backpropagation(weights, activation_layers, y_true):
    diff = (activation_layers[-1] - y_true)
    dot_one = activation_layers[-1] * diff
    delta_layers = [dot_one * (1 - activation_layers[-1])]

    for i in range(len(weights) - 1, 0, -1):
        a = np.dot(delta_layers[-1], weights[i][:-1].T)
        b = activation_layers[i - 1] * (1 - activation_layers[i - 1])
        delta_layers.append(a * b[:, :-1])
    delta_layers.reverse()
    return delta_layers


if __name__ == '__main__':

    input_size = NUM_FEATURES
    hidden_size_1 = 15
    # hidden_size_2 = 15
    output_size = 1
    learning_rate = 1.8
    epochs = 25000

    x_train, x_test, y_train, y_test = split_dataset()

    x_train = np.hstack((x_train, np.ones((x_train.shape[0], 1))))
    x_test = np.hstack((x_test, np.ones((x_test.shape[0], 1))))

    W1 = np.random.normal(scale=0.1, size=(input_size + 1, hidden_size_1))
    # W2 = np.random.normal(scale=0.1, size=(hidden_size_1 + 1, hidden_size_2))
    W3 = np.random.normal(scale=0.1, size=(hidden_size_1 + 1, output_size))

    N = y_train.size

    train_errors = []
    test_errors = []

    weights = [W1, W3]

    train_outs = []
    test_outs = []

    start_time = time.time()

    for i in range(epochs):

        # Feed forward
        train_outs = feed_forward(weights, x_train)

        # Backpropagation
        delta_layers = backpropagation(weights, train_outs, y_train)

        # Update weights
        weights[0] -= learning_rate * np.dot(x_train.T, delta_layers[0]) / N
        for i in range(1, len(weights)):
            weights[i] -= learning_rate * np.dot(train_outs[i - 1].T, delta_layers[i]) / N

        # Calculate errors
        train_errors.append(normalized_mse(train_outs[-1], y_train))
        test_errors.append(normalized_mse(feed_forward(weights, x_test)[-1], y_test))

    end_time = time.time()

    test_outs = feed_forward(weights, x_test)

    acc = accuracy(test_outs[-1], y_test)

    print(f"Time: {round(end_time - start_time, 2)}s")
    print("Accuracy: {}".format(acc))

    # Density plot sns
    sns.set(style="whitegrid")
    sns.set(rc={"figure.figsize": (10, 6)})
    sns.set(font_scale=1.5)
    sns.kdeplot(test_outs[-1].flatten(), label="Training")
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
    sns.lineplot(x=range(epochs), y=train_errors, label='Train')
    sns.lineplot(x=range(epochs), y=test_errors, label='Test')
    plt.legend()
    plt.title(f"Mean Squared Error (acc: {round(acc, 3)})")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.show()
