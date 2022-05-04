import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


# TODO: normalize input data - Medhad
# TODO: regularization, dropout - Bianca
# TODO: trainable bias - Mitja
# TODO: PCA - Daniel

def sigmoid(x):
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
    y = pd.get_dummies(df.label).values
    y = y[:, 0]
    x = df.drop('label', 1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=80, random_state=42)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    return x_train, x_test, y_train, y_test


def feed_forward(weights, inputs):
    activation_layers = [sigmoid(np.dot(inputs, weights[0]))]
    for i in range(1, len(weights)):
        activation_layers.append(sigmoid(np.dot(activation_layers[i - 1], weights[i])))
    return activation_layers


def backpropagation(weights, activation_layers, y_true):
    diff = (activation_layers[-1] - y_true)
    dot_one = activation_layers[-1] * diff
    delta_layers = [dot_one * (1 - activation_layers[-1])]

    for i in range(len(weights) - 1, 0, -1):
        delta_layers.append(np.dot(delta_layers[-1], weights[i].T) * activation_layers[i - 1] *
                            (1 - activation_layers[i - 1]))
    delta_layers.reverse()
    return delta_layers


if __name__ == '__main__':

    input_size = 10
    hidden_size_1 = 15
    hidden_size_2 = 15
    output_size = 1
    learning_rate = 0.008
    epochs = 2000

    x_train, x_test, y_train, y_test = split_dataset()

    W1 = np.random.normal(scale=0.5, size=(input_size, hidden_size_1))
    W2 = np.random.normal(scale=0.5, size=(hidden_size_1, hidden_size_2))
    W3 = np.random.normal(scale=0.5, size=(hidden_size_2, output_size))

    N = y_train.size

    train_errors = []
    test_errors = []

    weights = [W1, W2, W3]

    train_outs = []
    test_outs = []

    for i in range(epochs):
        # Feed forward
        train_outs = feed_forward(weights, x_train)
        test_outs = feed_forward(weights, x_test)

        # Backpropagation
        delta_layers = backpropagation(weights, train_outs, y_train)

        # Update weights
        weights[0] -= learning_rate * np.dot(x_train.T, delta_layers[0]) / N
        for i in range(1, len(weights)):
            weights[i] -= learning_rate * np.dot(train_outs[i - 1].T, delta_layers[i]) / N

        # Calculate errors
        train_errors.append(normalized_mse(train_outs[-1], y_train))
        test_errors.append(normalized_mse(feed_forward(weights, x_test)[-1], y_test))

    acc = accuracy(train_outs[-1], y_train)
    print("Accuracy: {}".format(acc))

    # Density plot
    sns.kdeplot(test_outs[-1].flatten(), label='Train')
    sns.kdeplot(y_test.flatten(), label='Test')
    plt.legend()
    plt.title("Mean Squared Error")
    plt.show()

    plt.plot(train_errors, label='Train')
    plt.plot(test_errors, label='Test')
    plt.legend()
    plt.title("Mean Squared Error")
    plt.show()
