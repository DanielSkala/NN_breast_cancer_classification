import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derivative_logistic_sigmoid(x):
    return x * (1 - x)


def feedforward(weights, layers, input):
    x = logistic_sigmoid(np.dot(W[0], x) + 1)
    for i in range(1, layers):
        x = logistic_sigmoid(np.dot(W[i], x) + 1)
    return x


def mean_squared_error(y_pred, y_true):
    return ((y_pred - y_true) ** 2).sum() / (2 * y_pred.size)


def accuracy(y_pred, y_true):
    acc = y_pred.argmax(axis=1) == y_true.argmax(axis=1)
    return acc.mean()


if __name__ == '__main__':

    input_size = 10
    hidden_size = 3
    output_size = 1
    learning_rate = 0.1
    iterations = 5000

    df = pd.read_csv('databases/wdbc_split.csv')
    y = pd.get_dummies(df.label).values
    y = y[:, 0]
    x = df = df.drop('label', 1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=40, random_state=4)

    W1 = np.random.normal(scale=0.5, size=(input_size, hidden_size))
    W2 = np.random.normal(scale=0.5, size=(hidden_size, output_size))
    N = y_train.size

    y_train = y_train.reshape(N, 1)

    train_errors = []

    for itr in range(iterations):
        Z1 = np.dot(x_train, W1)
        A1 = sigmoid(Z1)

        Z2 = np.dot(A1, W2)
        A2 = sigmoid(Z2)

        mse = mean_squared_error(A2, y_train)
        # acc = accuracy(A2, y_train)
        # results=results.append({"mse":mse, "accuracy":acc},ignore_index=True )
        print(mse)
        train_errors.append(mse)

        E1 = A2 - y_train
        # dW1 = np.multiply(E1, np.multiply(A2, 1 - A2))
        dW1 = E1 * A2 * (1 - A2)

        E2 = np.dot(dW1, W2.T)
        dW2 = E2 * A1 * (1 - A1)

        W2_update = np.dot(A1.T, dW1) / N
        W1_update = np.dot(x_train.T, dW2) / N

        W2 = W2 - learning_rate * W2_update
        W1 = W1 - learning_rate * W1_update

    Z1 = np.dot(x_test, W1)
    A1 = sigmoid(Z1)

    Z2 = np.dot(A1, W2)
    A2 = sigmoid(Z2)

    acc = mean_squared_error(A2, y_test)
    print("Accuracy: {}".format(acc))

    plt.plot(train_errors)
    plt.title("Mean Squared Error")
    plt.show()
