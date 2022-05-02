import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


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


if __name__ == '__main__':

    input_size = 10
    hidden_size = 100
    output_size = 1
    learning_rate = 0.008
    iterations = 20000

    df = pd.read_csv('databases/wdbc_split.csv')
    y = pd.get_dummies(df.label).values
    y = y[:, 0]
    x = df = df.drop('label', 1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=80, random_state=8)

    # plt.plot(y_train)
    # plt.plot(y_test)
    # plt.show()

    W1 = np.random.normal(scale=0.5, size=(input_size, hidden_size))
    W2 = np.random.normal(scale=0.5, size=(hidden_size, output_size))
    N = y_train.size

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    train_errors = []
    test_errors = []

    for itr in range(iterations):
        Z1 = np.dot(x_train, W1)
        A1 = sigmoid(Z1)

        Z2 = np.dot(A1, W2)
        A2 = sigmoid(Z2)

        mse = normalized_mse(A2, y_train)
        # acc = accuracy(A2, y_train)
        # results=results.append({"mse":mse, "accuracy":acc},ignore_index=True )
        # print(mse)
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

        acc = normalized_mse(A2, y_test)
        test_errors.append(acc)

    Z1 = np.dot(x_test, W1)
    A1 = sigmoid(Z1)

    Z2 = np.dot(A1, W2)
    A2 = sigmoid(Z2)
    print(np.mean(A2))

    acc = accuracy(A2, y_test)
    print("Accuracy: {}".format(acc))

    # Density plot
    sns.kdeplot(A2.flatten(), label='Train')
    sns.kdeplot(y_test.flatten(), label='Test')
    plt.legend()
    plt.title("Mean Squared Error")
    plt.show()

    plt.plot(train_errors, label='Train')
    plt.plot(test_errors, label='Test')
    plt.legend()
    plt.title("Mean Squared Error")
    plt.show()
