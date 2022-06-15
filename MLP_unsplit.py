import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from preprocessing import pca
import time
import copy

NUM_FEATURES = 5
KEEP_RATE = 0.95


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
    df1 = pd.read_csv('databases/wdbc_split_1.csv')
    df2 = pd.read_csv('databases/wdbc_split_2.csv')
    df3 = pd.read_csv('databases/wdbc_split_3.csv')
    df1 = pca(df1, NUM_FEATURES)  # Reducing dimensions
    df2 = pca(df2, NUM_FEATURES)  # Reducing dimensions
    df3 = pca(df3, NUM_FEATURES)  # Reducing dimensions
    y = pd.get_dummies(df1.label).values
    y = y[:, 1]
    x1 = df1.drop('label', 1)
    x2 = df2.drop('label', 1)
    x3 = df3.drop('label', 1)
    x_train1, x_test1, y_train, y_test = train_test_split(x1, y, test_size=200, random_state=23)
    x_train2, x_test2, y_train, y_test = train_test_split(x2, y, test_size=200, random_state=23)
    x_train3, x_test3, y_train, y_test = train_test_split(x2, y, test_size=200, random_state=23)

    print(sum(y_train) + sum(y_test))

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    return x_train1, x_test1, x_train2, x_test2, x_train3, x_test3, y_train, y_test


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
    diff = (activation_layers[-1] - y_true)
    dot_one = activation_layers[-1] * diff
    delta_layers = [dot_one * (1 - activation_layers[-1])]

    for i in range(len(weights) - 1, 0, -1):
        a = np.dot(delta_layers[-1], weights[i][:-1].T)
        b = activation_layers[i - 1] * (1 - activation_layers[i - 1])
        delta_layers.append(a * b[:, :-1])
    delta_layers.reverse()
    return delta_layers


def backpropagation_dropout(weights, activation_layers, y_true, dropouts):
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


if __name__ == '__main__':

    input_size = NUM_FEATURES
    # hidden_size_1 = 7
    # hidden_size_2 = 15
    hidden_sizes = [2, 4, 8, 16, 32, 64, 128]
    output_size = 1
    learning_rate = 0.1
    epochs = 10000
    accuracies_wdrop = []
    accuracies_drop = []

    for hidden_size_1 in hidden_sizes:

        x_train1, x_test1, x_train2, x_test2, x_train3, x_test3, y_train, y_test = split_dataset()

        x_train1 = np.hstack((x_train1, np.ones((x_train1.shape[0], 1))))
        x_test1 = np.hstack((x_test1, np.ones((x_test1.shape[0], 1))))
        x_train2 = np.hstack((x_train2, np.ones((x_train2.shape[0], 1))))
        x_test2 = np.hstack((x_test2, np.ones((x_test2.shape[0], 1))))
        x_train3 = np.hstack((x_train3, np.ones((x_train3.shape[0], 1))))
        x_test3 = np.hstack((x_test3, np.ones((x_test3.shape[0], 1))))

        W11 = np.random.normal(scale=0.1, size=(input_size + 1, hidden_size_1))
        # W12 = np.random.normal(scale=0.1, size=(hidden_size_1 + 1, hidden_size_2))
        W13 = np.random.normal(scale=0.1, size=(hidden_size_1 + 1, output_size))
        W21 = np.random.normal(scale=0.1, size=(input_size + 1, hidden_size_1))
        # W22 = np.random.normal(scale=0.1, size=(hidden_size_1 + 1, hidden_size_2))
        W23 = np.random.normal(scale=0.1, size=(hidden_size_1 + 1, output_size))
        W31 = np.random.normal(scale=0.1, size=(input_size + 1, hidden_size_1))
        # W32 = np.random.normal(scale=0.1, size=(hidden_size_1 + 1, hidden_size_2))
        W33 = np.random.normal(scale=0.1, size=(hidden_size_1 + 1, output_size))

        N = y_train.size

        train_errors11 = []
        test_errors11 = []
        train_errors12 = []
        test_errors12 = []
        train_errors21 = []
        test_errors21 = []
        train_errors22 = []
        test_errors22 = []
        train_errors31 = []
        test_errors31 = []
        train_errors32 = []
        test_errors32 = []

        weights1 = [W11, W13]
        weights2 = [W21, W23]
        weights3 = [W31, W33]
        weights1_copy = copy.deepcopy(weights1)
        weights2_copy = copy.deepcopy(weights2)
        weights3_copy = copy.deepcopy(weights3)

        train_outs11 = []
        train_outs12 = []
        train_outs21 = []
        train_outs22 = []
        train_outs31 = []
        train_outs32 = []
        test_outs11 = []
        test_outs12 = []
        test_outs21 = []
        test_outs22 = []
        test_outs31 = []
        test_outs32 = []

        start_time = time.time()

        for i in range(epochs):

            # Feed forward
            train_outs11 = feed_forward(weights1, x_train1)
            train_outs12, dropouts1 = feed_forward_dropout(weights1_copy, x_train1)
            train_outs21 = feed_forward(weights2, x_train2)
            train_outs22, dropouts2 = feed_forward_dropout(weights2_copy, x_train2)
            train_outs31 = feed_forward(weights3, x_train3)
            train_outs32, dropouts3 = feed_forward_dropout(weights3_copy, x_train3)

            # Backpropagation
            delta_layers11 = backpropagation(weights1, train_outs11, y_train)
            delta_layers12 = backpropagation_dropout(weights1_copy, train_outs12, y_train, dropouts1)
            delta_layers21 = backpropagation(weights2, train_outs21, y_train)
            delta_layers22 = backpropagation_dropout(weights2_copy, train_outs22, y_train, dropouts2)
            delta_layers31 = backpropagation(weights3, train_outs31, y_train)
            delta_layers32 = backpropagation_dropout(weights3_copy, train_outs32, y_train, dropouts3)

            # Update weights
            weights1[0] -= learning_rate * np.dot(x_train1.T, delta_layers11[0]) / N
            for i in range(1, len(weights1)):
                weights1[i] -= learning_rate * np.dot(train_outs11[i - 1].T, delta_layers11[i]) / N

            weights1_copy[0] -= learning_rate * np.dot(x_train1.T, delta_layers12[0]) / N
            for i in range(1, len(weights1_copy)):
                weights1_copy[i] -= learning_rate * np.dot(train_outs12[i - 1].T, delta_layers12[i]) / N

            weights2[0] -= learning_rate * np.dot(x_train2.T, delta_layers21[0]) / N
            for i in range(1, len(weights2)):
                weights2[i] -= learning_rate * np.dot(train_outs21[i - 1].T, delta_layers21[i]) / N

            weights2_copy[0] -= learning_rate * np.dot(x_train2.T, delta_layers22[0]) / N
            for i in range(1, len(weights2_copy)):
                weights2_copy[i] -= learning_rate * np.dot(train_outs22[i - 1].T, delta_layers22[i]) / N

            weights3[0] -= learning_rate * np.dot(x_train3.T, delta_layers31[0]) / N
            for i in range(1, len(weights3)):
                weights3[i] -= learning_rate * np.dot(train_outs31[i - 1].T, delta_layers31[i]) / N

            weights3_copy[0] -= learning_rate * np.dot(x_train3.T, delta_layers32[0]) / N
            for i in range(1, len(weights3_copy)):
                weights3_copy[i] -= learning_rate * np.dot(train_outs32[i - 1].T, delta_layers32[i]) / N

            # Calculate errors
            train_errors11.append(normalized_mse(train_outs11[-1], y_train))
            test_errors11.append(normalized_mse(feed_forward(weights1, x_test1)[-1], y_test))
            train_errors21.append(normalized_mse(train_outs21[-1], y_train))
            test_errors21.append(normalized_mse(feed_forward(weights2, x_test2)[-1], y_test))
            train_errors31.append(normalized_mse(train_outs31[-1], y_train))
            test_errors31.append(normalized_mse(feed_forward(weights3, x_test3)[-1], y_test))

            train_errors12.append(normalized_mse(train_outs12[-1], y_train))
            test_errors12.append(normalized_mse(feed_forward(weights1_copy, x_test1)[-1], y_test))
            train_errors22.append(normalized_mse(train_outs22[-1], y_train))
            test_errors22.append(normalized_mse(feed_forward(weights2_copy, x_test2)[-1], y_test))
            train_errors32.append(normalized_mse(train_outs32[-1], y_train))
            test_errors32.append(normalized_mse(feed_forward(weights3_copy, x_test3)[-1], y_test))

        end_time = time.time()

        test_outs11 = feed_forward(weights1, x_test1)
        test_outs12 = feed_forward(weights1_copy, x_test1)
        test_outs21 = feed_forward(weights2, x_test2)
        test_outs22 = feed_forward(weights2_copy, x_test2)
        test_outs31 = feed_forward(weights3, x_test3)
        test_outs32 = feed_forward(weights3_copy, x_test3)

        # print(test_outs2[-1])

        means_acc_wdrop = accuracy(test_outs11[-1], y_test)
        accuracies_wdrop.append(means_acc_wdrop)
        print(f"Time: {round(end_time - start_time, 2)}s")
        print("Accuracy 1 Without DropOut: {}".format(means_acc_wdrop))

        means_acc_drop = accuracy(test_outs12[-1], y_test)
        accuracies_drop.append(means_acc_drop)
        print(f"Time: {round(end_time - start_time, 2)}s")
        print("Accuracy 1 With DropOut: {}".format(means_acc_drop))

        sds_acc_wdrop = accuracy(test_outs21[-1], y_test)

        print(f"Time: {round(end_time - start_time, 2)}s")
        print("Accuracy 2 Without DropOut: {}".format(sds_acc_wdrop))

        sds_acc_drop = accuracy(test_outs22[-1], y_test)

        print(f"Time: {round(end_time - start_time, 2)}s")
        print("Accuracy 2 With DropOut: {}".format(sds_acc_drop))

        worst_acc_wdrop = accuracy(test_outs31[-1], y_test)

        print(f"Time: {round(end_time - start_time, 2)}s")
        print("Accuracy 3 Without DropOut: {}".format(worst_acc_wdrop))

        worst_acc_drop = accuracy(test_outs32[-1], y_test)

        print(f"Time: {round(end_time - start_time, 2)}s")
        print("Accuracy 3 With DropOut: {}".format(worst_acc_drop))

    # Density plot sns
    # sns.set(style="whitegrid")
    # sns.set(rc={"figure.figsize": (10, 6)})
    # sns.set(font_scale=1.5)
    # sns.kdeplot(test_outs1[-1].flatten(), label="Training1")
    # sns.kdeplot(test_outs2[-1].flatten(), label="Training2")
    # sns.kdeplot(y_test.flatten(), label="Test")
    # plt.legend()
    # plt.title("Density plot")
    # plt.xlabel("Predicted")
    # plt.ylabel("True")
    # plt.show()

    # Train Test error plot
    # sns.set(style="whitegrid")
    # sns.set(rc={"figure.figsize": (10, 6)})
    # sns.set(font_scale=1.5)
    # sns.lineplot(x=range(epochs), y=train_errors1, label='Train1')
    # sns.lineplot(x=range(epochs), y=test_errors1, label='Test1')
    # sns.lineplot(x=range(epochs), y=train_errors2, label='Train2')
    # sns.lineplot(x=range(epochs), y=test_errors2, label='Test2')
    # plt.legend()
    # plt.title(f"Mean Squared Error (acc: {round(acc, 3)})")
    # plt.xlabel("Epochs")
    # plt.ylabel("MSE")
    # plt.show()

    # sns.set(style="whitegrid")
    # sns.set(rc={"figure.figsize": (10, 6)})
    # sns.set(font_scale=1.5)
    # sns.pointplot(x=hidden_sizes, y=[0, 1], data=accuracies_wdrop)
    # sns.pointplot(x="time", y="total_bill", data=accuracies_drop)
    plt.scatter(x=hidden_sizes, y=[accuracies_wdrop])
    plt.scatter(x=hidden_sizes, y=[accuracies_drop], c="red")
    plt.legend()
    # plt.title(f"Mean Squared Error (acc: {round(acc, 3)})")
    plt.xlabel("Hidden Sizes")
    plt.ylabel("Accuracy")
    plt.show()