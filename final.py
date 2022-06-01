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

TEST_SIZE = 57

def split_dataset():
    df = pd.read_csv('databases/wdbc_unsplit_norm.csv')
    #df = pca(df, NUM_FEATURES)  # Reducing dimensions
    y = pd.get_dummies(df.label).values
    y = y[:, 1]
    x = df.drop('label', 1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=23)

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    return x_train, x_test, y_train, y_test

def pca(x_train, x_test, num_features):
	X = x_train + x_test
	scalar = StandardScaler()
	X = scalar.fit_transform(X)
	pca = PCA(n_components=num_features)
	principalComponents = pca.fit_transform(X)
	finalDf = pd.DataFrame(data=principalComponents,
							   columns=[f"f{i}" for i in range(1, num_features + 1)])
	return finalDf[:-TEST_SIZE], finalDf[-TEST_SIZE:]

x_train_raw, x_test_raw, y_train, y_test = split_dataset()

for num_features in range(2, 30, 3):
	x_train, x_test = pca(x_train_raw, x_test_raw, num_features)
	for hidden_size_1 in [1, 2, 4, 8, 16, 32]:
		for hidden_size_2 in [0, 1, 2, 4, 8, 16, 32]:
			if hidden_size_1 == 0:
				W1 = np.random.normal(scale=0.1, size=(num_features + 1, hidden_size_1))
    			W2 = np.random.normal(scale=0.1, size=(hidden_size_1 + 1, 1))
    			weights = [W1, W2]
    		else:
    			W1 = np.random.normal(scale=0.1, size=(input_size + 1, hidden_size_1))
    			W2 = np.random.normal(scale=0.1, size=(hidden_size_1 + 1, hidden_size_2))
    			W3 = np.random.normal(scale=0.1, size=(hidden_size_1 + 1, output_size))
    			weights = [W1, W2, W3]
    		print("*****************************************")
    		print("\nFor a two layer NN with first layer being of " +  str(hidden_size_1) + " neurons and the second layer being of " +  str(hidden_size_2) + " neurons:\n")
    		for keep_rate in range(0.8, 1, 0.05):
    			print("For a keep_rate of " + str(keep_rate) + " :\n")
    			for learning_rate in [0.01, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3]:
    				print("For a learning rate of " + str(learning_rate) + "\n")
    				for epochs in [500, 2000, 4000, 8000, 10000]:
    					print("Epoch: " + str(epochs))
						final_weights = train_model(weights, keep_rate, learning_rate, epochs, x_train, y_train)
						final_accuracy = accuracy(final_weights, x_test, y_test)
						print("The final accuracy is: " + str(final_accuracy) + "\n")

print("######################################")
print("######################################")
print("######################################")
print("ITS TRANSFER TIME!!!!")

# do transfer
	