import pandas as pd
from sklearn import linear_model
from tabulate import tabulate
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def corr_matrix(df):
    correlation_matrix = df.corr()

    ax = sns.heatmap(
        correlation_matrix,
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )
    ax.set_title('Correlation Matrix (WORST)')
    plt.show()

    print(tabulate(correlation_matrix, headers=correlation_matrix.columns, tablefmt='psql'))
    # loop over the matrix
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if correlation_matrix.iloc[i, j] >= 0.9:
                print(correlation_matrix.columns[i], 'and', correlation_matrix.columns[j],
                      'are highly correlated')
            else:
                print(correlation_matrix.columns[i], 'and', correlation_matrix.columns[j],
                      'are not')


def regression(df):

    avg_accuracy = 0
    for i in range(1000):
        # Shuffle all rows in dataframe
        df = df.sample(frac=1).reset_index(drop=True)

        X = df[['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10']]
        y = df['label']

        # Select first 500 rows for training
        X_train = X.head(500)
        y_train = y.head(500)

        # Select last 70 rows for testing
        X_test = X.tail(68)
        y_test = y.tail(68)

        regr = linear_model.LogisticRegression(solver='lbfgs', max_iter=1000)
        regr.fit(X_train, y_train)

        correct = 0
        # Test the remaining 68 rows and return the accuracy
        for i in range(len(X_test)):
            if regr.predict([X_test.iloc[i]]) == [y_test.iloc[i]]:
                correct += 1
        print('Accuracy:', correct / len(X_test), correct)
        avg_accuracy += correct / len(X_test)
    print('Average Accuracy:', avg_accuracy / 1000)


def pca(df, target_dim):
    x = df.drop('label', 1)
    scalar = StandardScaler()
    x = scalar.fit_transform(x)
    pca = PCA(n_components=target_dim)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data=principalComponents,
                               columns=[f"f{i}" for i in range(1, target_dim + 1)])
    finalDf = pd.concat([principalDf, df[['label']]], axis=1)

    # 3D plot
    if target_dim == 2:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        targets = ['M', 'B']
        colors = ['g', 'r']
        for target, color in zip(targets, colors):
            indicesToKeep = finalDf['label'] == target
            ax.scatter(finalDf.loc[indicesToKeep, 'f1'],
                       finalDf.loc[indicesToKeep, 'f2'],
                       # finalDf.loc[indicesToKeep, 'f3'],
                       c=color, s=5)
        ax.legend(targets)
        ax.grid()
        ax.set_xlabel('f1', fontsize=15)
        ax.set_ylabel('f2', fontsize=15)
        # ax.set_zlabel('f3', fontsize=15)
        ax.set_title('PCA from 30D to 2D', fontsize=20)
        # ax.view_init(30, 60)
        plt.show()

    return finalDf


if __name__ == '__main__':
    # output all lines
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    df = pd.read_csv('databases/wdbc.csv')

    # corr_matrix(df)
    # regression(df)
    reduced_df = pca(df, 2)
