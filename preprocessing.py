import pandas as pd
from sklearn import linear_model
from tabulate import tabulate
import seaborn as sns
import matplotlib.pyplot as plt


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
    ax.set_title('Correlation Matrix')
    ax.set_xlabel('Features')
    ax.set_ylabel('Features')
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


def linear_regression(df):
    X = df[['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10']]
    y = df['label']

    regr = linear_model.LogisticRegression(solver='lbfgs', max_iter=1000)
    regr.fit(X, y)

    # These are taken from the dataset
    pred_M = regr.predict(
        [[17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871]])
    pred_B = regr.predict(
        [[0.2271, 1.255, 1.441, 16.16, 0.005969, 0.01812, 0.02007, 0.007027, 0.01972, 0.002607]])
    print(f"\nPredicted M: {pred_M}\nPredicted B: {pred_B}")

    coefficients = regr.coef_
    print(f"Coefficients: {coefficients}")


if __name__ == '__main__':
    # output all lines
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    df = pd.read_csv('databases/wdbc_split.csv')

    corr_matrix(df)
    linear_regression(df)
