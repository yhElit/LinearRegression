import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
import statsmodels.api as sm
import seaborn as sns


def mean_squared_error_self(y_test, y_predict):
    actual = np.array(y_test)
    predicted = np.array(y_predict)
    differences = np.subtract(actual, predicted)
    squared_differences = np.square(differences)
    return squared_differences.mean()


def r2_score_self(y_test, y_predict):
    model_rss = np.sum((y_predict - y_test) ** 2)
    mean_rss = np.sum((np.mean(y_test) - y_test) ** 2)
    r2_manual = 1 - (model_rss / mean_rss)
    return r2_manual


def main():
    df = pd.read_csv("winequality-red.csv")

    # Create target
    y = df.pop("fixed acid").values.reshape(-1, 1)

    # Create feature
    df_data = df
    x = df.pop("citric acid").values.reshape(-1, 1)

    # Split dataset in random training and testing datasets
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75)

    # Create classifier
    clf = LinearRegression()

    # Train model
    clf.fit(x_train, y_train)

    # predict
    y_predict = clf.predict(x_test)

    # library
    MSE = mean_squared_error(y_test, y_predict)
    R2 = r2_score(y_test, y_predict)

    # self
    MSE_self = mean_squared_error_self(y_test, y_predict)
    R2_self = r2_score_self(y_test, y_predict)

    print("MSE library:", MSE)
    print("MSE self:", MSE_self)
    print("R2 library:", R2)
    print("R2 self:", R2_self)

    # Plot the histogram.
    # plt.hist(x, bins=25, density=True, alpha=0.6, color='b')
    # mu, std = norm.fit(x)

    # Plot the Distribution
    # xmin, xmax = plt.xlim()
    # x = np.linspace(xmin, xmax, 100)
    # p = norm.pdf(x, mu, std)
    # plt.plot(x, p, 'k', linewidth=2)
    # plt.show()

    data = sm.add_constant(x_test)
    targets = y_test
    ols = sm.OLS(targets, data).fit()
    print(ols.summary())

    data = sm.add_constant(df_data)
    targets = y
    ols = sm.OLS(targets, data).fit()
    print(ols.summary())

    # plot
    plt.plot(x_test, y_predict, label="Linear Regression", color="r")
    plt.scatter(x_test, y_predict, label="prediction", color="r")
    plt.scatter(x_test, y_test, label="data", color="g")
    plt.xlabel("citric acid")
    plt.ylabel("fixed acid")
    plt.legend()
    plt.show()

    residuals = []
    for i in range(len(y_predict)):
        residuals.append(y_test[i][0] - y_predict[i][0])

    res = sns.displot(data=residuals, kde=True)
    plt.xlabel("citric acid")
    plt.show()


if __name__ == '__main__':
    main()
