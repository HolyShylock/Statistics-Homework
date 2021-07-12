import numpy as np
import pandas as pd
import math
from scipy.special import expit

def get(df, ratio = 0.7):
    n = len(df)
    np.random.seed(19)
    index = np.random.permutation(n)
    ones = pd.DataFrame({'ones': np.ones(n)})
    data = pd.concat([ones, df], axis = 1)
    X = np.array(data.iloc[ : , : -1])
    y = np.array(data.iloc[ : , -1])
    trainSize = int(n * ratio)
    X_train = X[index[ : trainSize]]
    X_test = X[index[trainSize + 1 : ]]
    y_train = y[index[ : trainSize]]
    y_test = y[index[trainSize + 1 : ]]
    return X_train, X_test, y_train, y_test

def norm(X, y):
    X = np.matrix(X)
    m = X.shape[0]
    n = X.shape[1]
    for i in range(1, n):
        average = np.sum(X[ : , i]) / m
        error = np.multiply(X[ : , i] - average, X[ : , i] - average)
        std = math.sqrt(np.sum(error) / m)
        X[ : , i] = (X[ : ,i] - average) / std
    average = np.sum(y) / m
    error = np.multiply(y - average, y - average)
    std = math.sqrt(np.sum(error) / m)
    y = (y - average) / std
    return X, y

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost(X, y, theta, regular):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    m = len(X)
    part1 = np.multiply(-y, np.log(sigmoid(X * theta)))
    part2 = np.multiply(y - 1, np.log(1 - sigmoid(X * theta)))
    if regular == False:
        return np.sum(part1 + part2) / m
    else:
        k = 0.05
        part3 = np.multiply(theta, theta)
        return np.sum(part1 + part2) / m + (k / (2 * m)) * np.sum(part3)

def gradDescent(X, y, theta, alpha = 0.01, iters = 100, regular = False):
    X = np.matrix(X)
    m = X.shape[0]
    y = np.matrix(y).reshape(m, 1)
    k = 0.05
    for i in range(0, iters):
        error = sigmoid(X * theta) - y
        grad = X.T * error / m
        if regular == False:
            theta = theta - alpha * grad
        else:
            theta = theta - alpha * grad - (k / m) * theta
        print(cost(X, y, theta, regular))
    return theta

def prediction(X, y, theta):
    y_pre = X * theta
    m = len(y)
    count = 0
    for i in range(0, m):
        if y_pre[i] >= 0:
            y_pre[i] = 1
            if y[i] == 1:
                count += 1
        else:
            y_pre[i] = 0
            if y[i] == 0:
                count += 1
    res = "accuracy rate : " + str(float(count / m))
    print(res)
    return y_pre

'''def get_test(df):
    n = len(df)
    ones = pd.DataFrame({'ones': np.ones(n)})
    data = pd.concat([ones, df], axis = 1)
    X = np.matrix(data.iloc[ : 11, : -1])
    y = np.matrix(data.iloc[ : 11, -1]).reshape(11, 1)
    return X, y'''

data = pd.read_csv('heart_failure_clinical_records_dataset.csv')
X_train, X_test, y_train, y_test = get(data)
X_train, y_train = norm(X_train, y_train)
m = X_train.shape[0]
n = X_train.shape[1]
theta = np.zeros((n, 1))
alpha = 0.05
iters = 100
regular = False
theta = gradDescent(X_train, y_train, theta, alpha, iters, regular)
print(theta)
y_pre = prediction(X_test, y_test, theta)
print(y_pre)