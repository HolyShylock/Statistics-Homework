import numpy as np
import pandas as pd
import scipy.optimize as opt
import math
def sigmoid(z):
    return 1 / ( 1 + np.exp(-z))
def cost(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    part1 = np.multiply(-y, np.log(sigmoid(X * theta)))
    part2 = np.multiply(y - 1, np.log(1 - sigmoid(X * theta)))
    return np.sum(part1 + part2) / len(X)
def Cost(theta, X, y, k):
    theta = np.matrix(theta)
    m = len(X)
    X = np.matrix(X)
    y = np.matrix(y)
    part1 = np.multiply(-y, np.log(sigmoid(X * theta)))
    part2 = np.multiply(y - 1, np.log(1 - sigmoid(X * theta)))
    part3 = np.multiply(theta, theta)
    return np.sum(part1 + part2) / m + (k / (2 * m)) * np.sum(part3)
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
def gradient(theta, X, y):
    error = sigmoid(X * theta) - y
    return (X.T * error) / len(X)
def gradientDescent(theta, X, y, alpha, iters):
    m = len(X)
    n = len(theta)
    theta = np.matrix(theta)
    temp = np.matrix(np.zeros(theta.shape))
    X = np.matrix(X)
    y = np.matrix(y)
    for i in range(0, iters):
        h = sigmoid(X * theta) - y
        for j in range(0, n):
            term = np.multiply(h, X[ : , j])
            temp[j, 0] = temp[j, 0] - (alpha / m) * np.sum(term)
        theta = temp
        print(cost(theta, X, y))
    return theta
def regularGradientDescent(theta, k, alpha, X, y, iters):
    m = len(X)
    n = len(theta)
    theta = np.matrix(theta)
    temp = np.matrix(np.zeros(theta.shape))
    X = np.matrix(X)
    y = np.matrix(y)
    for i in range(0, iters):
        h = sigmoid(X * theta) - y
        for j in range(0, n):
            term = np.multiply(h, X[ : , j])
            temp[j, 0] = temp[j, 0] - (alpha / m) * np.sum(term) - (k * alpha / m) * temp[j, 0]
        theta = temp
        print(Cost(theta, X, y, k))
    return theta
def get(df, ratio = 0.7):
    n = len(df)
    np.random.seed(3)
    index = np.random.permutation(n)
    ones = pd.DataFrame({'ones': np.ones(n)})
    data = pd.concat([ones, df], axis = 1)
    X = np.array(data.iloc[ : , : -2])
    y = np.array(data.iloc[ : , -1])
    trainSize = int(n * ratio)
    X_train = X[index[ : trainSize]]
    X_test = X[index[trainSize + 1 : -1]]
    y_train = y[index[ : trainSize]]
    y_test = y[index[trainSize + 1 : ]]
    return X_train, X_test, y_train, y_test
def prediction(theta, X, y):
    p = sigmoid(X * theta)
    m = len(p)
    for i in range(0, m):
        p[i] = p[i] >= 0.5
    print(p)
data = pd.read_csv('heart_failure_clinical_records_dataset.csv')
'''X = get_X(data)
print(X.shape)
print(X)
y = get_y(data)
print(y.shape)
theta = np.matrix(np.zeros((1, 12)))
print(cost(theta, X, y))
print(gradient(theta, X, y))
print(descent(theta, X, y, 0.01, 1000))
print(cost(theta, X, y))'''
"""X_train, y_train, X_test, y_test = get(data)
print(X_train)
print(y_train)"""
X_train, X_test, y_train, y_test = get(data)
#X_train = normX(X_train)
#print(X_train)
#y_train = normy(y_train)
X_train, y_train = norm(X_train, y_train)
theta = np.zeros((12, 1))
k = 0.5
alpha = 0.01
iters = 1000
print(Cost(theta, X_train, y_train, k))
theta = regularGradientDescent(theta, k, alpha, X_train, y_train, iters)
print(theta)
#print(sigmoid(X_test * theta))
#print(y_test)
#prediction(theta, X_test, y_test)
#print(cost(theta, X_train, y_train))
#res = opt.minimize(fun=cost, x0=np.array(theta), args=(X_train, y_train), method='Newton-CG', jac=gradient)
#print(res)