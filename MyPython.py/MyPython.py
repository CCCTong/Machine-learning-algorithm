import numpy as np
import matplotlib.pyplot as plt
import math
import random


def hypothesis_func(theta, x):
    return x.dot(theta.T)


def loss_func(x, y, theta):
    n, m = x.shape
    ans = 0
    for i in range(n):
        ans += (hypothesis_func(theta, x[i]) - y[i]) ** 2
    return ans / (2 * n)


def get_grad(theta, j, x, y):
    n, m = x.shape
    ans = 0
    for i in range(n):
        ans += (hypothesis_func(theta, x[i]) - y[i]) * x[i][j]
    return ans / n


def grad_descent(train_x, train_y, alpha=0.1):
    n, m = train_x.shape
    theta = np.zeros(m)
    it, k, eps, last = 0, 0, 1e-10, 0
    while 1:
        it = it + 1
        tmp = np.zeros_like(theta)
        for j in range(m):
            tmp[j] = theta[j] - alpha * get_grad(theta, j, train_x, train_y)
        value = loss_func(train_x, train_y, theta)
        print("iterator times =", it, "loss = ", value)
        theta = tmp.copy()
        if math.fabs(value - last) < eps:
            break
        last = value
    return theta


def norm(train):
    n, m = train.shape
    ans = np.zeros_like(train)
    for j in range(m):
        feature = []
        for i in range(n):
            feature.append(train[i][j])
            ans[i][j] = train[i][j]
        feature = np.array(feature)
        mean, var = np.mean(feature), np.var(feature)
        if var == 0:
            continue
        for i in range(n):
            ans[i][j] = (train[i][j] - mean) / math.sqrt(var)
    return ans


def rnorm(train, theta):
    n, m = train.shape
    ans = np.zeros_like(theta)
    for j in range(m):
        feature = []
        ans[j] = theta[j]
        for i in range(n):
            feature.append(train[i][j])
        feature = np.array(feature)
        mean, var = np.mean(feature), np.var(feature)
        if var == 0 or j == 0:
            continue
        ans[0] -= theta[j] * mean / math.sqrt(var)
        ans[j] = theta[j] / math.sqrt(var)
    return ans


def norm_equation(x, y):
    theta = np.linalg.inv(np.dot(x.T, x))
    theta = np.dot(theta, x.T)
    return np.dot(theta, y)


def load_data(path):
    data = open(path, "r")
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    cur_x = []
    cur_y = []
    for line in data:
        cur_line = line.strip().split(",")
        float_line = list(map(float, cur_line))
        float_line.insert(0, 1)
        length = len(float_line)
        cur_x.append(float_line[0:length-2])
        cur_y.append(float_line[length-1])
    len_x = len(cur_x)
    len_y = len(cur_y)
    for i in range(len_x):
        if i & 1 == 1:
            train_x.append(cur_x[i])
            train_y.append(cur_y[i])
        else:
            test_x.append(cur_x[i])
            test_y.append(cur_y[i])
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    norm_train_x = norm(train_x)
    norm_test_x = norm(test_x)
    return train_x, train_y, test_x, test_y, norm_train_x, norm_test_x


def work():
    np.set_printoptions(suppress=True)
    train_x, train_y, test_x, test_y, norm_train_x, norm_test_x = load_data("C:/Users/boliz/Desktop/housing.txt")
    theta1 = grad_descent(norm_train_x, train_y)
    theta1 = rnorm(train_x, theta1)
    print(theta1)
    predict_y1 = np.dot(test_x, theta1)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(np.arange(1, test_y.shape[0]+1, 1), test_y, c='r', marker='o')
    ax1.scatter(np.arange(1, test_y.shape[0]+1, 1), predict_y1, c='b', marker='x')
    plt.show()
    return


if __name__ == '__main__':
    work()