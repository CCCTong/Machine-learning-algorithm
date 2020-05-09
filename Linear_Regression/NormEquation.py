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


def norm(train):
    mean = np.mean(train)
    var = np.var(train)
    train = (train - mean) / math.sqrt(var)
    return train


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
    return train_x, train_y, test_x, test_y


def norm_equation(x, y):
    theta = np.linalg.inv(np.dot(x.T, x))
    theta = np.dot(theta, x.T)
    return np.dot(theta, y)


def work():
    random.seed(1000000007)
    np.set_printoptions(suppress=True)
    train_x, train_y, test_x, test_y = load_data("C:/Users/boliz/Desktop/housing.txt")
    theta = norm_equation(train_x, train_y)
    predict_y = np.dot(test_x, theta)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(np.arange(1, test_y.shape[0]+1, 1), test_y, c='r', marker='o')
    ax1.scatter(np.arange(1, test_y.shape[0]+1, 1), predict_y, c='b', marker='x')
    plt.show()
    print(loss_func(test_x, test_y, theta))
    print(theta)
    return


if __name__ == '__main__':
    work()