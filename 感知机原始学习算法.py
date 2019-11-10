#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np

def Load_data(filename):
    temp = np.loadtxt(filename)
    data = temp[:,0:2]
    label = temp[:,2]
    return data, label

def train(data,label,alpha):
    m,n = data.shape
    w = np.zeros(n)
    b = 0
    iterator_times = 0
    tag = True
    while(tag):
        for i in range(m):
            if (label[i] * (np.dot(w,data[i])) <= 0):
                w = w + alpha * label[i] * data[i].T
                b = b + alpha * label[i]
                tag = false
                break
            else:
                tag = False
        iterator_times += 1
        print("iterator_times = {0}  w = {1}  b = {2}".format(iterator_times,w,b))
    return w,b
if __name__ == "__main__":
    data,label = Load_data("/Users/chutong/Desktop/Machine\ Learning\ Algorithm/data.txt")
    print(data)
    print(label)
