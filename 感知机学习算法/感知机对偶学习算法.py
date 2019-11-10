#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np

def Load_data(filename,K):
    temp = np.loadtxt(filename)
    data = temp[:,0:K]
    label = temp[:,K]
    return data, label

def train(data,label,alpha):
    m,n = data.shape
    w = np.zeros(m)
    b = 0
    iterator_times = 0
    tag = True
    while(tag):
        tmp = np.zeros(n)
        for i in range(m):
            tmp += w[i] * data[i] * label[i]
        for i in range(m):
            if (label[i] * (np.dot(tmp,data[i]) + b) <= 0):
                w[i] = w[i] + alpha
                b = b + alpha * label[i]
                tag = True
                break
            else:
                tag = False
        iterator_times += 1
        print("iterator_times = {0}  w = {1}  b = {2}".format(iterator_times,w,b))
    #print(data)
    tmp = np.zeros(n)
    for i in range(m):
        tmp += w[i] * data[i] * label[i]
    return tmp,b
if __name__ == "__main__":
    filename = "/Users/chutong/Desktop/Machine_Learning_Algorithm/感知机学习算法/data.txt"
    data,label = Load_data(filename,2) # K is dimension of data
    alpha = 0.3  # learining rate
    w,b = train(data,label,alpha) 
    print("Final W = {0} , b = {1}".format(w,b))
    
# data,label = Load_data("/Users/chutong/Desktop/Machine_Learning_Algorithm/感知机学习算法/data.txt")