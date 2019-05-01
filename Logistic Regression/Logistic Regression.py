# -*- coding: utf-8 -*-
# @Time    : 2019/5/1 16:54
# @Author  : DaHuang
# @FileName: Logistic Regression.py
# @Software: PyCharm


import pandas as pd
import numpy as np

def loaddata(filename):
    """
    加载数据集
    data: 原始数据集
    return: 特征数据x与标签类别数据y
    """
    dataSet = pd.read_table(filename, header=None)
    dataSet.columns = ['X1', 'X2', 'label']
    dataSet.insert(0, 'X0', 1)
    columns = [i for i in dataSet.columns if i != 'label']
    data_x = dataSet[columns]
    data_y = dataSet[['label']]
    return data_x,data_y


#sigmoid函数
def sigmoid(y):
    s = 1.0/(1.0+np.exp(-y))
    return s


def cost(xMat,weights,yMat):
    """
    计算损失函数
    xMat: 特征数据-矩阵
    weights: 参数
    yMat: 标签数据-矩阵
    return: 损失函数
    """
    m, n = xMat.shape
    hypothesis = sigmoid(np.dot(xMat, weights))  # 预测值
    cost = (-1.0 / m) * np.sum(yMat.T * np.log(hypothesis) + (1 - yMat).T * np.log(1 - hypothesis))  # 损失函数
    return cost



def BGD_LR(data_x,data_y,alpha=0.1,maxepochs=10000,epsilon=1e-4):
    xMat = np.mat(data_x)
    yMat = np.mat(data_y)
    m,n = xMat.shape
    weights = np.ones((n,1)) #初始化模型参数
    epochs_count = 0
    while epochs_count < maxepochs:
        loss = cost(xMat,weights,yMat) #上一次损失值
        hypothesis = sigmoid(np.dot(xMat,weights)) #预测值
        error = hypothesis -yMat #预测值与实际值误差
        print(loss)
        grad = (1.0/m)*np.dot(xMat.T,error) #损失函数的梯度
        last_weights = weights #上一轮迭代的参数
        weights = weights - alpha*grad #参数更新
        loss_new = cost(xMat,weights,yMat)#当前损失值
        if abs(loss_new-loss)<epsilon:#终止条件
            break
        epochs_count += 1
    print('迭代到第{}次，结束迭代！'.format(epochs_count))
    return weights


def acc(weights, test_x, test_y):
    xMat_test = np.mat(test_x)
    m, n = xMat_test.shape
    result = []
    for i in range(m):
        proba = sigmoid(np.dot(xMat_test[i,:], weights))
        if proba < 0.5:
            preict =  0
        else:
            preict = 1
        result.append(preict)
    test_x_ = test_x.copy()
    test_x_['predict'] = result
    acc = (test_x_['predict']==test_y['label']).mean()
    return acc


if __name__ == '__main__':
    data_x,data_y = loaddata('LR_data.txt')
    weights_bgd = BGD_LR(data_x, data_y, alpha=0.1,maxepochs=10000)
    print(weights_bgd)
    test_x = data_x.copy()
    test_y = data_y.copy()
    accuracy = acc(weights_bgd,test_x,test_y)
    print(accuracy)



#     print(data_y)