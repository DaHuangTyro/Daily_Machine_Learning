# -*- coding: utf-8 -*-
# @Time    : 2019/5/1 16:38
# @Author  : DaHuang
# @FileName: Gradient_descent.py
# @Software: PyCharm


import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt



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
    """
    使用批量梯度下降法BGD求解逻辑回归
    data_x: 特征数据
    data_y: 标签数据
    aplha: 步长，该值越大梯度下降幅度越大
    maxepochs: 最大迭代次数
    epsilon: 损失精度
    return: 模型参数
    """
    starttime = time.time()
    xMat = np.mat(data_x)
    yMat = np.mat(data_y)
    m,n = xMat.shape
    weights = np.ones((n,1)) #初始化模型参数
    epochs_count = 0
    loss_list = []
    epochs_list = []
    while epochs_count < maxepochs:
        loss = cost(xMat,weights,yMat) #上一次损失值
        hypothesis = sigmoid(np.dot(xMat,weights)) #预测值
        error = hypothesis -yMat #预测值与实际值误差
        grad = (1.0/m)*np.dot(xMat.T,error) #损失函数的梯度
        last_weights = weights #上一轮迭代的参数
        weights = weights - alpha*grad #参数更新
        loss_new = cost(xMat,weights,yMat)#当前损失值
        if abs(loss_new-loss)<epsilon:#终止条件
            break
        loss_list.append(loss_new)
        epochs_list.append(epochs_count)
        epochs_count += 1
    print(loss_new)
    print("批量梯度下降算法耗时：", time.time() - starttime)
    print('迭代到第{}次，结束迭代！'.format(epochs_count))
    plt.plot(epochs_list,loss_list)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()
    return weights


def SGD_LR(data_x, data_y, alpha=0.1, maxepochs=10000,epsilon=1e-4):
    """
    使用SGD求解逻辑回归
    data_x: 特征数据
    data_y: 标签数据
    aplha: 步长，该值越大梯度下降幅度越大
    maxepochs: 最大迭代次数
    epsilon: 损失精度
    return: 模型参数
    """
    starttime = time.time()
    xMat = np.mat(data_x)
    yMat = np.mat(data_y)
    m, n = xMat.shape
    weights = np.ones((n, 1))  # 模型参数
    epochs_count = 0
    loss_list = []
    epochs_list = []
    while epochs_count < maxepochs:
        rand_i = np.random.randint(m)  # 随机取一个样本
        loss = cost(xMat,weights,yMat) #前一次迭代的损失值
        hypothesis = sigmoid(np.dot(xMat[rand_i,:],weights)) #预测值
        error = hypothesis -yMat[rand_i,:] #预测值与实际值误差
        grad = np.dot(xMat[rand_i,:].T,error) #损失函数的梯度
        weights = weights - alpha*grad #参数更新
        loss_new = cost(xMat,weights,yMat)#当前迭代的损失值
        if abs(loss_new-loss)<epsilon:
            break
        loss_list.append(loss_new)
        epochs_list.append(epochs_count)
        epochs_count += 1
    print(loss_new)
    print("随机梯度下降算法耗时：", time.time() - starttime)
    print('迭代到第{}次，结束迭代！'.format(epochs_count))
    plt.plot(epochs_list,loss_list)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()
    return weights


def MBGD_LR(data_x, data_y, alpha=0.1,batch_size=10, maxepochs=10000,epsilon=1e-4):
    """
    使用MBGD求解逻辑回归
    data_x: 特征数据
    data_y: 标签数据
    aplha: 步长，该值越大梯度下降幅度越大
    maxepochs: 最大迭代次数
    epsilon: 损失精度
    return: 模型参数
    """
    starttime = time.time()
    xMat = np.mat(data_x)
    yMat = np.mat(data_y)
    m, n = xMat.shape
    weights = np.ones((n, 1))  # 模型参数
    epochs_count = 0
    loss_list = []
    epochs_list = []
    while epochs_count < maxepochs:
        randIndex = np.random.choice(range(len(xMat)), batch_size, replace=False)
        loss = cost(xMat,weights,yMat) #前一次迭代的损失值
        hypothesis = sigmoid(np.dot(xMat[randIndex],weights)) #预测值
        error = hypothesis -yMat[randIndex] #预测值与实际值误差
        grad = (1.0/batch_size)*np.dot(xMat[randIndex].T,error) #损失函数的梯度
        weights = weights - alpha*grad #参数更新
        loss_new = cost(xMat,weights,yMat)#当前迭代的损失值
        if abs(loss_new-loss)<epsilon:
            break
        loss_list.append(loss_new)
        epochs_list.append(epochs_count)
        epochs_count += 1
    print(loss_new)
    print("小批量梯度下降算法耗时：", time.time() - starttime)
    print('迭代到第{}次，结束迭代！'.format(epochs_count))
    plt.plot(epochs_list,loss_list)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()
    return weights


if __name__ == '__main__':
    data_x,data_y = loaddata('testSet.txt')
    weights_bgd = BGD_LR(data_x, data_y, alpha=0.1, maxepochs=10000, epsilon=1e-4)
    weights_sgd = SGD_LR(data_x, data_y, alpha=0.1, maxepochs=10000, epsilon=1e-4)
    weights_mbgd = MBGD_LR(data_x, data_y, alpha=0.1, batch_size=3, maxepochs=10000,epsilon=1e-4)


