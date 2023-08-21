# This is a 1-level deep learning network
# It means that it have on weight level(one hidden level) and one output level)
# There are one output nodes on output level
# There are three nodes on weight level

import numpy as np
import matplotlib.pyplot as plt
import sys

# 定义一个实现梯度下降的函数，x是输入，w是权重，b是偏置，y是预测值.要求学习率为0.01，学习的次数不能超过100次
def gradient_descent(x, y, w, b, learning_rate, num_iterations, threshold=1e-8):
    for i in range(num_iterations):
        # 计算预测值
        y_pred = np.dot(x, w) + b

        # 计算梯度
        dw = np.dot(x.T, (y_pred - y))
        db = np.sum(y_pred - y)

        # 更新权重和偏置
        w -= learning_rate * dw
        b -= learning_rate * db

        # 判断是否达到阈值
        if np.all(np.abs(y_pred - y) < threshold):
            break

    return w, b


def train(x, y, weightInit, baisInit, learningRate):
    # 定义一个足够小的数
    epsilon = 1e-8
    weight = weightInit
    bais = baisInit
    w, b = gradient_descent(x, y, weight, bais, learningRate, 100, epsilon)

    return weight, bais

def predict(x, weight, bais):
    # 第一层
    z = np.dot(x, weight) + bais
    # sigmod函数
    y = 1 / (1 + np.exp(-z))
    # 第二层
    return y

if __name__ == '__main__':
    print('main start in main')

    # x是输入
    inputX = np.array([3, 6])
    print(inputX)

    # y是已经打好标的输出，用于学习
    y = np.array([8])

    # w是权重和偏置,这里先初始化
    weightInit = np.array([[1, 2, 3], [0.8, 2, 1.6]])
    baisInit = np.array([[1.5, 1.5, 1.5], [2, 2, 2]])

    # 设置学习率
    learningRate = 0.01

    # 训练，用inputX[1]. 为了使得函数没有副作用，函数里的weight
    weight, bais = train(inputX, y, weightInit, baisInit, learningRate)

    # 预测
    print(weight)
    print(bais)
    print(predict(inputX,weight,bais))
