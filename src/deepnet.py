# coding: utf-8
import sys, os
# sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
from common.functions import *

class TwoLayerNet:
    def __init__(self):
        self.layers = []
        pass

    # 损失函数
    def loss(self, t):
        pass

    # 反向传播更新梯度
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 设定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads
    
    # 训练
    def train(self, x, t):
        pass

    # 预测
    def predict(self, x):
        pass