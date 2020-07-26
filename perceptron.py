#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2020 wangshuaibupt. All Rights Reserved
# 
################################################################################
"""
perceptron.py
感知机
Authors: wangshuaibupt(wangshuaibupt@126.com)
Date:    2020/07/25 14:57:06
"""
import random
import logging

import functools
import numpy as np


def activation(x):
    """激活函数"""
    # return x
    return 1 if x > 0 else 0


def activation_list(x_list):
    """列表每个元素都使用激活函数"""
    out = []
    for i in range(0, len(x_list)):
        out.append(activation(x_list[i]))
    return out


class VectorOP(object):
    """向量操作"""

    @staticmethod
    def dot_product(x, y, bia=None):
        """带偏置项的点积"""
        if bia is None:
            bia = 0.0
        if len(x) == 0 or len(y) == 0:
            return 0.0
        return functools.reduce(lambda a, b: a + b, VectorOP.vector_multiply(x, y), bia)

    @staticmethod
    def vector_add(x, y):
        """向量对应元素相加"""
        return list(map(lambda x_y: x_y[0] + x_y[1], zip(x, y)))

    @staticmethod
    def vector_subtraction(x, y):
        """向量对应元素项相减"""
        return list(map(lambda x_y: x_y[0] - x_y[1], zip(x, y)))

    @staticmethod
    def vector_add(x, y):
        """向量对应元素相加"""
        return list(map(lambda x_y: x_y[0] + x_y[1], zip(x, y)))

    @staticmethod
    def scala_multiply(v, s):
        """将向量v中的每个元素和标量s相乘"""
        return map(lambda e: e * s, v)

    @staticmethod
    def vector_multiply(x, y):
        """向量对应元素相乘"""
        return list(map(lambda multiply: multiply[0] * multiply[1], zip(x, y)))


class Perceptron(object):
    """感知机函数"""

    def __init__(self, features, lables, iterations, learning_rate, activation):
        """初始化
        :param  features 特征个数
        :param  lables 标签
        :param  activation 激活函数
        """
        self.features = features
        self.lables = lables
        self.activation = activation
        self.input_parm_num = 0
        # 最长迭代次数
        self.iterations = iterations
        # 学习速率
        self.learning_rate = learning_rate

        if len(self.features) > 0:
            self.input_parm_num = len(self.features[0])
            # 权重向量
            self.w = [0.0] * self.input_parm_num
            # 偏执
            self.bia = 0.0

    def one_iteration(self):
        """单次迭代将所有数据过一遍"""
        samples = zip(self.features, self.lables)
        for feature, lable in samples:
            sub = self.predict(feature) - lable
            delta_w = VectorOP.scala_multiply(feature, self.learning_rate * sub)
            self.w = list(map(lambda a, b: a - b, self.w, delta_w))
            self.bia = self.bia - self.learning_rate * sub * 1.0

    def train(self):
        """函数训练"""
        for i in range(0, self.iterations):
            logging.info("iterations num is %s", i)
            self.one_iteration()

    def predict(self, x):
        """预测"""
        return self.activation(VectorOP.dot_product(self.w, x, self.bia))


if __name__ == '__main__':

    features = [[1, 1, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 0]]
    lables = [1, 0, 0, 0, 0, 0, 0, 0]
    iterations = 1000
    learning_rate = 0.1
    p_obj = Perceptron(features, lables, iterations, learning_rate, activation)
    p_obj.train()
    print ("权重矩阵预测值：{w}".format(w=p_obj.w))
    print ("偏置量预测值：{bia}".format(bia=p_obj.bia))
    print p_obj.predict([1.0, 1.0, 1.0])
    print p_obj.predict([1.0, 0.0, 1.0])
    print p_obj.predict([0.0, 1.0, 1.0])
