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
    return x


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
    # 自动生成样本
    trains_features = np.random.rand(1000, 10)
    test_w = np.random.randint(1, 10, (10, 1))
    test_bia = random.randint(1, 100)
    trains_lables = np.dot(trains_features, test_w) + test_bia

    # 样本格式转化
    trains_features_list = trains_features.tolist()
    trains_lables_list_tmp = trains_lables.tolist() 
    trains_lables_list = []
    for i in range(len(trains_lables_list_tmp)):
        trains_lables_list.append(trains_lables_list_tmp[i][0])

    # 初始化参数
    features = trains_features
    lables = trains_lables_list
    iterations = 100
    learning_rate = 0.1
    p_obj = Perceptron(features, lables, iterations, learning_rate, activation)
    p_obj.train()
    print ("权重矩阵预测值：{w}".format(w=p_obj.w))
    print ("偏置量预测值：{bia}".format(bia=p_obj.bia))
    print ("权重矩阵真值：{w}".format(w=test_w.tolist()))
    print ("偏置量真值：{bia}".format(bia=test_bia))
    print p_obj.predict([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
