#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2020 wangshuaibupt. All Rights Reserved
# 
################################################################################
"""
Authors: wangshuaibupt(wangshuaibupt@126.com)
Date:    2020/07/25 14:57:06
"""
import random

import perceptron
import numpy as np


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def activation(x):
    """激活函数"""
    # return sigmoid(x)
    return x


def activation_list(x_list):
    """列表每个元素都使用激活函数"""
    out = []
    for i in range(0, len(x_list)):
        out.append(activation(x_list[i]))
    return out


class LineCF(perceptron.Perceptron):
    """感知机函数"""

    def __init__(self, features, lables, iterations, learning_rate, activation):
        """初始化
        :param  features 特征个数
        :param  lables 标签
        :param  activation 激活函数
        """
        perceptron.Perceptron.__init__(self, features, lables, iterations, learning_rate, activation)


if __name__ == '__main__':
    # 自动生成样本
    trains_features = np.random.rand(1000, 10) - 0.5
    test_w = np.random.rand(10, 1) - 0.5 
    test_bia = 0.0 
    trains_lables = np.dot(trains_features, test_w) + test_bia

    # 样本格式转化
    trains_features_list = trains_features.tolist()
    trains_lables_list_tmp = trains_lables.tolist() 
    trains_lables_list = []
    for i in range(len(trains_lables_list_tmp)):
        trains_lables_list.append(trains_lables_list_tmp[i][0])
    trains_lables_list = activation_list(trains_lables_list)
    # 初始化参数
    features = trains_features
    lables = trains_lables_list

    iterations = 100
    learning_rate = 0.1
    p_obj = LineCF(features, lables, iterations, learning_rate, activation)
    p_obj.train()
    print ("权重矩阵预测值：{w}".format(w=p_obj.w))
    print ("偏置量预测值：{bia}".format(bia=p_obj.bia))
    print ("权重矩阵真值：{w}".format(w=test_w.tolist()))
    print ("偏置量真值：{bia}".format(bia=test_bia))
    print (p_obj.predict([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]))
