# -*- coding: utf-8 -*-
"""
@Time:2022-02-24 12:19
@Author:yang qifan
@File:lfm.py
@IDE:PyCharm
"""

import numpy as np
import os
from LFM.util import read


def ifm_train(train_data, F, alpha, beta, step):
    """
    :param train_data: 训练数据
    :param F: 隐向量维度
    :param alpha: 正则化系数
    :param beta: 学习速率
    :param step: 迭代次数
    :return:
        dict:key itemid,value:list
        dict:key userdid,value:list
    """
    user_vec = {}
    item_vec = {}
    for step_index in range(step):
        # 每一次迭代中，从训练样本中获取实例
        for data_instance in train_data:
            userid, itemid, label = data_instance
            # 若user或item第一次出现，则进行初始化
            if userid not in user_vec:
                user_vec[userid] = init_model(F)
            if itemid not in item_vec:
                item_vec[itemid] = init_model(F)
        # label为公式中的p(u,i)
        delta = label - model_predict(user_vec[userid], item_vec[itemid])
        for index in range(F):
            user_vec[userid][index] += beta * (delta * item_vec[itemid][index] - alpha * user_vec[userid][index])
            item_vec[itemid][index] += beta * (delta * user_vec[userid][index] - alpha * item_vec[itemid][index])
        beta = beta * 0.9  # 对学习率进行一个衰减，目的是让模型在接近收敛时变化的慢一点
    return user_vec, item_vec


def init_model(vector_len):
    """
    :param vector_len: 向量维数
    :return:
        a ndarray
    """
    return np.random.randn(vector_len)


def model_predict(user_vector, item_vector):
    """
    模型预测user和item的距离
    :param user_vector:模型产出的表征user的向量
    :param item_vector:模型产出的表征item的向量
    :return:
        a num
    """
    res = np.dot(user_vector, item_vector) / (np.linalg.norm(user_vector) * np.linalg.norm(item_vector))
    return res


def model_train_process():
    """
    test lfm model train
    :return:
    """
    train_data = read.get_train_data("../data/ratings.csv")
    user_vec, item_vec = ifm_train(train_data, 50, 0.01, 0.1, 50)
    print(user_vec["1"])
    print(item_vec["2455"])


if __name__ == '__main__':
    model_train_process()
