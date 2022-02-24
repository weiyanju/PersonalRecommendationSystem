# -*- coding: utf-8 -*-
"""
@Time:2022-01-27 15:03
@Author:yang qifan
@File:read.py
@IDE:PyCharm
"""
import os


def get_item_info(input_file):
    """
    get item info:[title,genre]
    :param input_file:item info file
    :return:
        a dict:
            key:itemid,value:[title,genre]
    """
    if not os.path.exists(input_file):
        return {}
    item_info = {}
    linenum = 0
    fp = open(input_file, encoding='utf-8')
    for line in fp:
        if linenum == 0:
            linenum += 1
            continue
        item = line.strip().split(',')
        if len(item) < 3:
            continue
        elif len(item) == 3:
            itemid, title, genre = item[0], item[1], item[2]
        elif len(item) > 3:
            itemid = item[0]
            genre = item[-1]
            title = ",".join(item[1:-1])
        item_info[itemid] = [title, genre]
    fp.close()
    return item_info


def get_ave_score(input_file):
    """
    get item ave rating score
    :param input_file: user rating file
    :return:
        a dict:
            key:itemid,value:ave_score
    """
    if not os.path.exists(input_file):
        return {}
    fp = open(input_file)
    linenume = 0
    record_dict = {}  # 电影的评分人数，总评分
    score_dict = {}  # 电影的平均分
    for line in fp:
        if linenume == 0:
            linenume += 1
            continue
        item = line.strip().split(',')
        if len(item) < 4:
            continue
        userid, itemid, rating = item[0], item[1], item[2]
        if itemid not in record_dict:
            record_dict[itemid] = [0, 0]
        record_dict[itemid][0] += 1  # 评分人数+1
        record_dict[itemid][1] += eval(rating)  # 评分相加
    fp.close()
    for itemid in record_dict:
        score_dict[itemid] = round(record_dict[itemid][1] / record_dict[itemid][0], 3)  # 平均分 = 总评分/评分人数
    return score_dict


def get_train_data(input_file):
    """
    get train data for LFM model train
    :param input_file: user item rating file
    :return:
        a list:
            [(userid,itemid,label),(userid1,itemid1,label)]
    """
    if not os.path.exists(input_file):
        return []
    score_dict = get_ave_score(input_file)
    pos_dict = {}  # {"userid":(itemid,1),}
    neg_dict = {}  # {"userid":(itemid,ave_score),}
    train_data = []  # [("userid","itemid","flag")]
    linenum = 0
    score_thr = 4.0
    fp = open(input_file)
    for line in fp:
        if linenum == 0:
            linenum += 1
            continue
        item = line.strip().split(',')
        if len(item) < 4:
            continue
        userid, itemid, rating = item[0], item[1], float(item[2])
        if userid not in pos_dict:
            pos_dict[userid] = []
        if userid not in neg_dict:
            neg_dict[userid] = []

        if rating >= score_thr:  # 大于阈值则看作正样本
            pos_dict[userid].append((itemid, 1))
        else:
            score = score_dict.get(itemid, 0)  # 获取电影的平均分
            neg_dict[userid].append((itemid, score))
    fp.close()
    for userid in pos_dict:
        data_num = min(len(pos_dict[userid]), len(neg_dict.get(userid, [])))  # 取用户正样本和负样本个数的小值
        if data_num > 0:
            train_data += [(userid, sample[0], sample[1]) for sample in pos_dict[userid]][:data_num]  # 正样本直接加入
        else:
            continue
        # 对负样本按照平均评分进行排序，element是[itemid,score]
        # 这里表示如果用户对热门（评分高）的电影评分不高（小于4），则很可能这个用户对这部电影没有兴趣
        sorted_neg_list = sorted(neg_dict[userid], key=lambda element: element[1], reverse=True)[:data_num]
        train_data += [(userid, sample[0], 0) for sample in sorted_neg_list]
    return train_data


if __name__ == "__main__":
    # item_dict = get_item_info("../data/movies.csv")
    # print(len(item_dict))
    # print(item_dict["1"])
    # print(item_dict["11"])
    # score_dict = get_ave_score("../data/ratings.csv")
    # print(score_dict)
    # print(len(score_dict))
    # print(score_dict["31"])
    train_data = get_train_data("../data/ratings.csv")
    print(len(train_data))
    print(train_data[:64])
