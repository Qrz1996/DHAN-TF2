"""
Created on May 25, 2020

create amazon electronic dataset

@author: Ziyao Geng
"""
import pandas as pd
import numpy as np
import pickle
import random
from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences


def sparseFeature(feat, feat_num, embed_dim=4):
    """
    create dictionary for sparse feature
    :param feat: feature name
    :param feat_num: the total number of sparse features that do not repeat
    :param embed_dim: embedding dimension
    :return:
    """
    return {'feat': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}


def denseFeature(feat):
    """
    create dictionary for dense feature
    :param feat: dense feature name
    :return:
    """
    return {'feat': feat}


def create_amazon_electronic_dataset(file=r'/public/home/qrz/data/Amazon_electronic/precessed/remap.pkl', embed_dim=64, maxlen=40):
    """
    :param file: dataset path
    :param embed_dim: latent factor
    :param maxlen:
    :return: user_num, item_num, train_df, test_df
    """
    print('==========Data Preprocess Start============')
    with open(file, 'rb') as f:
        reviews_df = pickle.load(f)
        cate_list = pickle.load(f)
        user_count, item_count, cate_count, example_count = pickle.load(f)

    reviews_df = reviews_df
    reviews_df.columns = ['user_id', 'item_id', 'time']

    train_data, val_data, test_data = [], [], []

    for user_id, hist in tqdm(reviews_df.groupby('user_id')):       # groupby用法
        pos_list = hist['item_id'].tolist()

        def gen_neg():                          # 生成负样本
            neg = pos_list[0]
            while neg in pos_list:
                neg = random.randint(0, item_count - 1)
            return neg

        neg_list = [gen_neg() for i in range(len(pos_list))]
        hist = []
        for i in range(1, len(pos_list)):
            hist.append([pos_list[i - 1], cate_list[pos_list[i-1]]])
            hist_i = hist.copy()
            if i == len(pos_list) - 1:      # 最后一个最为测试数据
                # test_data.append([hist_i, [pos_list[i], cate_list[pos_list[i]]], 1])    # pos_list是item_id列表，cate_list是按照物品id升序排列的cate列表
                # test_data.append([hist_i, [neg_list[i], cate_list[neg_list[i]]], 0])    # pos_list[i]找到物品id，依据物品id找到cate_id
                test_data.append([user_id, hist_i, [pos_list[i], cate_list[pos_list[i]]], 1])  # pos_list是item_id列表，cate_list是按照物品id升序排列的cate列表
                test_data.append([user_id, hist_i, [neg_list[i], cate_list[neg_list[i]]], 0])  # pos_list[i]找到物品id，依据物品id找到cate_id
            elif i == len(pos_list) - 2:    # 倒数第二个作为验证数据
                # val_data.append([hist_i, [pos_list[i], cate_list[pos_list[i]]], 1])
                # val_data.append([hist_i, [neg_list[i], cate_list[neg_list[i]]], 0])
                val_data.append([user_id, hist_i, [pos_list[i], cate_list[pos_list[i]]], 1])
                val_data.append([user_id, hist_i, [neg_list[i], cate_list[neg_list[i]]], 0])
            else:
                # train_data.append([hist_i, [pos_list[i], cate_list[pos_list[i]]], 1])
                # train_data.append([hist_i, [neg_list[i], cate_list[neg_list[i]]], 0])
                train_data.append([user_id, hist_i, [pos_list[i], cate_list[pos_list[i]]], 1])
                train_data.append([user_id, hist_i, [neg_list[i], cate_list[neg_list[i]]], 0])

    # feature columns，
    feature_columns = [[],
                       [sparseFeature('item_id', item_count, embed_dim),
                        sparseFeature('cate_id', cate_count, embed_dim),
                        sparseFeature('user_id', user_count, embed_dim),
                        ]]  # sparseFeature('cate_id', cate_count, embed_dim)

    # behavior
    # behavior_list = ['item_id']  # , 'cate_id'
    behavior_list = ['item_id', 'cate_id']

    # shuffle
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    # create dataframe
    # train = pd.DataFrame(train_data, columns=['hist', 'target_item', 'label'])
    # val = pd.DataFrame(val_data, columns=['hist', 'target_item', 'label'])
    # test = pd.DataFrame(test_data, columns=['hist', 'target_item', 'label'])
    train = pd.DataFrame(train_data, columns=['user_id', 'hist', 'target_item', 'label'])
    val = pd.DataFrame(val_data, columns=['user_id', 'hist', 'target_item', 'label'])
    test = pd.DataFrame(test_data, columns=['user_id', 'hist', 'target_item', 'label'])

    # if no dense or sparse features, can fill with 0
    print('==================Padding===================')
    # [dense_inputs, sparse_inputs, seq_inputs, item_inputs]
    train_X = [np.array([0.] * len(train)), np.array(train['user_id'].tolist()),     # 是一个列表
               pad_sequences(train['hist'], maxlen=maxlen),
               np.array(train['target_item'].tolist())]
    train_y = train['label'].values
    val_X = [np.array([0] * len(val)),np.array(val['user_id'].tolist()),
             pad_sequences(val['hist'], maxlen=maxlen),
             np.array(val['target_item'].tolist())]
    val_y = val['label'].values
    test_X = [np.array([0] * len(test)), np.array(test['user_id'].tolist()),
              pad_sequences(test['hist'], maxlen=maxlen),
              np.array(test['target_item'].tolist())]
    test_y = test['label'].values
    print('============Data Preprocess End=============')
    return feature_columns, behavior_list, cate_count, (train_X, train_y), (test_X, test_y)

# create_amazon_electronic_dataset('raw_data/remap.pkl')
