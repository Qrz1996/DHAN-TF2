import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences

def sparseFeature(feat, feat_num, embed_dim=64):
    return{'feat': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}

def denseFeature(feat):
    return {'feat': feat}

def buid_amazon_electronic_dataset(data_path, embed_dim=64, maxlen=40):
    with open(data_path, 'rb') as f:
        train_set = pickle.load(f)
        # print('train_set shape', train_set[0])      # (114002, [23550, 23428], 9480, 0)
        test_set = pickle.load(f)
        # print('test_set shape', test_set[0])
        cate_list = pickle.load(f)
        user_count, item_count, cate_count = pickle.load(f)

        train_set = pd.DataFrame(train_set, columns=['user_id', 'hist', 'target_item', 'label'])
        test_set = pd.DataFrame(test_set, columns=['user_id', 'hist', 'target_item', 'label'])

        train_set['target_item'] = train_set['target_item'].apply(lambda x: [x, cate_list[x]])
        test_set['target_item'] = test_set['target_item'].apply(lambda x: [x, cate_list[x]])

        train_set['hist'] = train_set['hist'].apply(lambda x: [[x[i], cate_list[x[i]]] for i in range(len(x))])
        test_set['hist'] = test_set['hist'].apply(lambda x: [[x[i], cate_list[x[i]]] for i in range(len(x))])

        train_X = [np.array([0.] * len(train_set)),
                   np.array(train_set['user_id'].tolist()),
                   pad_sequences(train_set['hist'], maxlen=maxlen, padding='post'),
                   np.array(train_set['target_item'].to_list())]
        train_Y = train_set['label'].values
        test_x = [np.array([0.] * len(test_set)),
                   np.array(test_set['user_id'].tolist()),
                   pad_sequences(test_set['hist'], maxlen=maxlen, padding='post'),
                   np.array(test_set['target_item'].to_list())]
        test_Y = test_set['label'].values

        feature_columns = [[],
                           [
                               sparseFeature('item_id', item_count, embed_dim),
                               sparseFeature('cate_id', cate_count, embed_dim),
                               sparseFeature('user_id', user_count, embed_dim * 2)
                           ]]

        behavior_feature_list = ['item_id', 'cate_id']

        # for i in range(len(train_X)):
        #     print(train_X[i].shape)

        return feature_columns, behavior_feature_list, cate_count, (train_X, train_Y), (test_x, test_Y)

# buid_amazon_electronic_dataset(r'D:\datasets\Amazon_electronic\DHAN\dataset_tf2.pkl')


class DataInput:
    def __init__(self, train_X, train_Y, batch_size):
        self.batch_size = batch_size
        self.train_X = train_X
        self.train_Y = train_Y
        self.epoch_size = len(self.train_X[0]) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.train_X[0]):
            self.epoch_size += 1
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.i == self.epoch_size:
            raise StopIteration

        start_index = self.i * self.batch_size
        end_index = min((self.i + 1) * self.batch_size, len(self.train_X[0]))
        self.i += 1

        x = [np.array(self.train_X[i][start_index: end_index].tolist()) for i in range(len(self.train_X))]
        x[0] = np.expand_dims(x[0], axis=-1)
        x[1] = np.expand_dims(x[1], axis=-1)
        y = self.train_Y[start_index: end_index]
        # for i in range(len(x)):
        #     print(x[i])

        return self.i, (x, y)