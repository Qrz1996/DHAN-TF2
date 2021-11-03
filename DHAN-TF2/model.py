import tensorflow as tf
from tensorflow.keras.layers import Embedding, BatchNormalization, Dense, Dropout, LayerNormalization, PReLU, Input
from tensorflow.keras.regularizers import l2

from modules import *

class DHAN(tf.keras.Model):
    def __init__(self, feature_columns, behavior_feature_list, att_hidden_units=(80, 40), ffn_hidden_units=(80, 40),
                 att_activation='prelu',ffn_activation='prelu', maxlen=40, dnn_dropout=0., embed_reg=1e-4, cate_count=56):
        super(DHAN, self).__init__()

        self.cate_count = cate_count
        self.maxlen = maxlen
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.other_sparse_nums = len(self.sparse_feature_columns) - len(behavior_feature_list)
        self.dense_len = len(self.dense_feature_columns)
        self.behavior_feat_nums = len(behavior_feature_list)

        # 给历史sparse特征和其他sparse特征建立Embedding
        self.other_sparse_embedding = [Embedding(input_length=1, input_dim=feat['feat_num'],
                                                 output_dim=feat['embed_dim'],
                                                 embeddings_initializer='random_uniform',
                                                 embeddings_regularizer=l2(embed_reg))
                                    for feat in self.sparse_feature_columns if feat['feat'] not in behavior_feature_list]
        self.behavior_sparse_embedding = [Embedding(input_length=1, input_dim=feat['feat_num'],
                                                    output_dim=feat['embed_dim'],
                                                    embeddings_initializer='random_uniform',
                                                    embeddings_regularizer=l2(embed_reg))
                                   for feat in self.sparse_feature_columns if feat['feat'] in behavior_feature_list]

        self.cate_attention_layer = CateAttentionLayer(att_hidden_units, att_activation, cate_count)

        self.attention_layer = AttentionLayer(att_hidden_units, att_activation)

        self.ln = LayerNormalization(trainable=True)

        self.bn1 = BatchNormalization(trainable=True)
        self.bn2 = BatchNormalization(trainable=True)

        self.ffn = [Dense(unit, activation=PReLU() if ffn_activation=='prelu' else Dice()) for unit in ffn_hidden_units]

        self.dropout = Dropout(dnn_dropout)

        self.final_dense = Dense(1)

    def call(self, inputs):
        dense_inputs, sparse_inputs, seq_inputs, item_inputs = inputs
        # print(sparse_inputs.shape)
        mask = tf.cast(tf.not_equal(seq_inputs[:, :, 0], 0), dtype=tf.float32)

        # 非item的其他sparse特征embedding
        if self.dense_len > 0:
            other_info = dense_inputs
        else:
            other_info = self.other_sparse_embedding[0](sparse_inputs[:, 0])    # user_id总是存在的

        for i in range(1, self.other_sparse_nums):
            other_info = tf.concat(other_info, self.other_sparse_embedding[i](sparse_inputs[:, i]), axis=-1)

        # 历史行为embedding和target item embedding
        seq_embed = tf.concat([self.behavior_sparse_embedding[i](seq_inputs[:, :, i])
                               for i in range(self.behavior_feat_nums)], axis=-1)
        item_embed = tf.concat([self.behavior_sparse_embedding[i](item_inputs[:, i])
                                for i in range(self.behavior_feat_nums)], axis=-1)

        # 得到历史行为中的cate_ids, [b, T]
        his_cates = seq_inputs[:, :, 1]
        # 第一层注意力层抽取用户兴趣表示和依据cate分组的隐向量
        # user_interest:[b, d*2]  cate_hidden:[b, L, d*2]
        user_interest, cate_hidden = self.cate_attention_layer([item_embed, seq_embed, mask, his_cates])

        # 第二层对不同的cate做注意力
        # ---------------------------------------
        cate_attention = self.attention_layer([item_embed, cate_hidden, cate_hidden])

        # info_all = user_interest
        info_all = tf.concat([user_interest, cate_attention], axis=-1)

        # info_all = self.ln(info_all)
        # info_all = self.bn1(info_all)
        # ---------------------------------------

        # 拼接输入
        if self.dense_len > 0 or self.other_sparse_nums > 0:
            info_all = tf.concat([info_all, item_embed, other_info], axis=-1)  # [b, embed_len]
        else:
            info_all = tf.concat([info_all, item_embed], axis=-1)


        info_all = self.bn2(info_all)

        # mlp
        for dense in self.ffn:
            info_all = dense(info_all)

        info_all = self.dropout(info_all)
        output = tf.nn.sigmoid(self.final_dense(info_all))

        return output

    def summary(self):
        dense_inputs = Input(shape=(self.dense_len, ), dtype=tf.float32)
        sparse_inputs = Input(shape=(self.other_sparse_nums, ), dtype=tf.int32)
        seq_inputs = Input(shape=(self.maxlen, self.behavior_feat_nums), dtype=tf.int32)
        item_inputs = Input(shape=(self.behavior_feat_nums, ), dtype=tf.int32)
        tf.keras.Model(inputs=[dense_inputs, sparse_inputs, seq_inputs, item_inputs],
                       outputs=self.call([dense_inputs, sparse_inputs, seq_inputs, item_inputs])).summary()



def cheshi_model():
    dense_features = [{'feat': 'a'}, {'feat': 'b'}]
    sparse_features = [{'feat': 'item_id', 'feat_num': 100, 'embed_dim': 64},
                       {'feat': 'cate_id', 'feat_num': 100, 'embed_dim': 64},
                       {'feat': 'adv_id', 'feat_num': 100, 'embed_dim': 64}]
    behavior_list = ['item_id', 'cate_id']
    features = [dense_features, sparse_features]
    model = DHAN(features, behavior_list)
    model.summary()


# cheshi_model()







