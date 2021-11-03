import tensorflow as tf
from tensorflow.keras.layers import Dense, PReLU

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, att_hidden_units, att_activation):
        super(AttentionLayer, self).__init__()
        self.att_dense = [Dense(unit, activation=PReLU() if att_activation=='prelu' else Dice())
                          for unit in att_hidden_units]
        self.att_final_dense = Dense(1)

    def call(self, inputs):
        # q:[b, d*2]  k:[b, L, d*2] v :[b, L, d*2]
        q, k, v = inputs

        q = tf.tile(q, multiples=[1, k.shape[1]])
        q = tf.reshape(q, [-1, k.shape[1], k.shape[2]])

        info = tf.concat([q, k, q - k, q * k], axis=-1)

        for dense in self.att_dense:
            info = dense(info)

        outputs = self.att_final_dense(info)    # [b, L, 1]
        outputs = tf.squeeze(outputs, axis=-1)           # [b, L]

        # 第二层没有padding项也就不需要mask

        # softmask
        outputs = tf.nn.softmax(logits=outputs)
        outputs = tf.expand_dims(outputs, axis=1)   # [b, 1, L]

        outputs = tf.matmul(outputs, v)             # [b, 1, d*2]
        outputs = tf.squeeze(outputs, axis=1)               # [b, d*2]


        return outputs

class CateAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, att_hidden_units, att_activation, cate_count):
        super(CateAttentionLayer, self).__init__()

        self.cate_count = cate_count
        self.att_dense = [Dense(unit, activation=PReLU() if att_activation == 'prelu' else Dice())
                          for unit in att_hidden_units]
        self.att_final_dense = Dense(1)

    def call(self, inputs):
        # q:[b, d*2]  k:[b, T, d*2] v :[b, T, d*2]
        q, k, mask, hist_cate = inputs

        # print('q.shape', q.shape)
        # print('mask.shape', mask.shape)
        # print('hist_cate.shape', hist_cate.shape)
        # print('cate_count', self.cate_count)

        q = tf.tile(q, [1, k.shape[1]])
        q = tf.reshape(q, [-1, k.shape[1], k.shape[2]])

        info = tf.concat([q, k, q - k, q * k], axis=-1)

        for dense in self.att_dense:
            info = dense(info)

        outputs = self.att_final_dense(info)  # [b, T, 1]
        outputs = tf.squeeze(outputs, axis=-1)  # [b, T]

        # mask
        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(mask, 0), paddings, outputs)
        outputs = outputs / (k.shape[2] ** 0.5)

        # softmax
        outputs = tf.nn.softmax(logits=outputs)
        outputs = tf.expand_dims(outputs, axis=1)  # [b, 1, T]

        user_interest = tf.matmul(outputs, k)  # [b, 1, d*2]

        # 按照cate值分组
        eps = 1e-10
        cate_hidden = tf.zeros_like(user_interest, tf.float32)    # [b, 1, d*2]
        for i in range(self.cate_count):
            idx_org = tf.equal(hist_cate, i)    # [b, T]    [False, False, Ture, False, True, ..., False]
            idx = tf.expand_dims(idx_org, 1)    # [b, 1, T]

            # 组别权重值 [0, 0, w_2, 0, w_4, ..., 0]
            weights_org = tf.where(idx, outputs, tf.zeros_like(outputs))
            # print('weights_org.shape', weights_org.shape)
            weights_org = tf.squeeze(weights_org, axis=1)       # [b, T]
            # print('weights_org.shape', weights_org.shape)

            # 每个用户的组别权重和
            weights_sum = tf.reduce_sum(weights_org, 1)
            weights_sum = tf.expand_dims(weights_sum, 1)
            weights_sum += eps      # [b, 1]

            weights = tf.divide(weights_org, weights_sum)
            weights = tf.expand_dims(weights, 1)    # [b, 1, T]

            final_out = tf.matmul(weights, k)       # [b, 1, d*2]
            # print('final_out.sahpe', final_out.shape)

            if i == 0:
                cate_hidden = final_out
            else:
                cate_hidden = tf.concat([cate_hidden, final_out], axis=1)


        user_interest = tf.squeeze(user_interest, axis=1)   # [b, d*2]

        # print('user_interest.shape', user_interest.shape)
        # print('cate_hidden.shape', cate_hidden.shape)

        return user_interest, cate_hidden


class Dice(tf.keras.layers.Layer):
    def __init__(self):
        super(Dice, self).__init__()

    def call(self, inputs):
        pass

