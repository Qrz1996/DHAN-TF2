#coding=UTF-8
import numpy as np
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

import pickle

from utils import *
from model import DHAN
from experiment import *


if __name__ == '__main__':
    # data_path = r'D:\datasets\Amazon_electronic\DHAN\dataset_tf2.pkl'
    data_path = r'/public/home/qrz/data/Amazon_electronic/DHAN/dataset_tf2.pkl'

    # 超参
    learning_rate = 1e-3
    epochs = 10
    batch_size = 2049
    test_batch_size = 512

    # 数据
    feature_columns, behavior_feature_list, cate_count, train, test = buid_amazon_electronic_dataset(data_path)
    # feature_columns, behavior_feature_list, cate_count, train, test = create_amazon_electronic_dataset()
    train_X, train_Y = train
    test_X, test_Y = test

    # 模型
    model = DHAN(feature_columns, behavior_feature_list, cate_count=cate_count)
    # model.summary()
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate), metrics=[AUC()])

    # 训练
    model.fit(
        train_X,
        train_Y,
        epochs=epochs,
        # validation_split=0.2,
        validation_data=(test_X, test_Y),
        batch_size=batch_size,
        # callbacks=[EarlyStopping(monitor='val_auc', patience=2,restore_best_weights=True)]
    )

    # print('test AUC: %f' % model.evaluate(test_X, test_Y, batch_size=batch_size)[1])

    # 自定义训练过程
    # optimizer = Adam(learning_rate=1.)
    # # loss = tf.keras.losses.BinaryCrossentropy()
    # metrics = AUC()
    #
    # # @tf.function
    # def train_step(x, y):
    #     with tf.GradientTape() as tape:
    #         y = tf.expand_dims(y, axis=-1)
    #         pred = model(x)
    #         loss_value = tf.keras.losses.binary_crossentropy(y, pred)
    #         loss_value = tf.reduce_mean(loss_value)
    #     grads = tape.gradient(loss_value, model.trainable_variables)
    #     # print(loss_value)
    #     optimizer.apply_gradients(zip(grads, model.trainable_variables))
    #     return loss_value
    #
    # for epoch in range(epochs):
    #     for step, (x, y) in iter(DataInput(train_X, train_Y, batch_size=batch_size)):
    #         # variable_names = model.trainable_variables[0]
    #         # print(variable_names, '更新前值：', variable_names)
    #         loss_value = train_step(x, y)
    #         if step % 100 == 0:
    #             print('step:{} loss:{}'.format(step, loss_value))
    #
    #         if step % 1000 == 0:
    #             # test_auc = model.evaluate(test_X, test_Y, batch_size=test_batch_size)[1]
    #             # print('epoch:{} step:{} train loss:{:.4f}'.format(epoch, step, test_auc))
    #             metrics.reset_states()
    #             for _, (x, y) in iter(DataInput(test_X, test_Y, batch_size=test_batch_size)):
    #                 pred = model(x)
    #                 metrics.update_state(y, pred)
    #             print('epoch:{} step:{} train auc:{:.4f}'.format(epoch, step, metrics.result()))
