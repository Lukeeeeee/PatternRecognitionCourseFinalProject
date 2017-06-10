from __future__ import print_function
import tensorflow as tf
from modelConfig import ModelConfig as conf
from src.data.dataConfig import DataConfig
from src.data.Data import Data
import datetime
import os
import numpy as np
import json


class Model(object):

    def __init__(self, data):

        self.sess = tf.InteractiveSession()
        self.data = data

        ti = datetime.datetime.now()
        self.log_dir = ('../../log/' + str(ti.month) + '-' + str(ti.day) + '-' + str(ti.hour) + '-'
                   + str(ti.minute) + '-' + str(ti.second) + '/')

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.loss_log_file = open(self.log_dir + 'loss.txt', "w")

        self.config_log_file = open(self.log_dir + 'config.txt', "a")

        if not os.path.exists(self.log_dir + '/predication/'):
            os.makedirs(self.log_dir + '/predication/')


        self.weight = {
            'conv1_filter': tf.Variable(
                tf.random_normal([conf.FILTER_SIZE, conf.FILTER_SIZE, conf.INPUT_CHANNEL, conf.LAYER1_CHANNEL])),
            'conv2_filter': tf.Variable(
                tf.random_normal([conf.FILTER_SIZE, conf.FILTER_SIZE, conf.LAYER1_CHANNEL, conf.LAYER2_CHANNEL])),
            'conv3_filter': tf.Variable(
                tf.random_normal([conf.FILTER_SIZE, conf.FILTER_SIZE, conf.LAYER2_CHANNEL, conf.LAYER3_CHANNEL])),
            'fc1_weight': tf.Variable(
                tf.random_normal([conf.FULLY_CONNECTED_INPUT_SIZE, conf.FULLY_CONNECTED_OUT_SIZE])),
            'out_weight': tf.Variable(tf.random_normal([conf.OUT_LAYER_INPUT_SIZE, conf.OUT_LAYER_OUT_SIZE]))
        }

        self.bias = {
            'conv1': tf.Variable(tf.random_normal([conf.LAYER1_CHANNEL])),
            'conv2': tf.Variable(tf.random_normal([conf.LAYER2_CHANNEL])),
            'conv3': tf.Variable(tf.random_normal([conf.LAYER3_CHANNEL])),
            'fc1': tf.Variable(tf.random_normal([conf.FULLY_CONNECTED_OUT_SIZE])),
            'out': tf.Variable(tf.random_normal([conf.OUT_LAYER_OUT_SIZE]))
        }

        self.fc1 = None

        self.input = tf.placeholder(tf.float32, [None, conf.INPUT_X_SIZE, conf.INPUT_Y_SIZE, conf.INPUT_CHANNEL])
        self.label = tf.placeholder(tf.float32, [None, conf.OUTPUT_SIZE])
        self.is_training = tf.placeholder(tf.bool)
        self.predication = self.create_net_work()
        self.loss, self.optimizer = self.create_train_method()

        self.sess.run(tf.global_variables_initializer())

    def create_net_work(self):
        conv1 = self.conv2d(x=self.input, filer=self.weight['conv1_filter'], b=self.bias['conv1'], strides=conf.CONVOLUTIONAL_STRIDES)
        conv1 = self.maxpool2d(x=conv1, k=conf.MAX_POOLING_STRIDES_1)

        conv2 = self.conv2d(x=conv1, filer=self.weight['conv2_filter'], b=self.bias['conv2'], strides=conf.CONVOLUTIONAL_STRIDES)
        conv2 = self.maxpool2d(x=conv2, k=conf.MAX_POOLING_STRIDES_2)

        conv3 = self.conv2d(x=conv2, filer=self.weight['conv3_filter'], b=self.bias['conv3'], strides=conf.CONVOLUTIONAL_STRIDES)
        conv3 = self.maxpool2d(x=conv3, k=conf.MAX_POOLING_STRIDES_3)

        fc1 = tf.reshape(conv3, [-1, conf.FULLY_CONNECTED_INPUT_SIZE])
        fc1 = tf.add(tf.matmul(fc1, self.weight['fc1_weight']), self.bias['fc1'])
        fc1 = tf.nn.relu(fc1)
        self.fc1 = fc1
        self.conv3 = conv3
        self.conv1 = conv1
        self.conv2 = conv2

        if self.is_training is True:
            fc1 = tf.nn.dropout(fc1, conf.DROPOUT_PROBABILITY)

        out = tf.add(tf.matmul(fc1, self.weight['out_weight']), self.bias['out'])
        out = tf.nn.softmax(out)
        return out

    def create_train_method(self):
        # print(self.label)
        # print(self.predication)
        loss = tf.reduce_mean(tf.square(self.label - self.predication))

        print(self.label)
        print(self.predication)

        optimizer = tf.train.AdagradOptimizer(learning_rate=conf.LEARNING_RATE).minimize(loss)

        return loss, optimizer

    def train(self):
        i = 0
        for epoch in range(conf.EPOCH):
            avg_loss = 0.0
            for x, label in self.data.return_one_batch_data():
                # input_x, input_label = tf.train.shuffle_batch(tensors=[x, label], batch_size=20, num_threads=4,
                # capacity=50000, min_after_dequeue=20, allow_smaller_final_batch=True)
                # self.input = input_x
                # self.label = input_label

                _, cost = self.sess.run([self.optimizer, self.loss], feed_dict={self.input: x, self.label: label})

                avg_loss = (avg_loss * i + cost) / (i + 1.0)

                # _, cost = self.sess.run([self.optimizer, self.loss])

                i += 1
                print("Epoch = %3d, Iter= %3d, Current Batch Loss= %.10lf, Average Epoch Loss= %.10lf" %
                      (epoch, i, cost, avg_loss))
                print("Epoch = %3d, Iter= %3d, Current Batch Loss= %.10lf, Average Epoch Loss= %.10lf" %
                      (epoch, i, cost, avg_loss), file=self.loss_log_file)
            model.test(test_image_id=21)
            model.test(test_image_id=350)

    def test(self, test_image_id):
        x = self.data.load_test_data(test_image_id=test_image_id)

        res = self.predication.eval(feed_dict={self.input: x})
        res = np.array(res)

        predication_file = open(self.log_dir + '/predication/' + str(test_image_id) + '.txt', mode="w")

        count = 0
        label_region_list = []
        # max_prob_index = np.argmax(res, axis=0)[1]

        for i in range(0, 310, 10):
            for j in range(0, 230, 10):
                print("%d %d %d %d %.5lf, %.5lf" % (i, j, i + DataConfig.SUB_REGION_X, j + DataConfig.SUB_REGION_Y,
                                                    res[count][0], res[count][1]), file=predication_file)
                # if count == max_prob_index:
                #     label_region_list.append((i, j, i+DataConfig.SUB_REGION_X, j + DataConfig.SUB_REGION_Y))
                if res[count][1] - res[count][0] > 0.2:
                    label_region_list.append((i, j, i + DataConfig.SUB_REGION_X, j + DataConfig.SUB_REGION_Y))
                count += 1

        predication_file.close()

        self.data.draw_new_label(image_id=test_image_id, region_list=label_region_list)

    def debug(self):
        for x, label in self.data.return_one_batch_data():
            res = self.conv3.eval(feed_dict={self.input: x})
            res = self.input.eval(feed_dict={self.input: x})
            res = self.conv1.eval(feed_dict={self.input: x})

    @staticmethod
    def conv2d(x, filer, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, filer, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    @staticmethod
    def maxpool2d(x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                              padding='SAME')

    def log_config(self):
        temp = DataConfig()
        print(json.dumps(temp, default=DataConfig.save_to_dict, indent=4), file=self.config_log_file)

        temp = conf()
        print(json.dumps(temp, default=conf.save_to_dict, indent=4),file=self.config_log_file)

    def end(self):
        self.loss_log_file.close()
        self.config_log_file.close()

if __name__ == '__main__':
    a = Data(label_dir="../../data/label.md")
    model = Model(data=a)
    model.log_config()

    model.train()
    model.test(test_image_id=21)
    model.test(test_image_id=350)
    model.end()
    # model.debug()

    #Do a demo of results:
