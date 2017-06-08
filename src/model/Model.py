import tensorflow as tf
from modelConfig import ModelConfig as conf

class Model(object):

    def __init__(self, data):

        self.sess = tf.InteractiveSession()
        self.data = data

        self.input = tf.placeholder(tf.float32, [None, conf.INPUT_X_SIZE, conf.INPUT_Y_SIZE, conf.INPUT_CHANNEL])
        self.label = tf.placeholder(tf.float32, [None, conf.OUTPUT_SIZE])
        self.is_training = tf.placeholder(tf.bool)
        self.predication = self.create_net_work()
        self.loss, self.optimizer = self.create_train_method()

        self.weight = {
            'conv1_filter': tf.Variable(tf.random_normal([conf.FILTER_SIZE, conf.FILTER_SIZE, conf.INPUT_CHANNEL, conf.LAYER1_CHANNEL])),
            'conv2_filter': tf.Variable(tf.random_normal([conf.FILTER_SIZE, conf.FILTER_SIZE, conf.LAYER1_CHANNEL, conf.LAYER2_CHANNEL])),
            'conv3_filter': tf.Variable(tf.random_normal([conf.FILTER_SIZE, conf.FILTER_SIZE, conf.LAYER2_CHANNEL, conf.LAYER3_CHANNEL])),
            'fc1_weight': tf.Variable(tf.random_normal([conf.FULLY_CONNECTED_INPUT_SIZE, conf.FULLY_CONNECTED_OUT_SIZE])),
            'out_weight': tf.Variable(tf.random_normal([conf.OUT_LAYER_INPUT_SIZE, conf.OUT_LAYER_OUT_SIZE]))
        }

        self.bias = {
            'conv1': tf.Variable(tf.random_normal(conf.LAYER1_CHANNEL)),
            'conv2': tf.Variable(tf.random_normal(conf.LAYER2_CHANNEL)),
            'conv3': tf.Variable(tf.random_normal(conf.LAYER3_CHANNEL)),
            'fc1': tf.Variable(tf.random_normal(conf.FULLY_CONNECTED_OUT_SIZE)),
            'out': tf.Variable(tf.random_normal(conf.OUT_LAYER_OUT_SIZE))
        }

    def create_net_work(self):
        conv1 = self.conv2d(x=self.input, filer=self.weight['conv1_filter'], b=self.bias['conv1'], strides=conf.CONVOLUTIONAL_STRIDES)
        conv1 = self.maxpool2d(x=conv1, k=conf.MAX_POOLING_STRIDES)

        conv2 = self.conv2d(x=conv1, filer=self.weight['conv2_filter'], b=self.bias['conv2'], strides=conf.CONVOLUTIONAL_STRIDES)
        conv2 = self.maxpool2d(x=conv2, k=conf.MAX_POOLING_STRIDES)

        conv3 = self.conv2d(x=conv2, filer=self.weight['conv3_filter'], b=self.bias['conv3'], strides=conf.CONVOLUTIONAL_STRIDES)
        conv3 = self.maxpool2d(x=conv3, k=conf.MAX_POOLING_STRIDES)

        fc1 = tf.reshape(conv3, [-1, conf.FULLY_CONNECTED_INPUT_SIZE])
        fc1 = tf.add(tf.matmul(fc1, self.weight['fc1_weight']), self.bias['fc1'])
        fc1 = tf.nn.relu(fc1)

        if self.is_traning is True:
            fc1 = tf.nn.dropout(fc1, conf.DROPOUT_PROBABILITY)

        out = tf.add(tf.matmul(fc1, self.weight['out_weight']), self.bias['out'])
        out = tf.nn.softmax(out)
        return out

    def compute_accuracy(self):

        pass

    def create_train_method(self):
        loss = tf.reduce_mean(tf.square(self.label - self.predication))

        optimizer = tf.train.AdagradOptimizer(learning_rate=conf.LEARNING_RATE).minimize(loss)

        return loss, optimizer

    def train(self, data):

        _, cost = self.sess.run([self.optimizer, self.loss], feed_dict={})


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
