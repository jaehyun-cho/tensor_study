import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

class Model:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()
    
    def _build_net(self):
        with tf.variable_scope(self.name):
            # input placeholders
            self.X = tf.placeholder(tf.float32, [None, 784])
            X_img = tf.reshape(self.X, [-1, 28, 28, 1]) # img 28x28x1 (black & white)
            self.Y = tf.placeholder(tf.float32, [None, 10])
            self.keep_prob = tf.placeholder(tf.float32)

            # L1 ImgIn shape=(?, 28, 28, 1)
            W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
            #   Conv     -> (?, 28, 28, 32)
            #   Pool     -> (?. 14, 14, 32)
            L1_conv = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
            L1_relu = tf.nn.relu(L1_conv)
            L1_pool = tf.nn.max_pool(L1_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            L1_drop = tf.nn.dropout(L1_pool, keep_prob=self.keep_prob)

            # L2 ImgIn shape = (?, 14, 14, 32)
            W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
            #   Conv     -> (?, 28, 28, 32)
            #   Pool     -> (?. 14, 14, 32)
            L2_conv = tf.nn.conv2d(L1_drop, W2, strides=[1, 1, 1, 1], padding='SAME')
            L2_relu = tf.nn.relu(L2_conv)
            L2_pool = tf.nn.max_pool(L2_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            L2_drop = tf.nn.dropout(L2_pool, keep_prob=self.keep_prob)

            # L2 ImgIn shape = (?, 7, 7, 32)
            W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
            #   Conv     -> (?, 7, 7, 32)
            #   Pool     -> (?. 4, 4, 32)
            L3_conv = tf.nn.conv2d(L2_drop, W3, strides=[1, 1, 1, 1], padding='SAME')
            L3_relu = tf.nn.relu(L3_conv)
            L3_pool = tf.nn.max_pool(L3_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            L3_drop = tf.nn.dropout(L3_pool, keep_prob=self.keep_prob)
            L3 = tf.reshape(L3_drop, [-1, 4 * 4 * 128])

            # Final FC 4x4x128 input -> 625 outputs
            W4 = tf.get_variable('W4', shape=[4*4*128, 625],
                                initializer=tf.contrib.layers.xavier_initializer())
            b4 = tf.Variable(tf.random_normal([625]))
            L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
            L4 = tf.nn.dropout(L4, keep_prob=self.keep_prob)

            # Final FC 625 input -> 10 outputs
            W5 = tf.get_variable('W5', shape=[625, 10],
                                initializer=tf.contrib.layers.xavier_initializer())
            b5 = tf.Variable(tf.random_normal([10]))
            self.hypothesis = tf.matmul(L4, W5) + b5

        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.hypothesis, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        # Test model and check accuracy
        correct_prediction = tf.equal(tf.argmax(self.hypothesis, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, keep_prop=1.0):
        return self.sess.run(self.hypothesis,
                            feed_dict={self.X: x_test, self.keep_prob: keep_prop})

    def get_accuracy(self, x_test, y_test, keep_prop=1.0):
        return self.sess.run(self.accuracy,
                            feed_dict={self.X: x_test, self.Y: y_test, self.keep_prob: keep_prop})
    
    def train(self, x_data, y_data, keep_prop=0.7):
        return self.sess.run([self.cost, self.optimizer],
                            feed_dict={self.X: x_data, self.Y: y_data, self.keep_prob: keep_prop})

with tf.Session() as sess:
    m1 = Model(sess, 'm1')

    sess.run(tf.global_variables_initializer())

    print('Learning Started!')
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)
        
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = m1.train(batch_xs, batch_ys)
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
    
    print('Learning Finished!')
    print('Accuracy: ', m1.get_accuracy(mnist.test.images, mnist.test.labels))