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
            self.training = tf.placeholder(tf.bool)

            # input placeholders
            self.X = tf.placeholder(tf.float32, [None, 784])
            X_img = tf.reshape(self.X, [-1, 28, 28, 1]) # img 28x28x1 (black & white)
            self.Y = tf.placeholder(tf.float32, [None, 10])

            # Convolutional Layer #1
            conv1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], padding='SAME', strides=2)
            dropout1 = tf.layers.dropout(inputs=pool1, rate=0.7, training=self.training)

            # Convolutional Layer #2
            conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], padding='SAME', strides=2)
            dropout2 = tf.layers.dropout(inputs=pool2, rate=0.7, training=self.training)

            # Convolutional Layer #1
            conv3 = tf.layers.conv2d(inputs=dropout2, filters=128, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], padding='SAME', strides=2)
            dropout3 = tf.layers.dropout(inputs=pool3, rate=0.7, training=self.training)

            flat = tf.reshape(dropout3, [-1, 4 * 4 * 128])
            dense4 = tf.layers.dense(inputs=flat, units=625, activation=tf.nn.relu)
            dropout4 = tf.layers.dropout(inputs=dense4, rate=0.5, training=self.training)

            self.hypothesis = tf.layers.dense(inputs=dropout4, units=10)

        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.hypothesis, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        # Test model and check accuracy
        correct_prediction = tf.equal(tf.argmax(self.hypothesis, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, training=False):
        return self.sess.run(self.hypothesis,
                            feed_dict={self.X: x_test, self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.accuracy,
                            feed_dict={self.X: x_test, self.Y: y_test, self.training: training})
    
    def train(self, x_data, y_data, training=True):
        return self.sess.run([self.cost, self.optimizer],
                            feed_dict={self.X: x_data, self.Y: y_data, self.training: training})

with tf.Session() as sess:
    models = []
    num_models = 7
    for m in range(num_models):
        models.append(Model(sess, 'model' + str(m)))

    sess.run(tf.global_variables_initializer())

    print('Learning Started!')
    for epoch in range(training_epochs):
        avg_cost_list = np.zeros(len(models))
        total_batch = int(mnist.train.num_examples / batch_size)
        
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            # train each model
            for m_idx, m in enumerate(models):
                c, _ = m.train(batch_xs, batch_ys)
                avg_cost_list[m_idx] += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', avg_cost_list)
    
    print('Learning Finished!')

    test_size = len(mnist.test.labels)
    predictions = np.zeros(test_size*10).reshape(test_size, 10)

    for m_idx, m in enumerate(models):
        print('Accuracy: ', m.get_accuracy(mnist.test.images, mnist.test.labels))
        p = m.predict(mnist.test.images)
        predictions += p

    ensamble_correct_prediction = tf.equal(
        tf.argmax(predictions, 1), tf.argmax(mnist.test.labels, 1))
    ensamble_accuracy = tf.reduce_mean(
        tf.cast(ensamble_correct_prediction, tf.float32))
    print('Ensamble Accuracy: ', sess.run(ensamble_accuracy))