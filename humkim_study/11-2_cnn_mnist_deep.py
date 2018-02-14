import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# input placeholders
X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1]) # img 28x28x1 (black & white)
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

# L1 ImgIn shape=(?, 28, 28, 1)
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
#   Conv     -> (?, 28, 28, 32)
#   Pool     -> (?. 14, 14, 32)
L1_conv = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
L1_relu = tf.nn.relu(L1_conv)
L1_pool = tf.nn.max_pool(L1_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L1_drop = tf.nn.dropout(L1_pool, keep_prob=keep_prob)

# L2 ImgIn shape = (?, 14, 14, 32)
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
#   Conv     -> (?, 28, 28, 32)
#   Pool     -> (?. 14, 14, 32)
L2_conv = tf.nn.conv2d(L1_drop, W2, strides=[1, 1, 1, 1], padding='SAME')
L2_relu = tf.nn.relu(L2_conv)
L2_pool = tf.nn.max_pool(L2_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L2_drop = tf.nn.dropout(L2_pool, keep_prob=keep_prob)

# L2 ImgIn shape = (?, 7, 7, 32)
W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
#   Conv     -> (?, 7, 7, 32)
#   Pool     -> (?. 4, 4, 32)
L3_conv = tf.nn.conv2d(L2_drop, W3, strides=[1, 1, 1, 1], padding='SAME')
L3_relu = tf.nn.relu(L3_conv)
L3_pool = tf.nn.max_pool(L3_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L3_drop = tf.nn.dropout(L3_pool, keep_prob=keep_prob)
L3 = tf.reshape(L3_drop, [-1, 4 * 4 * 128])

# Final FC 4x4x128 input -> 625 outputs
W4 = tf.get_variable('W4', shape=[4*4*128, 625],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([625]))
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

# Final FC 625 input -> 10 outputs
W5 = tf.get_variable('W5', shape=[625, 10],
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L4, W5) + b5

learning_rate = 0.001

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# parameters
training_epochs = 15
batch_size = 100

# initialize
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    # train my model
    print('Learning stared. It takes sometime.')
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)
        
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.7}
            c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            avg_cost += c / total_batch
        
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
    
    print('Learning Finished!')
    
    # Test model and check accuracy
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accuracy:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.0}))