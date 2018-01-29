import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import random
tf.set_random_seed(777)  # for reproducibility

# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
batch_xs, batch_ys = mnist.train.next_batch(100)


nb_classes = 10
# MNIST data image of shape 28 * 28 = 784
X = tf.placeholder(tf.float32, [None, 784])
# 0 - 9 digits recognition = 10 classes
Y = tf.placeholder(tf.float32, [None, nb_classes])

with tf.name_scope("layer1") as scope:
    W1 = tf.Variable(tf.random_normal([784, 256]), name='weight1')
    b1 = tf.Variable(tf.random_normal([256]), name='bias1')
    layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

    W1_hist = tf.summary.histogram("weight1", W1)
    b1_hist = tf.summary.histogram("biases1", b1)
    layer1_hist = tf.summary.histogram("layer1", layer1)

with tf.name_scope("layer2") as scope:
    W2 = tf.Variable(tf.random_normal([256, nb_classes]), name='weight2')
    b2 = tf.Variable(tf.random_normal([nb_classes]), name='bias2')

    # tf.sigmoid computes softmax activations
    # softmax = exp(logits) / reduce_sum(exp(logits), dim)
    logits = tf.matmul(layer1, W2) + b2
    hypothesis = tf.sigmoid(logits)

    W2_hist = tf.summary.histogram("weight2", W2)
    b2_hist = tf.summary.histogram("biases2", b2)
    hypothesis_hist = tf.summary.histogram("hypothesis", hypothesis)

# Cross entropy cost/loss function
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

cost_summ = tf.summary.scalar("cost", cost)

# Test model
is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))


# parameters
training_epochs = 15
batch_size = 100

with tf.Session() as sess:
    summary = tf.summary.merge_all()
    # Create summary writer
    writer = tf.summary.FileWriter('./logs/mnist_logs')
    writer.add_graph(sess.graph)

    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += c / total_batch

        print('Epoch: ', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

    print ("Accuracy: ", accuracy.eval(session=sess,
            feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
            
    # Get one and predict
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label:", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
    print("Prediction:", sess.run(tf.argmax(hypothesis, 1), 
                            feed_dict={X: mnist.test.images[r:r+1]}))

    plt.imshow(mnist.test.images[r:r+1].
                reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()