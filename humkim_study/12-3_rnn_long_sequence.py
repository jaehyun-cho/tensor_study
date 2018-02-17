import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to logn for the endless immensity of the sea.")
char_set = list(set(sentence))    # index -> char
char_dic = {w: i for i, w in enumerate(char_set)}   # char -> index

print(char_set)

# hyper parameters
data_dim = len(char_set)
hidden_size = len(char_set)
num_classes = len(char_set)
seq_length = 10

dataX = []
dataY = []
for i in range(0, len(sentence) - seq_length):
    x_str = sentence[i: i + seq_length]
    y_str = sentence[i+1:i+seq_length + 1]
    print(i, x_str, '->', y_str)

    x = [char_dic[c] for c in x_str]
    y = [char_dic[c] for c in y_str]

    dataX.append(x)
    dataY.append(y)

batch_size = len(dataX)

# sample_idx = [char2idx[c] for c in sample]  # char to index convert
# x_data = [sample_idx[:-1]]  # X data sample(0 ~ n-1) hello: hell
# y_data = [sample_idx[1:]]   # Y data sample(1 ~ n) hello: ello

# dic_size = len(char2idx)    # RNN input size (one hot size)
# rnn_hidden_size = len(char2idx) # RNN output size
# num_classes = len(idx2char) # final output size (RNN or softmax, etc.)
# batch_size = 1  # one sample data, ont batch
# sequence_length = len(sample) - 1   # number of lstm unfolding (unit #)

X = tf.placeholder(tf.int32, [None, seq_length])  # X data
Y = tf.placeholder(tf.int32, [None, seq_length])  # Y Label
X_one_hot = tf.one_hot(X, num_classes)  # one hot: 1 -> 0 1 0 0 0 0 0 0 0 0
print(X_one_hot)

cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, X_one_hot, initial_state=initial_state, dtype=tf.float32)
weights = tf.ones([batch_size, seq_length])

sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(3000):
        l, _ = sess.run([loss, train], feed_dict={X: dataX, Y: dataY})
        results = sess.run(prediction, feed_dict={X: dataX})

        if i % 10 == 0:
            print('result: ', results)
            for j, r in enumerate(results):
                result_str = [char_set[c] for c in np.squeeze(r)]
                print(i, j, 'loss: ', l, 'prediction: ', ''.join(result_str))

    # Let's print the last char of each result to check it works
    results = sess.run(outputs, feed_dict={X: dataX})
    for j, result in enumerate(results):
        index = np.argmax(result, axis=1)
        if j is 0:  # print all for the first result to make a sentence
            print(''.join([char_set[t] for t in index]), end='')
        else:
            print(char_set[index[-1]], end='')