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

X = tf.placeholder(tf.int32, [None, seq_length])  # X data
Y = tf.placeholder(tf.int32, [None, seq_length])  # Y Label
X_one_hot = tf.one_hot(X, num_classes)  # one hot: 1 -> 0 1 0 0 0 0 0 0 0 0
print(X_one_hot)

cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
cell = tf.contrib.rnn.MultiRNNCell([cell]*2, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, X_one_hot, initial_state=initial_state, dtype=tf.float32)

# for fully connected layer connect!
X_for_softmax = tf.reshape(outputs, [-1, hidden_size])  # -1 -> 알아서!
softmax_w = tf.get_variable('softmax_w', [hidden_size, num_classes])
softmax_b = tf.get_variable('softmax_b', [num_classes])
outputs = tf.matmul(X_for_softmax, softmax_w) + softmax_b

# 다시 펼치기! reshape out for sequence_loss
outputs = tf.reshape(outputs, [batch_size, seq_length, num_classes])

weights = tf.ones([batch_size, seq_length])

sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(500):
        l, _ , results= sess.run([loss, train, outputs], feed_dict={X: dataX, Y: dataY})

        if i % 10 == 0:
            print('result: ', results)
            for j, result in enumerate(results):
                index = np.argmax(result, axis=1)
                result_str = [char_set[c] for c in index]
                print(i, j, 'loss: ', l, 'prediction: ', ''.join(result_str))

    # Let's print the last char of each result to check it works
    results = sess.run(outputs, feed_dict={X: dataX})
    for j, result in enumerate(results):
        index = np.argmax(result, axis=1)
        if j is 0:  # print all for the first result to make a sentence
            print(''.join([char_set[t] for t in index]), end='')
        else:
            print(char_set[index[-1]], end='')