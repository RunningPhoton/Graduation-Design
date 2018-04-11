import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 超参数
# 学习率
lr = 0.001
# 遍历数
epochs = 50
# dropout保留率
keep_prob = 0.8
batch_size = 100
# 读取的单位是图中一行像素，28个
input_x = 28
# 每张图像为28×28,而每一个序列长度为1×28,所以总共28步,
time_steps = 28
# 输入为10类
output_y = 10
# 隐层大小
rnn_size = 128
# lstm隐藏层数
num_layers = 3
# 每display_step输出一次
display_step = 100
x = tf.placeholder(dtype=tf.float32, shape=[None, time_steps, input_x], name='input_x')
y = tf.placeholder(dtype=tf.float32, shape=[None, output_y], name='target_y')

weights_in = tf.Variable(tf.random_normal([input_x, rnn_size]), name='weights')
weights_out = tf.Variable(tf.random_normal([rnn_size, output_y]), name='weights')

bias_in = tf.Variable(tf.zeros([rnn_size]), name='bias')
bias_out = tf.Variable(tf.zeros([output_y]), name='bias')

val_data = mnist.validation.images[:1000].reshape((-1, 28, 28))
val_label = mnist.validation.labels[:1000]

test_data = mnist.test.images[:1000].reshape((-1, 28, 28))
test_label = mnist.test.labels[:1000]
# 定义LSTM网络
def built_lstm(input_data, weights_in, weights_out, bias_in, bias_out):
    # 进行数据处理, 一个输入数据格式是batch_size * time_step * input_x
    x_in = tf.reshape(input_data, [-1, input_x])
    x_in = tf.matmul(x_in, weights_in) + bias_in
    x_in = tf.reshape(x_in, [time_steps, batch_size, rnn_size])
    # RNN cell
    def get_lstm_cell(rnn_size):
        lstm_cell = tf.contrib.rnn.LSTMCell(
            rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        drop = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
        return drop
    multiple_cell = tf.contrib.rnn.MultiRNNCell([get_lstm_cell(rnn_size) for _ in range(num_layers)])
    init_state = multiple_cell.zero_state(batch_size, tf.float32)
    lstm_output, lstm_state = tf.nn.dynamic_rnn(
        multiple_cell, x_in, initial_state=init_state, time_major=True, dtype=tf.float32)
    # 选择lstm_output最后一个输出
    output = tf.add(tf.matmul(lstm_output[-1], weights_out), bias_out)
    return output

# 定义损失函数以及优化器
y_pred = built_lstm(x, weights_in, weights_out, bias_in, bias_out)
print(y_pred)
print(y)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1)), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    n_batches = mnist.train.num_examples // batch_size
    # writer = tf.summary.FileWriter('./graphs/lstm_mnist', sess.graph)
    for epoch in range(epochs):
        total_loss = 0
        for i in range(n_batches):
            xs, ys = mnist.train.next_batch(batch_size)
            # 因为xs的shape是(None,784)的我们需要reshape一下
            xs = xs.reshape((batch_size, time_steps, input_x))
            _, tmp_loss = sess.run([optimizer, loss], feed_dict={x: xs, y: ys})
            total_loss += tmp_loss
            if epoch % display_step == 0:
                train_acc = sess.run(accuracy, feed_dict={x: xs, y: ys})
                val_loss, val_acc = \
                    sess.run([loss, accuracy], feed_dict={x: val_data, y: val_label})

                print('Epoch  {}/{:.3f} train loss {:.3f},train_acc {:.3f},val_loss {:.3f},val_acc {:.3f} '.
                      format(epoch, epochs, total_loss/n_batches, train_acc/n_batches, val_loss, val_acc))
    test_loss, test_acc = sess.run([loss, accuracy], feed_dict={x: test_data, y: test_label})
    print('test_loss {:.3f},test_acc {:.3f}'.format(test_loss, test_acc))
    # writer.close()
