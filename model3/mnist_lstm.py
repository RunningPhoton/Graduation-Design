import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data


# 数据
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# 学习率
lr = 1e-3
# 每个时刻的输入特征是28维的，就是每个时刻输入一行，一行有28个像素
input_size = 28
# 时序持续长度为28，即每做一次预测，需要先输入28行
timestep_size = 28
display_step = 100
epochs = 2000
# 每个隐含层的节点数
# hidden_size
# LSTM layer 的层数
# layer_num
# 最后输出分类类别数量
class_num = 10

# 定义输入层
def build_inputs(class_num):
    x = tf.placeholder(tf.float32, [None, 784], name='inputs')
    y = tf.placeholder(tf.float32, [None, class_num], name='targets')
    keep_prob = tf.placeholder(tf.float32, [])
    # 在训练和测试的时候，想用不同的 batch_size.所以采用占位符的方式
    batch_size = tf.placeholder(tf.int32, [])
    return x, y, keep_prob, batch_size

# 定义lstm网络结构
def built_lstm(rnn_size, num_layer, keep_probability, batch_size):

    def get_cell(rnn_size, keep_probability):
        lstm_cell = rnn.LSTMCell(num_units=rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2),
                                 forget_bias=1.0, state_is_tuple=True)
        cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_probability)
        return cell

    mlstm_cell = rnn.MultiRNNCell([get_cell(rnn_size, keep_probability) for _ in range(num_layer)], state_is_tuple=True)

    init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)

    return mlstm_cell, init_state

# 定义输出层
def build_output(lstm_state, in_size, out_size):
    with tf.variable_scope('softmax'):
        softmax_w = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1), dtype=tf.float32)
        softmax_b = tf.Variable(tf.constant(0.1, shape=[out_size]), dtype=tf.float32)
    logits = tf.matmul(lstm_state, softmax_w) + softmax_b

    output = tf.nn.softmax(logits)
    return output, logits

# 建立损失函数
def build_loss(predictions, targets):
    cross_entropy = -tf.reduce_mean(targets * tf.log(predictions))
    return cross_entropy

# 定义准确度
def build_accuracy(prediction, targets):
    corect_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(targets, 1))
    accuracy = tf.reduce_mean(tf.cast(corect_prediction, "float"))
    return accuracy

# 定义优化器，包括梯度裁剪操作
def build_optimizer(loss, learning_rate, grad_clip):
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
    train_op = tf.train.AdamOptimizer(learning_rate)
    optimizer = train_op.apply_gradients(zip(grads, tvars))

    return optimizer

def output_msg(f_name, msg):
    f = open(f_name, "a+");
    f.write(msg + '\n')
    f.close();
# 连接模块搭建网络
class LSTM:
    def __init__(self, lstm_size, num_layers, num_classes = class_num,
                 num_steps = timestep_size, learning_rate = lr, grad_clip = 5):

        # 初始化
        tf.reset_default_graph()

        # 输入层
        self.inputs, self.targets, self.keep_prob, self.batch_size = build_inputs(num_classes)

        # LSTM层
        mlstm_cell, self.initial_state = built_lstm(lstm_size, num_layers, self.keep_prob, self.batch_size)

        # 对输入进行处理
        X = tf.reshape(self.inputs, [-1, 28, 28])

        # 运行LSTM
        outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=X, initial_state=self.initial_state, time_major=False)
        self.final_state = outputs[:, -1, :]

        # 预测结果
        self.prediction, self.logits = build_output(self.final_state, lstm_size, num_classes)

        # Loss
        self.loss = build_loss(self.prediction, self.targets)

        # accuracy
        self.accuracy = build_accuracy(self.prediction, self.targets)

        # optimizer
        self.optimizer = build_optimizer(self.loss, learning_rate, grad_clip)

# lstm_size_list = [32, 64, 128, 192, 256, 320]
# num_layer_list = [1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20]
lstm_size_list = [192]
num_layer_list = [18]
for lstm_size in lstm_size_list:
    # out_file_name = "out_lstm_size_" + str(lstm_size) + ".txt"
    # f = open(out_file_name, "w+")
    # message = 'batch_size = ' + str(128) + '\nlstm_size = ' + lstm_size.__str__() + '\nlearning_rate = ' + lr.__str__()
    # f.write(message)
    # f.close()
    for num_layer in num_layer_list:
        # output_msg(out_file_name, "\n当前隐藏层layer数量： {}\n".format(num_layer))
        model = LSTM(lstm_size, num_layer)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch_i in range(epochs):
                _batch_size = 128
                batch = mnist.train.next_batch(_batch_size)

                if (epoch_i + 1) % display_step == 0:
                    train_accuracy, train_loss = sess.run([model.accuracy, model.loss], feed_dict={
                        model.inputs: batch[0], model.targets: batch[1], model.keep_prob: 1.0, model.batch_size: _batch_size})

                    validation_batch = mnist.validation.next_batch(_batch_size)
                    validation_accuracy, validation_loss = sess.run([model.accuracy, model.loss], feed_dict={
                        model.inputs: validation_batch[0], model.targets: validation_batch[1], model.keep_prob: 1.0,
                        model.batch_size: _batch_size})
                    msg = 'Epoch {:>3}/{} - Training Loss: {:>6.6f} - Training Accuracy: {:>6.6f} ' \
                              '- Validation loss: {:>6.6f} - Validation Accuracy: {:>6.6f}'\
                            .format(epoch_i + 1, epochs, train_loss, train_accuracy, validation_loss, validation_accuracy)
                    print(msg)
                    # output_msg(out_file_name, msg)
                sess.run(model.optimizer, feed_dict={
                    model.inputs: batch[0],
                    model.targets: batch[1],
                    model.keep_prob: 0.5,
                    model.batch_size: _batch_size})
            # 计算测试数据的准确率
            msg = 'lstm_size: {} - layer_num: {} test_accuracy: {}'.format(
                lstm_size,
                num_layer,
                sess.run(model.accuracy, feed_dict={
                    model.inputs: mnist.test.images,
                    model.targets: mnist.test.labels,
                    model.keep_prob: 1.0,
                    model.batch_size: mnist.test.images.shape[0]})
            )
            print(msg)
            # output_msg(out_file_name, msg)