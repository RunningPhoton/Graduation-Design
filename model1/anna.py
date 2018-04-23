# coding: utf-8


# In[1]:

import time
from collections import namedtuple

import numpy as np
import tensorflow as tf

# # 1 数据加载与预处理

# In[2]:

with open('anna.txt', 'r') as f:
    text = f.read()
vocab = set(text)
vocab_to_int = {c: i for i, c in enumerate(vocab)}
int_to_vocab = dict(enumerate(vocab))
encoded = np.array([vocab_to_int[c] for c in text], dtype=np.int32)
batch_size = 100  # Sequences per batch
num_steps = 100  # Number of sequence steps per batch
learning_rate = 0.001  # Learning rate
keep_prob = 0.5  # Dropout keep probability
validation_encoded = encoded[:batch_size*num_steps]
encoded = encoded[batch_size*num_steps:]
# In[3]:

# text[:100]

# In[4]:

# encoded[:100]

# In[5]:

# len(vocab)


# # 2 分割mini-batch
#
#
#
# <img src="assets/sequence_batching@1x.png" width=500px>
#
#
# 完成了前面的数据预处理操作，接下来就是要划分我们的数据集，在这里我们使用mini-batch来进行模型训练，那么我们要如何划分数据集呢？在进行mini-batch划分之前，我们先来了解几个概念。
#
# 假如我们目前手里有一个序列1-12，我们接下来以这个序列为例来说明划分mini-batch中的几个概念。首先我们回顾一下，在DNN和CNN中，我们都会将数据分batch输入给神经网络，加入我们有100个样本，如果设置我们的batch_size=10，那么意味着每次我们都会向神经网络输入10个样本进行训练调整参数。同样的，在LSTM中，batch_size意味着每次向网络输入多少个样本，在上图中，当我们设置batch_size=2时，我们会将整个序列划分为6个batch，每个batch中有两个数字。
#
# 然而由于RNN中存在着“记忆”，也就是循环。事实上一个循环神经网络能够被看做是多个相同神经网络的叠加，在这个系统中，每一个网络都会传递信息给下一个。上面的图中，我们可以看到整个RNN网络由三个相同的神经网络单元叠加起来的序列。那么在这里就有了第二个概念sequence_length（也叫steps），中文叫序列长度。上图中序列长度是3，可以看到将三个字符作为了一个序列。
#
# 有了上面两个概念，我们来规范一下后面的定义。我们定义一个batch中的序列个数为N（batch_size），定义单个序列长度为M（也就是我们的steps）。那么实际上我们每个batch是一个N x M的数组。在这里我们重新定义batch_size为一个N x M的数组，而不是batch中序列的个数。在上图中，当我们设置N=2， M=3时，我们可以得到每个batch的大小为2 x 3 = 6个字符，整个序列可以被分割成12 / 6 = 2个batch。

# In[6]:
def get_validation_batch(arr, n_seqs, n_steps):
    arr = arr[: n_seqs * n_steps]
    arr = arr.reshape((n_seqs, -1))
    x = arr
    y = np.zeros_like(x)
    y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
    return x, y
def get_batches(arr, n_seqs, n_steps):
    '''
    对已有的数组进行mini-batch分割

    arr: 待分割的数组
    n_seqs: 一个batch中序列个数
    n_steps: 单个序列包含的字符数
    '''

    batch_size = n_seqs * n_steps
    n_batches = int(len(arr) / batch_size)
    # 这里我们仅保留完整的batch，对于不能整出的部分进行舍弃
    arr = arr[:batch_size * n_batches]

    # 重塑
    arr = arr.reshape((n_seqs, -1))

    # print('batch start\n')
    for n in range(0, arr.shape[1], n_steps):
        # inputs
        x = arr[:, n:n + n_steps]
        # targets
        y = np.zeros_like(x)
        y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
        yield x, y


# 上面的代码定义了一个generator，调用函数会返回一个generator对象，我们可以获取一个batch。

# In[7]:

# batches = get_batches(encoded, 10, 50)
# x, y = next(batches)
#
# # In[8]:
#
# print('x\n', x[:10, :10])
# print('\ny\n', y[:10, :10])


# # 3 模型构建
# 模型构建部分主要包括了输入层，LSTM层，输出层，loss，optimizer等部分的构建，我们将一块一块来进行实现。

# ## 3.1 输入层

# In[9]:

def build_inputs(ns):
    '''
    构建输入层

    num_seqs: 每个batch中的序列个数
    num_steps: 每个序列包含的字符数
    '''

    bs = tf.placeholder(tf.int32, [], name='batch_size')
    inputs = tf.placeholder(tf.int32, shape=(None, ns), name='inputs')
    targets = tf.placeholder(tf.int32, shape=(None, ns), name='targets')

    # 加入keep_prob
    kb = tf.placeholder(tf.float32, name='keep_prob')

    return inputs, targets, kb, bs


# ## 3.2 LSTM层

# In[10]:

def build_lstm(lstm_size, num_layers, batch_size, keep_prob):
    '''
    构建lstm层

    keep_prob
    lstm_size: lstm隐层中结点数目
    num_layers: lstm的隐层数目
    batch_size: batch_size

    '''
    drop_stacks = []
    for i in range(num_layers):

        # 构建一个基本lstm单元
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size, state_is_tuple=True)

        # 添加dropout
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)

        drop_stacks.append(drop)
    # 堆叠
    cell = tf.contrib.rnn.MultiRNNCell(drop_stacks, state_is_tuple=True)
    initial_state = cell.zero_state(batch_size, tf.float32)

    return cell, initial_state


# ## 3.3 输出层

# In[11]:

def build_output(lstm_output, in_size, out_size):
    '''
    构造输出层

    lstm_output: lstm层的输出结果
    in_size: lstm输出层重塑后的size
    out_size: softmax层的size

    '''

    # 将lstm的输出按照列concate，例如[[1,2,3],[7,8,9]],
    # tf.concat的结果是[1,2,3,7,8,9]
    seq_output = tf.concat(lstm_output, axis=1)  # tf.concat(concat_dim, values)
    # reshape
    x = tf.reshape(seq_output, [-1, in_size])

    # 将lstm层与softmax层全连接
    with tf.variable_scope('softmax'):
        softmax_w = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(out_size))

    # 计算logits
    logits = tf.matmul(x, softmax_w) + softmax_b

    # softmax层返回概率分布
    out = tf.nn.softmax(logits, name='predictions')

    return out, logits


# ## 3.4 训练误差计算

# In[12]:

def build_loss(logits, targets, bs, num_classes):
    '''
    根据logits和targets计算损失

    logits: 全连接层的输出结果（不经过softmax）
    targets: targets
    lstm_size
    num_classes: vocab_size

    '''

    # One-hot编码
    y_one_hot = tf.one_hot(targets, num_classes)
    y_reshaped = tf.reshape(y_one_hot, [-1, num_classes])

    # Softmax cross entropy loss
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
    loss = tf.reduce_mean(loss)

    return loss


# ## 3.5 Optimizer
# 我们知道RNN会遇到梯度爆炸（gradients exploding）和梯度弥散（gradients disappearing)的问题。LSTM解决了梯度弥散的问题，但是gradient仍然可能会爆炸，因此我们采用gradient clippling的方式来防止梯度爆炸。即通过设置一个阈值，当gradients超过这个阈值时，就将它重置为阈值大小，这就保证了梯度不会变得很大。

# In[13]:

def build_optimizer(loss, learning_rate, grad_clip):
    '''
    构造Optimizer

    loss: 损失
    learning_rate: 学习率

    '''

    # 使用clipping gradients
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
    train_op = tf.train.AdamOptimizer(learning_rate)
    optimizer = train_op.apply_gradients(zip(grads, tvars))

    return optimizer


# ## 3.6 模型组合
# 使用tf.nn.dynamic_run来运行RNN序列

# In[14]:

class CharRNN:
    def __init__(self, num_classes, num_steps=50,
                 lstm_size=128, num_layers=2, learning_rate=0.001,
                 grad_clip=5):

        tf.reset_default_graph()

        # 输入层
        self.inputs, self.targets, self.keep_prob, self.batch_size = build_inputs(num_steps)

        # LSTM层
        cell, self.initial_state = build_lstm(lstm_size, num_layers, self.batch_size, self.keep_prob)

        # 对输入进行one-hot编码
        x_one_hot = tf.one_hot(self.inputs, num_classes)

        # 运行RNN
        outputs, state = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=self.initial_state)
        self.final_state = state

        # 预测结果
        self.prediction, self.logits = build_output(outputs, lstm_size, num_classes)

        # Loss 和 optimizer (with gradient clipping)
        self.loss = build_loss(self.logits, self.targets, self.batch_size, num_classes)
        self.optimizer = build_optimizer(self.loss, learning_rate, grad_clip)


# # 4 模型训练

# ### 参数设置
# 在模型训练之前，我们首先初始化一些参数，我们的参数主要有：
#
# - num_seqs: 单个batch中序列的个数
# - num_steps: 单个序列中字符数目
# - lstm_size: 隐层结点个数
# - num_layers: LSTM层个数
# - learning_rate: 学习率
# - keep_prob: dropout层中保留结点比例

# In[15]:

def output_message(file_name, msg):
    with open(file_name, "a+", encoding='utf-8') as f:
        f.write(msg)
# In[16]:
dir = '01/'
# lstm_sizes = [128, 192, 256, 384, 512]
lstm_sizes = [600]
for lstm_size in lstm_sizes:
    out_file_name = dir + "out_lstm_size_" + str(lstm_size) + ".txt"
    f = open(out_file_name, "w+", encoding='utf-8')
    message = 'batch_size = '+batch_size.__str__()+'\nnum_steps = '+num_steps.__str__()+'\nlstm_size = '+lstm_size.__str__()+'\nlearning_rate = '+learning_rate.__str__()+'\nkeep_prob'+keep_prob.__str__()
    f.write(message)
    f.close()
    for num_layers in range(1, 14):

        output_message(out_file_name, "\n当前隐藏层layer数量： %d\n" % num_layers)
        epochs = 40

        model = CharRNN(len(vocab), num_steps=num_steps,
                        lstm_size=lstm_size, num_layers=num_layers,
                        learning_rate=learning_rate)

        validation_batch_size = batch_size
        validation_x, validation_y = get_validation_batch(encoded, batch_size, num_steps)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            counter = 0
            for e in range(epochs):
                # Train network
                # new_state = sess.run(model.initial_state)
                loss = 0
                for x, y in get_batches(encoded, batch_size, num_steps):
                    counter += 1
                    start = time.time()

                    feed = {model.inputs: x,
                            model.targets: y,
                            model.keep_prob: keep_prob,
                            model.batch_size: batch_size}
                    batch_loss, new_state, _ = sess.run([model.loss,
                                                         model.final_state,
                                                         model.optimizer],
                                                        feed_dict=feed)
                    end = time.time()
                    # control the print lines
                    if counter % 100 == 0:
                        validation_loss = sess.run(model.loss,
                                                   feed_dict={
                                                       model.inputs: validation_x,
                                                       model.targets: validation_y,
                                                       model.keep_prob: keep_prob,
                                                       model.batch_size: validation_batch_size
                                                   })
                        msg = '轮数: {}/{}... '.format(e + 1, epochs) + \
                              '训练步数: {}... '.format(counter) + \
                              '训练误差: {:.4f}... '.format(batch_loss) + \
                              '验证误差: {:.4f}...'.format(validation_loss) + \
                              '{:.4f} sec/batch'.format(end - start)
                        print(msg)
                        output_message(out_file_name, msg+'\n')
