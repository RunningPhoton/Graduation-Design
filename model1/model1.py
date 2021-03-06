
import time
import numpy as np
import tensorflow as tf

#　数据加载与预处理

with open('data.txt', 'r') as f:
    text = f.read()
vocab = set(text)
vocab_to_int = {c: i for i, c in enumerate(vocab)}
int_to_vocab = dict(enumerate(vocab))
encoded = np.array([vocab_to_int[c] for c in text], dtype=np.int32)
batch_size = 100  # 每个batch中的输入量
num_steps = 100  # 每个sequence的长度
learning_rate = 0.001  # 学习率
keep_prob = 0.5  # Dropout层保存率
epochs = 5
# 验证集

validation_encoded = encoded[:batch_size*num_steps]
encoded = encoded[batch_size*num_steps:]

# 分割mini-batch

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
    # 仅保留完整的batch，对于不能整出的部分进行舍弃
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



# 模型构建
# 模型构建部分主要包括了输入层，LSTM层，输出层，loss，optimizer等部分的构建。

# 输入层

def build_inputs(ns):
    '''
    构建输入层
    '''

    # bs = tf.placeholder(tf.int32, [], name='batch_size')
    inputs = tf.placeholder(tf.int32, shape=(None, ns), name='inputs')
    targets = tf.placeholder(tf.int32, shape=(None, ns), name='targets')

    # 加入keep_prob

    kb = tf.placeholder(tf.float32, name='keep_prob')

    return inputs, targets, kb

# LSTM层

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
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)

        # 添加dropout
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)

        drop_stacks.append(drop)
    # 堆叠
    cell = tf.contrib.rnn.MultiRNNCell(drop_stacks, state_is_tuple=True)
    initial_state = cell.zero_state(batch_size, tf.float32)

    return cell, initial_state


# 输出层

def build_output(lstm_output, in_size, out_size):
    '''
    构造输出层
    lstm_output: lstm层的输出结果
    in_size: lstm输出层重塑后的size
    out_size: softmax层的size
    '''

    # 将lstm的输出按照列concate，例如[[1,2,3],[7,8,9]],
    # tf.concat的结果是[1,2,3,7,8,9]
    seq_output = tf.concat(lstm_output, axis=1)
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

# 训练误差计算

def build_loss(logits, targets, num_classes):
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


# Optimizer
# 梯度消减

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


# 模块组合
# 使用tf.nn.dynamic_run运行RNN模型

class CharRNN:
    def __init__(self, num_classes, num_steps=50,
                 lstm_size=128, num_layers=2, learning_rate=0.001,
                 grad_clip=5, sampling=False, batch_size = 100):


        tf.reset_default_graph()

        # 输入层
        self.batch_size = batch_size
        if sampling == True:
            num_steps = 1
            self.batch_size = 1

        self.inputs, self.targets, self.keep_prob = build_inputs(num_steps)

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
        self.loss = build_loss(self.logits, self.targets, num_classes)
        self.optimizer = build_optimizer(self.loss, learning_rate, grad_clip)


# 模型训练

# 调节的参数主要有：

# - lstm_size: 隐层结点个数
# - num_layers: LSTM层个数

# 文件输出函数

def output_message(file_name, msg):
    with open(file_name, "a+", encoding='utf-8') as f:
        f.write(msg)

# 文件输出目录

dir = '02/'

# lstm核心变化

# lstm_sizes = [128, 192, 256, 384, 512, 600, 768]
checkpoint_dir = 'checkpoints'
lstm_sizes = [512]
for lstm_size in lstm_sizes:
    # out_file_name = dir + "out_lstm_size_" + str(lstm_size) + ".txt"
    # f = open(out_file_name, "w+", encoding='utf-8')
    # message = 'batch_size = '+batch_size.__str__()+'\nnum_steps = '+num_steps.__str__()+'\nlstm_size = '+lstm_size.__str__()+'\nlearning_rate = '+learning_rate.__str__()+'\nkeep_prob'+keep_prob.__str__()
    # f.write(message)
    # f.close()

    # lstm层数变化

    for num_layers in range(2, 3):

        # output_message(out_file_name, "\n当前隐藏层layer数量： %d\n" % num_layers)


        model = CharRNN(len(vocab), num_steps=num_steps,
                        lstm_size=lstm_size, num_layers=num_layers,
                        learning_rate=learning_rate, batch_size=batch_size)


        validation_batch_size = batch_size
        validation_x, validation_y = get_validation_batch(encoded, batch_size, num_steps)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()

            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('loaded')
            counter = 0
            for e in range(epochs):
                # Train network
                new_state = sess.run(model.initial_state)
                loss = 0
                for x, y in get_batches(encoded, batch_size, num_steps):
                    counter += 1
                    start = time.time()

                    feed = {
                        model.inputs: x,
                        model.targets: y,
                        model.keep_prob: keep_prob,
                        model.initial_state: new_state
                    }
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
                                                       model.initial_state: new_state
                                                   })
                        msg = '轮数: {}/{}... '.format(e + 1, epochs) + \
                              '训练步数: {}... '.format(counter) + \
                              '训练误差: {:.4f}... '.format(batch_loss) + \
                              '验证误差: {:.4f}...'.format(validation_loss) + \
                              '{:.4f} sec/batch'.format(end - start)
                        print(msg)

            saver.save(sess, "checkpointss/i{}_l{}.ckpt".format(counter, lstm_size))

                        # output_message(out_file_name, msg+'\n')




def pick_top_n(preds, vocab_size, top_n=5):
    """
    从预测结果中选取前top_n个最可能的字符

    preds: 预测结果
    vocab_size
    top_n
    """
    p = np.squeeze(preds)
    # 将除了top_n个预测值的位置都置为0
    p[np.argsort(p)[:-top_n]] = 0
    # 归一化概率
    p = p / np.sum(p)
    # 随机选取一个字符
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c




def sample(checkpoint, n_samples, lstm_size, num_layers, vocab_size, prime="The "):
    """
    生成新文本

    checkpoint: 某一轮迭代的参数文件
    n_sample: 新闻本的字符长度
    lstm_size: 隐层结点数
    vocab_size
    prime: 起始文本
    """
    # 将输入的单词转换为单个字符组成的list
    samples = [c for c in prime]
    # sampling=True意味着batch的size=1 x 1
    model = CharRNN(vocab_size, lstm_size=lstm_size, num_layers=num_layers, sampling=True, learning_rate=learning_rate)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 加载模型参数，恢复训练
        saver.restore(sess, checkpoint)
        new_state = sess.run(model.initial_state)
        preds = model.prediction
        x = np.zeros((1, 1))
        for c in prime:
            # 输入单个字符
            x[0, 0] = vocab_to_int[c]
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state],
                                        feed_dict=feed)

        c = pick_top_n(preds, vocab_size)
        # 添加字符到samples中
        samples.append(int_to_vocab[c])

        # 不断生成字符，直到达到指定数目
        for i in range(n_samples):
            x[0, 0] = c
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.prediction, model.final_state],
                                        feed_dict=feed)

            c = pick_top_n(preds, vocab_size)
            samples.append(int_to_vocab[c])

    return ''.join(samples)



tf.train.latest_checkpoint(checkpoint_dir)

# 选用最终的训练参数作为输入进行文本生成
checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
samp = sample(checkpoint, 1000, lstm_sizes[0], 2, len(vocab), prime="The")
print(samp)

# output layer = 2, lstm_size = 512

# There, and, with having been
# a completely sort of spiritual seconds of men. And as anything
# that she had been drunks and tears in a few sense. The coming, that had
# seen him the station with his wife and weaking of the people he was
# defined an electronit finishing to his brother-in--were never throbe,
# was now staring at the strain a stead a craim of children.
#
# "I am not a few telling your present to sprend that," said Alexey
# Alexandrovitch a particularly smile. "What is the same as I am
# at once the mutter, in his sing men, I'm going in, I am.
#
# "Anna don't were to do?"
#
# "Were you to be strengthening? I have too money, but that's a man of
# my children.... We've to be a solution of homoral signs of some time,
# but that will so little for their man, whom I wonker to her and the
# calling on the conversation with a subject as a lovior of a painful other
# it. I don't understand."
#
# "I shall be the tark to be done," he said, listening his wife at her
# wife.
#
# "You're so, but without them for you such infinite"
