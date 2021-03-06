import tensorflow as tf
import time
import numpy as np
from tensorflow.python.layers.core import Dense

with open('input_data.txt') as f:
    source_data = f.read()
with open('target_data.txt') as f:
    target_data = f.read()

# print(target_data.split('\n')[:10])


# 输入数据处理
def extract_character_vocab(data):
    '''
    < PAD>: 补全字符。
    < EOS>: 解码器端的句子结束标识符。
    < UNK>: 未遇到过的词。
    < GO>: 解码器端的句子起始标识符。
    '''
    # 构建映射表
    special_words = ['<PAD>', '<UNK>', '<GO>', '<EOS>']
    set_words = list(set([char for line in data.split('\n') for char in line]))
    # 把四个特殊字符添加进词典
    int_to_vocab = {idx: word for idx, word in enumerate(special_words + set_words)}
    vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}
    return int_to_vocab, vocab_to_int

# source_data 的词汇转换
si2l, sl2i = extract_character_vocab(source_data)

# target_data 的词汇转换
ti2l, tl2i = extract_character_vocab(target_data)

#
source_int = [[sl2i.get(letter, sl2i['<UNK>'])
               for letter in line] for line in source_data.split('\n')]

target_int = [[tl2i.get(letter, tl2i['<UNK>'])
               for letter in line] + [tl2i['<EOS>']] for line in target_data.split('\n')]


# 输入层
def get_inputs():
    '''
    模型输入tensor
    '''

    inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    # 定义target序列最大长度(target_sequence_length和source_sequence_length会作为feed_dict参数)
    target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')
    max_target_sequence_length = tf.reduce_max(target_sequence_length, name='target_sequence_length')
    source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')

    return inputs, targets, learning_rate, target_sequence_length, max_target_sequence_length, source_sequence_length

# 编码层 输出最后一层状态
def get_encoder_layer(input_data, rnn_size, num_layers,
                      source_sequence_length, source_vocab_size,
                      encoding_embedding_size):
    '''
    构造Encoder层
    input_data: 输入tensor
    rnn_size: rnn隐层节点数量
    num_layers: 堆叠rnn cell的数量
    source_sequence_length: 源数据序列长度
    source_vocab_size: 源数据词典大小
    encoding_embedding_size: enbedding的大小
    '''
    # 词向量嵌入

    encoder_embed_input = tf.contrib.layers.embed_sequence(
        input_data, source_vocab_size, encoding_embedding_size)

    # RNN核

    def get_lstm_cell(rnn_size):
        lstm_cell = tf.contrib.rnn.LSTMCell(
            rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        drop = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
        return drop
    # 堆叠

    multiple_cell = tf.contrib.rnn.MultiRNNCell([get_lstm_cell(rnn_size) for _ in range(num_layers)])
    encoder_output, encoder_state = tf.nn.dynamic_rnn(
        multiple_cell, encoder_embed_input, sequence_length=source_sequence_length,
        dtype=tf.float32)
    return encoder_output, encoder_state

# 预处理target 数据
def process_decoder_input(data, vocab_to_int, batch_size):
    '''
    补充<GO>,移除最后一个字符
    '''

    # 移除最后一个字符

    ending = tf.strided_slice(data, [0, 0], [batch_size, -1], [1, 1])
    decoder_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)
    return decoder_input

# 译码层
def decoding_layer(tl2i, decoding_embedding_size, num_layers, rnn_size,
                   target_sequence_length, max_target_sequence_length,
                   encoder_state, decoder_input):
    '''
    构造Decoder层
    tl2i: target数据的映射表
    decoding_embedding_size: embed向量大小
    num_layers: 堆叠的RNN单元数量
    rnn_size: lstm隐层节点数
    target_sequence_length: target数据序列长度
    max_target_sequence_length: target数据序列最大长度
    encoder_state: encoder端编码的状态向量
    decoder_input: decoder端输入
    '''
    # Embedding
    target_vocab_size = len(tl2i)
    decoder_embeddings = tf.Variable(tf.random_uniform(
        [target_vocab_size, decoding_embedding_size]))
    decoder_embed_input = tf.nn.embedding_lookup(
        decoder_embeddings, decoder_input)

    # 构造Decoder中的RNN单元
    def get_decoder_cell(rnn_size):
        decoder_cell = tf.contrib.rnn.LSTMCell(
            rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        drop = tf.contrib.rnn.DropoutWrapper(decoder_cell, output_keep_prob=keep_prob)
        return drop
    cell = tf.contrib.rnn.MultiRNNCell(
        [get_decoder_cell(rnn_size) for _ in range(num_layers)])


    # Output全连接层
    output_layer = Dense(target_vocab_size,
                         kernel_initializer=
                         tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

    # Training decoder
    with tf.variable_scope("decode"):
        # 得到help对象
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input,
                                                            sequence_length=target_sequence_length,
                                                            time_major=False)

        # 构造decoder
        training_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
                                                           training_helper,
                                                           encoder_state,
                                                           output_layer)

        training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
            training_decoder, impute_finished=True,
            maximum_iterations=max_target_sequence_length)

    # Predicting decoder
    # 与training共享参数
    with tf.variable_scope("decode", reuse=True):

        # 创建一个常量tensor并复制为batch_size的大小
        start_tokens = tf.tile(
            tf.constant([tl2i['<GO>']], dtype=tf.int32),
            [batch_size], name='start_tokens'
        )

        predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embeddings,
                                                                     start_tokens,
                                                                     tl2i['<EOS>'])

        predicting_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
                                                             predicting_helper,
                                                             encoder_state,
                                                             output_layer)

        predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
            predicting_decoder, impute_finished=True,
            maximum_iterations=max_target_sequence_length)

    return training_decoder_output, predicting_decoder_output

def seq2seq_model(input_data, targets, lr, target_sequence_length,
                  max_target_sequence_length, source_sequence_length,
                  source_vocab_size, target_vocab_size,
                  encoding_embedding_size, decoding_embedding_size,
                  rnn_size, num_layers):
    # 获取encoder的状态输出
    _, encoder_state = get_encoder_layer(input_data,
                                         rnn_size,
                                         num_layers,
                                         source_sequence_length,
                                         source_vocab_size,
                                         encoding_embedding_size)

    # 预处理后的decoder输入
    decoder_input = process_decoder_input(targets, tl2i, batch_size)

    #将状态向量与输入传递给decoder
    training_decoder_output, predicting_decoder_output = decoding_layer(tl2i,
                                                                       decoding_embedding_size,
                                                                       num_layers,
                                                                       rnn_size,
                                                                       target_sequence_length,
                                                                       max_target_sequence_length,
                                                                       encoder_state,
                                                                       decoder_input)
    return training_decoder_output, predicting_decoder_output

# 超参数
# Number of Epochs
epochs = 30

# Batch Size
batch_size = 128

# Embedding Size
encoding_embedding_size = 15
decoding_embedding_size = 15

# Learning Rate
learning_rate = 0.001

# keep probility
keep_prob = 0.8

# max_layer
# max_layer = 12
max_layer = 2
# lstm_size_list = [32, 64, 96, 128, 160, 192]
def pad_sentence_batch(sentence_batch, pad_int):
    '''
    对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length
    :param sentence_batch:
    :param pad_int: <PAD>对应的索引号
    :return:
    '''

    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]


def get_batches(targets, sources, batch_size, source_pad_int, target_pad_int):
    '''
    定义生成器，用来获取batch
    :param targets:
    :param sources:
    :param batch_size:
    :param source_pad_int:
    :param target_pad_int:
    :return:
    '''

    for batch_i in range(0, len(sources) // batch_size):
        start_i = batch_i * batch_size
        sources_batch = sources[start_i: start_i + batch_size]
        targets_batch = targets[start_i: start_i + batch_size]
        # 补全序列
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))

        # 记录每条记录的长度
        targets_lengths = []
        for target in targets_batch:
            targets_lengths.append(len(target))

        source_lengths = []
        for source in sources_batch:
            source_lengths.append(len(source))

        yield pad_targets_batch, pad_sources_batch, targets_lengths, source_lengths
lstm_size_list = [128]
for rnn_size in lstm_size_list:
    # out_file_name = "out2_lstm_size_" + str(rnn_size) + ".txt"
    # f = open(out_file_name, "w+")
    # message = 'batch_size = '+batch_size.__str__()+'\nlstm_size = '+rnn_size.__str__()+'\nlearning_rate = '+learning_rate.__str__()+'\nkeep_prob'+keep_prob.__str__()
    # f.write(message)
    # f.close()

    for num_layers in range(2, max_layer + 1):

        # f = open(out_file_name, "a+");
        # f.write("\n当前隐藏层layer数量： %d\n" % num_layers)
        # f.close();

        # 构造graph
        train_graph = tf.Graph()

        with train_graph.as_default():
            # 获得模型输入
            input_data, targets, lr, target_sequence_length, max_target_sequence_length, source_sequence_length = get_inputs()
            training_decoder_output, predicting_decoder_output = seq2seq_model(
                input_data,
                targets,
                lr,
                target_sequence_length,
                max_target_sequence_length,
                source_sequence_length,
                len(sl2i),
                len(tl2i),
                encoding_embedding_size,
                decoding_embedding_size,
                rnn_size,
                num_layers
            )
            training_logits = tf.identity(training_decoder_output.rnn_output, 'logits')
            predicting_logits = tf.identity(predicting_decoder_output.sample_id, name='predictions')

            masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')

            with tf.name_scope('optimiation'):
                # 损失函数
                cost = tf.contrib.seq2seq.sequence_loss(
                    training_logits,
                    targets,
                    masks
                )

                # 优化器
                optimizer = tf.train.AdamOptimizer(lr)

                # 梯度消减
                gradients = optimizer.compute_gradients(cost)
                capped_gradients = [(tf.clip_by_value(grad, -5.0, 5.0), var) for grad, var in gradients if grad is not None]
                train_op = optimizer.apply_gradients(capped_gradients)



        # Train

        # 将数据集分割为train和validation
        train_source = source_int[batch_size:]
        train_target = target_int[batch_size:]
        # 留出一个batch用作validation
        valid_source = source_int[:batch_size]
        valid_target = target_int[:batch_size]
        (valid_targets_batch, valid_sources_batch, valid_targets_lengths, valid_sources_lengths) = \
            next(get_batches(valid_target, valid_source, batch_size, sl2i['<PAD>'], tl2i['<PAD>']))

        display_step = 50 # 每50轮输出loss

        checkpoint = 'trained_model_0.ckpt'


        # with tf.Session(graph=train_graph) as sess:
        #     sess.run(tf.global_variables_initializer())
        #     saver = tf.train.Saver()
        #     for epoch_i in range(1, epochs + 1):
        #         for batch_i, (targets_batch, sources_batch, targets_lengths, sources_lengths) in enumerate(
        #             get_batches(train_target, train_source, batch_size, sl2i['<PAD>'], tl2i['<PAD>'])):
        #
        #             start_time = time.time()
        #             _, loss = sess.run([train_op, cost],
        #                                feed_dict={
        #                                    input_data: sources_batch,
        #                                    targets:targets_batch,
        #                                    lr: learning_rate,
        #                                    target_sequence_length: targets_lengths,
        #                                    source_sequence_length: sources_lengths
        #                                })
        #             end_time = time.time()
        #             if batch_i % display_step == 0:
        #                 # 计算validation loss
        #                 validation_loss = sess.run(
        #                     [cost], feed_dict={
        #                         input_data: valid_sources_batch,
        #                         targets: valid_targets_batch,
        #                         lr: learning_rate,
        #                         target_sequence_length: valid_targets_lengths,
        #                         source_sequence_length: valid_sources_lengths
        #                     })
        #                 msg = 'Epoch {:>3}/{} Batch {:>4}/{} - Training Loss: {:>6.3f}  ' \
        #                       '- Validation loss: {:>6.3f} {:.4f} sec/batch'\
        #                     .format(epoch_i,
        #                               epochs,
        #                               batch_i,
        #                               len(train_source) // batch_size,
        #                               loss,
        #                               validation_loss[0], end_time - start_time)
        #                 print(msg)
        #                 # f = open(out_file_name, "a+");
        #                 # f.write(msg + '\n')
        #                 # f.close();
        #
        #     saver.save(sess, 'checkpoints/' + checkpoint)
            # print('Model Trained and Saved')

def source_to_seq(text):
    '''
    对源数据进行转换
    '''
    sequence_length = 7
    return [sl2i.get(word, sl2i['<UNK>']) for word in text] + [sl2i['<PAD>']]*(sequence_length-len(text))

# 输入一个单词
input_word = 'networks'
text = source_to_seq(input_word)

checkpoint = "checkpoints/trained_model_0.ckpt"

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    sess.run(tf.global_variables_initializer())
    # 加载模型
    loader = tf.train.import_meta_graph(checkpoint + '.meta')
    loader.restore(sess, checkpoint)

    input_data = loaded_graph.get_tensor_by_name('inputs:0')
    logits = loaded_graph.get_tensor_by_name('predictions:0')
    source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_length:0')
    target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')

    answer_logits = sess.run(logits, {input_data: [text]*batch_size,
                                      target_sequence_length: [len(input_word)]*batch_size,
                                      source_sequence_length: [len(input_word)]*batch_size})[0]


pad = sl2i["<PAD>"]

print('original input:', input_word)

print('\nSource')
print('  Word 编号:    {}'.format([i for i in text]))
print('  Input Words: {}'.format(" ".join([si2l[i] for i in text])))

print('\nTarget')
print('  Word 编号:       {}'.format([i for i in answer_logits if i != pad]))
print('  Response Words: {}'.format(" ".join([ti2l[i] for i in answer_logits if i != pad])))

# original input: networks
#
# Source
#   Word 编号:    [18, 5, 14, 22, 9, 27, 13, 4]
#   Input Words: n e t w o r k s
#
# Target
#   Word 编号:       [5, 13, 18, 9, 27, 4, 14, 22]
#   Response Words: e k n o r s t w
