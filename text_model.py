# -*- coding: utf-8 -*-
import tensorflow as tf

class TextConfig(object):

    embedding_dim = 100      #dimension of word embedding
    vocab_size =6000         #number of vocabulary
    pre_trianing=None        #use vector_char trained by word2vec

    seq_length=200          #max length of sentence
    num_classes=10          #number of labels

    num_layers= 1           #the number of layer
    hidden_dim = 128        #the number of hidden units
    attention_size = 100    #the size of attention layer


    keep_prob=0.5          #droppout
    learning_rate= 1e-3    #learning rate
    lr_decay= 0.9          #learning rate decay
    grad_clip= 5.0         #gradient clipping threshold

    num_epochs=10          #epochs
    batch_size= 64         #batch_size
    print_per_batch =100   #print result

    train_filename='./data/cnews.train.txt'  #train data
    test_filename='./data/cnews.test.txt'    #test data
    val_filename='./data/cnews.val.txt'      #validation data
    vocab_filename='./data/vocab.txt'        #vocabulary
    vector_word_filename='./data/vector_word.txt'  #vector_word trained by word2vec
    vector_word_npz='./data/vector_word.npz'   # save vector_word to numpy file


class TextRNN(object):
    def __init__(self, config):
        self.config = config
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self.rnn()

    def rnn(self):
        # Define Basic RNN Cell
        def basic_rnn_cell(rnn_size):
            # return tf.contrib.rnn.GRUCell(rnn_size)
            return tf.contrib.rnn.LSTMCell(rnn_size,state_is_tuple=True)

        # Define Forward RNN Cell
        with tf.name_scope('fw_rnn'):
            fw_rnn_cell = tf.contrib.rnn.MultiRNNCell([basic_rnn_cell(self.config.hidden_dim) for _ in range(self.config.num_layers)])
            fw_rnn_cell = tf.contrib.rnn.DropoutWrapper(fw_rnn_cell, output_keep_prob=self.keep_prob)

        # Define Backward RNN Cell
        with tf.name_scope('bw_rnn'):
            bw_rnn_cell = tf.contrib.rnn.MultiRNNCell([basic_rnn_cell(self.config.hidden_dim) for _ in range(self.config.num_layers)])
            bw_rnn_cell = tf.contrib.rnn.DropoutWrapper(bw_rnn_cell, output_keep_prob=self.keep_prob)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            # self.embedding = tf.Variable(tf.random_uniform([self.config.vocab_size, self.config.embedding_dim], -1.0, 1.0), trainable=False,name='W')
            self.embedding = tf.get_variable("embeddings", shape=[self.config.vocab_size, self.config.embedding_dim],
                                         initializer=tf.constant_initializer(self.config.pre_trianing))
            embedding_inputs = tf.nn.embedding_lookup(self.embedding, self.input_x)


        with tf.name_scope('bi_rnn'):
            # rnn_output, _ = tf.nn.dynamic_rnn(fw_rnn_cell, inputs=embedding_inputs, sequence_length=self.seq_len, dtype=tf.float32)
            rnn_output, _ = tf.nn.bidirectional_dynamic_rnn(fw_rnn_cell, bw_rnn_cell, inputs=embedding_inputs,
                                                            sequence_length=self.sequence_lengths, dtype=tf.float32)
        if isinstance(rnn_output, tuple):
            rnn_output = tf.concat(rnn_output, 2)

        # Attention Layer
        with tf.name_scope('attention'):
            input_shape = rnn_output.shape  # (batch_size, sequence_length, hidden_size)
            sequence_size = input_shape[1].value  # the length of sequences processed in the RNN layer
            hidden_size = input_shape[2].value  # hidden size of the RNN layer
            attention_w = tf.Variable(tf.truncated_normal([hidden_size, self.config.attention_size], stddev=0.1),
                                      name='attention_w')
            attention_b = tf.Variable(tf.constant(0.1, shape=[self.config.attention_size]), name='attention_b')
            attention_u = tf.Variable(tf.truncated_normal([self.config.attention_size], stddev=0.1), name='attention_u')
            z_list = []
            for t in range(sequence_size):
                u_t = tf.tanh(tf.matmul(rnn_output[:, t, :], attention_w) + tf.reshape(attention_b, [1, -1]))
                z_t = tf.matmul(u_t, tf.reshape(attention_u, [-1, 1]))
                z_list.append(z_t)
            # Transform to batch_size * sequence_size
            attention_z = tf.concat(z_list, axis=1)
            self.alpha = tf.nn.softmax(attention_z)
            attention_output = tf.reduce_sum(rnn_output * tf.reshape(self.alpha, [-1, sequence_size, 1]), 1)

        # Add dropout
        with tf.name_scope('dropout'):
            # attention_output shape: (batch_size, hidden_size)
            self.final_output = tf.nn.dropout(attention_output, self.keep_prob)

        # Fully connected layer
        with tf.name_scope('output'):
            fc_w = tf.Variable(tf.truncated_normal([hidden_size, self.config.num_classes], stddev=0.1), name='fc_w')
            fc_b = tf.Variable(tf.zeros([self.config.num_classes]), name='fc_b')
            self.logits = tf.matmul(self.final_output, fc_w) + fc_b
            self.y_pred_cls = tf.argmax(self.logits, 1, name='predictions')

        # Calculate cross-entropy loss
        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)

        # Create optimizer
        with tf.name_scope('optimization'):
            optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, self.config.grad_clip)
            self.optim = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)

        # Calculate accuracy
        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(self.y_pred_cls, tf.argmax(self.input_y, 1))
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))




