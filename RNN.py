import numpy as np
import tensorflow as tf

from copy import deepcopy
import pickle
import datetime
import time
import os
import sys

# command line argv
if len(sys.argv) is 1:
    print("need argv")
    sys.exit(1)

DATADIR = ""
KVAL = 0

for argv in sys.argv[1:]:    
    option = argv.split('=')[0]
    arg = argv.split('=')[1]
    if option == '--data-dir':
        DATADIR = arg
    elif option == '--kval':
        KVAL = int(arg)

is_B = False
if 'taskB' in DATADIR:
    is_B = True

# Preprocesing Params
pos_val = 38
k_val = KVAL
learning_rate = 0

if k_val == 50:
    learning_rate = 0.01
elif k_val == 200:
    learning_rate = 0.001
elif k_val == 1000:
    learning_rate = 0.00001

if learning_rate == 0:
    sys.exit(1)

# Saver params
FILENAME = DATADIR.split('/')[-1]

# Data Preprocessing
with open(DATADIR, 'r') as f:
    file = f.readlines()
    file = [e.split('\n')[:-1][0] for e in file]
    idxs = [i for i, x in enumerate(file) if x == '*']
    tweets = []
    for i in range(len(idxs) - 1):
        tweet = file[idxs[i]+1:idxs[i+1]]
        tweets.append(tweet)

seqlens = list(map(int, [tweet[0] for tweet in tweets]))
raw_labels = [tweet[1] for tweet in tweets]
outliers_idx = [i for i,x in enumerate(seqlens) if x > 30]
if (is_B):
    zero_idx = [i for i,x in enumerate(raw_labels) if x == '0']
    outliers_idx = list(set(outliers_idx+zero_idx))
seqlens = [x for i, x in enumerate(seqlens) if i not in outliers_idx]
raw_labels = [x for i, x in enumerate(raw_labels) if i not in outliers_idx]

label_datas = []
idx = 0
for i, tweet in enumerate(tweets):
    if i in outliers_idx:
        continue
    if is_B and i in zero_idx:
        continue
    label_data = np.zeros((max(seqlens), k_val+pos_val+1))
    for i, line in enumerate(tweet[2:]):
        n1 = np.zeros(k_val+1, dtype=np.float)
        for idx in line.split('/')[0].split(', '):
            n1[int(idx)] = 1
        n2 = np.zeros(pos_val, dtype=np.float)
        idx2 = int(line.split('/')[1:][0])
        n2[idx2] = 1
        n2[-1] = line.split('/')[1:][1]
        label_data[i] = np.append(n1, n2)
    label_datas.append(label_data)
label_datas = np.array(label_datas, dtype=np.float32)

seqlens = [sum(label_datas[i].any(axis=1)) for i in range(len(label_datas))]
assert sum(label_datas[i].any(axis=1)) == seqlens[i]
seqlens = np.array(seqlens)

print ("data length : {}".format(len(label_datas)))
print ("data shape : {}".format(str(label_datas.shape)))

# Label Encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labels = np.asarray(deepcopy(raw_labels))
enc = LabelEncoder()
labels = enc.fit_transform(labels).reshape(-1, 1)
ohe = OneHotEncoder(sparse=False)
labels = ohe.fit_transform(labels)


# Train, Test Split
split_frac = 0.9
split_index = int(split_frac * len(label_datas))

train_x, val_x = label_datas[:split_index], label_datas[split_index:]
train_y, val_y = labels[:split_index], labels[split_index:]
train_seq, val_seq = seqlens[:split_index], seqlens[split_index:]

'''
split_frac = 0.5
split_index = int(split_frac * len(val_x))

val_x, test_x = val_x[:split_index], val_x[split_index:]
val_y, test_y = val_y[:split_index], val_y[split_index:]
val_seq, test_seq = val_seq[:split_index], val_seq[split_index:]
'''

print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(train_x.shape),
      "\nValidation set: \t{}".format(val_x.shape))
      #"\nTest set: \t\t{}".format(test_x.shape))
print("Train label set: \t{}".format(train_y.shape),
      "\nValidation label set: \t{}".format(val_y.shape))
      #"\nTest label set: \t{}".format(test_y.shape)
print("Train seq set: \t\t{}".format(train_seq.shape),
      "\nValidation seq set: \t{}".format(val_seq.shape))
      #"\nTest seq set: \t\t{}".format(test_seq.shape)

# FLAGS Initialization
tf.flags.DEFINE_integer('n_samples', None,
                        'Number of samples to use from the dataset. Set n_samples=None to use the whole dataset')
tf.flags.DEFINE_integer('n_classes', train_y.shape[1],
                       'Number of output classes')
tf.flags.DEFINE_integer('dim', label_datas.shape[-1],
                        'Number of word dimension')
tf.flags.DEFINE_string('checkpoints_root', 'checkpoints',
                       'Checkpoints directory. Parameters will be saved there')
tf.flags.DEFINE_string('summaries_dir', 'logs',
                       'Directory where TensorFlow summaries will be stored')
tf.flags.DEFINE_integer('batch_size', 100,
                        'Batch size')
tf.flags.DEFINE_integer('train_steps', 500,
                        'Number of training steps')
tf.flags.DEFINE_integer('hidden_size', 75,
                        'Hidden size of LSTM layer')
tf.flags.DEFINE_integer('random_state', 0,
                        'Random state used for data splitting. Default is 0')
tf.flags.DEFINE_float('learning_rate', learning_rate,
                      'RMSProp learning rate')
tf.flags.DEFINE_float('dropout_keep_prob', 0.5,
                      '0<dropout_keep_prob<=1. Dropout keep-probability')
tf.flags.DEFINE_integer('sequence_len', None,
                        'Maximum sequence length. Let m be the maximum sequence length in the'
                        ' dataset. Then, it\'s required that sequence_len >= m. If sequence_len'
                        ' is None, then it\'ll be automatically assigned to m')
tf.flags.DEFINE_integer('validate_every', 100,
                        'Step frequency in order to evaluate the model using a validation set')
FLAGS = tf.flags.FLAGS

class NeuralNetwork(object):
    def __init__(self, hidden_size, max_length, dim, n_classes=3, learning_rate=0.01, beta=0.01,
                 random_state=None):
        """
        Builds a TensorFlow LSTM model
        :param hidden_size: Array holding the number of units in the LSTM cell of each rnn layer
        :param vocab_size: Vocabulary size (number of possible words that may appear in a sample)
        :param embedding_size: Words will be encoded using a vector of this size
        :param max_length: Maximum length of an input tensor
        :param n_classes: Number of classification classes
        :param learning_rate: Learning rate of RMSProp algorithm
        :param random_state: Random state for dropout
        """
        # for L2-reg
        self.beta = beta

        # Build TensorFlow graph
        self.input = self.__input(max_length, dim)
        self.seq_len = self.__seq_len()
        self.target = self.__target(n_classes)
        self.dropout_keep_prob = self.__dropout_keep_prob()
        #self.word_embeddings = self.__word_embeddings(self.input, vocab_size, embedding_size, random_state)
        self.scores, self.w = self.__scores(self.input, self.seq_len, hidden_size, n_classes, self.dropout_keep_prob,
                                    random_state)
        self.predict = self.__predict(self.scores)
        self.losses = self.__losses(self.scores, self.target)
        self.loss = self.__loss(self.losses, self.w)
        self.train_step = self.__train_step(learning_rate, self.loss)
        self.accuracy = self.__accuracy(self.predict, self.target)
        self.merged = tf.summary.merge_all()

    def __input(self, max_length, dim):
        """
        :param max_length: Maximum length of an input tensor
        :return: Input placeholder with shape [batch_size, max_length]
        """
        return tf.placeholder(tf.float32, [None, max_length, dim], name='input')

    def __seq_len(self):
        """
        :return: Sequence length placeholder with shape [batch_size]. Holds each tensor's real length in a given batch,
                 allowing a dynamic sequence length.
        """
        return tf.placeholder(tf.int32, [None], name='lengths')

    def __target(self, n_classes):
        """
        :param n_classes: Number of classification classes
        :return: Target placeholder with shape [batch_size, n_classes]
        """
        return tf.placeholder(tf.float32, [None, n_classes], name='target')

    def __dropout_keep_prob(self):
        """
        :return: Placeholder holding the dropout keep probability
        """
        return tf.placeholder(tf.float32, name='dropout_keep_prob')

    def __cell(self, hidden_size, dropout_keep_prob, seed=None):
        """
        Builds a LSTM cell with a dropout wrapper
        :param hidden_size: Number of units in the LSTM cell
        :param dropout_keep_prob: Tensor holding the dropout keep probability
        :param seed: Optional. Random state for the dropout wrapper
        :return: LSTM cell with a dropout wrapper
        """
        lstm_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(hidden_size)
        #lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_size, state_is_tuple=True)
        dropout_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=dropout_keep_prob,
                                                     output_keep_prob=dropout_keep_prob, seed=seed)
        return dropout_cell

    def __word_embeddings(self, x, vocab_size, embedding_size, seed=None):
        """
        Builds the embedding layer with shape [vocab_size, embedding_size]
        :param x: Input with shape [batch_size, max_length]
        :param vocab_size: Vocabulary size (number of possible words that may appear in a sample)
        :param embedding_size: Words will be represented using a vector of this size
        :param seed: Optional. Random state for the embeddings initiallization
        :return: Embedding lookup tensor with shape [batch_size, max_length, embedding_size]
        """
        with tf.name_scope('word_embeddings'):
            embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1, 1, seed=seed))
            embedded_words = tf.nn.embedding_lookup(embeddings, x)
        return embedded_words

    def __rnn_layer(self, hidden_size, x, seq_len, dropout_keep_prob, variable_scope=None, random_state=None):
        """
        Builds a LSTM layer
        :param hidden_size: Number of units in the LSTM cell
        :param x: Input with shape [batch_size, max_length]
        :param seq_len: Sequence length tensor with shape [batch_size]
        :param dropout_keep_prob: Tensor holding the dropout keep probability
        :param variable_scope: Optional. Name of variable scope. Default is 'rnn_layer'
        :param random_state: Optional. Random state for the dropout wrapper
        :return: outputs with shape [batch_size, max_seq_len, hidden_size]
        """
        with tf.variable_scope(variable_scope, default_name='rnn_layer'):
            # Build LSTM cell
            lstm_cell = self.__cell(hidden_size, dropout_keep_prob, random_state)

            # Dynamically unroll LSTM cells according to seq_len. From TensorFlow documentation:
            # "The parameter `sequence_length` is used to copy-through state and zero-out outputs when past a batch
            # element's sequence length."
            outputs, _ = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32, sequence_length=seq_len)
        return outputs

    def __scores(self, x, seq_len, hidden_size, n_classes, dropout_keep_prob, random_state=None):
        """
        Builds the LSTM layers and the final fully connected layer
        :param embedded_words: Embedding lookup tensor with shape [batch_size, max_length, embedding_size]
        :param seq_len: Sequence length tensor with shape [batch_size]
        :param hidden_size: Array holding the number of units in the LSTM cell of each rnn layer
        :param n_classes: Number of classification classes
        :param dropout_keep_prob: Tensor holding the dropout keep probability
        :param random_state: Optional. Random state for the dropout wrapper
        :return: Linear activation of each class with shape [batch_size, n_classes]
        """
        # Build LSTM layers
        outputs = x
        for h in hidden_size:
            outputs = self.__rnn_layer(h, outputs, seq_len, dropout_keep_prob)

        # Current shape of outputs: [batch_size, max_seq_len, hidden_size]. Reduce mean on index 1
        outputs = tf.reduce_mean(outputs, reduction_indices=[1])

        # Current shape of outputs: [batch_size, hidden_size]. Build fully connected layer
        with tf.name_scope('final_layer/weights'):
            w = tf.Variable(tf.truncated_normal([hidden_size[-1], n_classes], seed=random_state))
            self.variable_summaries(w, 'final_layer/weights')
        with tf.name_scope('final_layer/biases'):
            b = tf.Variable(tf.constant(0.1, shape=[n_classes]))
            self.variable_summaries(b, 'final_layer/biases')
        with tf.name_scope('final_layer/matmul'):
            scores = tf.matmul(outputs, w) + b
            tf.summary.histogram('final_layer/wx_plus_b', scores)
        return (scores, w)

    def __predict(self, scores):
        """
        :param scores: Linear activation of each class with shape [batch_size, n_classes]
        :return: Softmax activations with shape [batch_size, n_classes]
        """
        with tf.name_scope('final_layer/softmax'):
            softmax = tf.nn.softmax(scores, name='predictions')
            tf.summary.histogram('final_layer/softmax', softmax)
        return softmax

    def __losses(self, scores, target):
        """
        :param scores: Linear activation of each class with shape [batch_size, n_classes]
        :param target: Target tensor with shape [batch_size, n_classes]
        :return: Cross entropy losses with shape [batch_size]
        """
        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=target)
        return cross_entropy

    def __loss(self, losses, w):
        """
        :param losses: Cross entropy losses with shape [batch_size]
        :return: Cross entropy loss mean
        """
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(losses, name='loss')
            tf.summary.scalar('loss', loss)
        with tf.name_scope('regularizer'):
            regularizer = tf.nn.l2_loss(w)
            loss = tf.reduce_mean(loss + self.beta * regularizer)
        return loss

    def __train_step(self, learning_rate, loss):
        """
        :param learning_rate: Learning rate of RMSProp algorithm
        :param loss: Cross entropy loss mean
        :return: RMSProp train step operation
        """
        return tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

    def __accuracy(self, predict, target):
        """
        :param predict: Softmax activations with shape [batch_size, n_classes]
        :param target: Target tensor with shape [batch_size, n_classes]
        :return: Accuracy mean obtained in current batch
        """
        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(tf.argmax(predict, 1), tf.argmax(target, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
            tf.summary.scalar('accuracy', accuracy)
        return accuracy

    def initialize_all_variables(self):
        """
        :return: Operation that initiallizes all variables
        """
        return tf.global_variables_initializer()

    @staticmethod
    def variable_summaries(var, name):
        """
        Attach a lot of summaries to a Tensor for Tensorboard visualization.
        Ref: https://www.tensorflow.org/versions/r0.11/how_tos/summaries_and_tensorboard/index.html
        :param var: Variable to summarize
        :param name: Summary name
        """
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean/' + name, mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev/' + name, stddev)
            tf.summary.scalar('max/' + name, tf.reduce_max(var))
            tf.summary.scalar('min/' + name, tf.reduce_min(var))
            tf.summary.histogram(name, var)

# Prepare summaries
summaries_dir = '{0}/{1}-{2}'.format(FLAGS.summaries_dir,
                                 datetime.datetime.now().strftime('%d_%b_%Y-%H_%M_%S'),
                                 FILENAME)
train_writer = tf.summary.FileWriter(summaries_dir + '/train')
validation_writer = tf.summary.FileWriter(summaries_dir + '/validation')

# Prepare model directory
model_name = str(int(time.time()))
model_dir = '{0}/{1}-{2}'.format(FLAGS.checkpoints_root, model_name, FILENAME)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    
# Save configuration
FLAGS._parse_flags()
config = FLAGS.__dict__['__flags']
with open('{}/config.pkl'.format(model_dir), 'wb') as f:
    pickle.dump(config, f)


nn = NeuralNetwork(hidden_size=[FLAGS.hidden_size],
                   max_length=train_x.shape[1],
                   dim=FLAGS.dim,
                   n_classes=FLAGS.n_classes,
                   learning_rate=FLAGS.learning_rate)


def get_batches(x, y, seq, batch_size):
    n_batches = len(x)//batch_size
    x, y, seq = x[:n_batches*batch_size], y[:n_batches*batch_size], seq[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size], seq[ii:ii+batch_size]

# Train model
sess = tf.Session()
sess.run(nn.initialize_all_variables())
saver = tf.train.Saver()
x_val, y_val, val_seq_len = val_x, val_y, val_seq
train_writer.add_graph(nn.input.graph)

for i in range(FLAGS.train_steps):
    # Perform training step
    x_train, y_train, train_seq_len = next(get_batches(train_x, train_y, train_seq, FLAGS.batch_size))
    train_loss, _, summary = sess.run([nn.loss, nn.train_step, nn.merged],
                                      feed_dict={nn.input: x_train,
                                                 nn.target: y_train,
                                                 nn.seq_len: train_seq_len,
                                                 nn.dropout_keep_prob: FLAGS.dropout_keep_prob})
    train_writer.add_summary(summary, i)  # Write train summary for step i (TensorBoard visualization)
    # print('{0}/{1} train loss: {2:.4f}'.format(i + 1, FLAGS.train_steps, train_loss))

    # Check validation performance
    if (i + 1) % FLAGS.validate_every == 0:
        val_loss, accuracy, summary = sess.run([nn.loss, nn.accuracy, nn.merged],
                                               feed_dict={nn.input: x_val,
                                                          nn.target: y_val,
                                                          nn.seq_len: val_seq_len,
                                                          nn.dropout_keep_prob: 1})
        validation_writer.add_summary(summary, i)  # Write validation summary for step i (TensorBoard visualization)
        print('[validation loss] {0:.4f} (accuracy {1:.4f})'.format(val_loss, accuracy))

# Save the Model
save_path = saver.save(sess, "{}/model.ckpt".format(model_dir))
print("Model saved in file: %s" % save_path)


