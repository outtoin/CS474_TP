import tensorflow as tf
import pickle
import numpy as np

import os
import sys
from copy import deepcopy

CHECKPOINT = ""
TESTDIR = ""
KVAL = 0

for argv in sys.argv[1:]:    
    option = argv.split('=')[0]
    arg = argv.split('=')[1]
    if option == '--checkpoint':
        CHECKPOINT = arg
    elif option == '--test-dir':
        TESTDIR = arg
    elif option == '--kval':
        KVAL = int(arg)

if CHECKPOINT == "" or TESTDIR == "" or KVAL == 0:
    sys.exit(1)

tf.flags.DEFINE_string('checkpoints_dir', CHECKPOINT,
                       'Checkpoints directory (example: checkpoints/1479670630). Must contain (at least):\n'
                       '- config.pkl: Contains parameters used to train the model \n'
                       '- model.ckpt: Contains the weights of the model \n'
                       '- model.ckpt.meta: Contains the TensorFlow graph definition \n')
FLAGS = tf.flags.FLAGS

# Preprocessing Params
pos_val = 38
k_val = KVAL

# Data Preprocessing
with open(TESTDIR, 'r') as f:
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
seqlens = [x for i, x in enumerate(seqlens) if i not in outliers_idx]
raw_labels = [x for i, x in enumerate(raw_labels) if i not in outliers_idx]

label_datas = []
idx = 0
for i, tweet in enumerate(tweets):
    if i in outliers_idx:
        continue
    label_data = np.zeros((30, k_val+pos_val+1))
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


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labels = np.asarray(deepcopy(raw_labels))
enc = LabelEncoder()
labels = enc.fit_transform(labels).reshape(-1, 1)
ohe = OneHotEncoder(sparse=False)
labels = ohe.fit_transform(labels)

x_test = label_datas
y_test = labels
test_seq_len = seqlens

print("x_test: {}".format(x_test.shape))
print("y_test: {}".format(y_test.shape))
print("test_seq_len: {}".format(test_seq_len.shape))

writer = open('test/results.txt', 'a+t')
RESULT = ""
try:
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()

        # Import graph and restore its weights
        print('Restoring graph ...')
        saver = tf.train.import_meta_graph("{}/model.ckpt.meta".format(FLAGS.checkpoints_dir))
        saver.restore(sess, ("{}/model.ckpt".format(FLAGS.checkpoints_dir)))

        # Recover input/output tensors
        input = graph.get_operation_by_name('input').outputs[0]
        target = graph.get_operation_by_name('target').outputs[0]
        seq_len = graph.get_operation_by_name('lengths').outputs[0]
        dropout_keep_prob = graph.get_operation_by_name('dropout_keep_prob').outputs[0]
        predict = graph.get_operation_by_name('final_layer/softmax/predictions').outputs[0]
        accuracy = graph.get_operation_by_name('accuracy/accuracy').outputs[0]

        if input.shape[1] > x_test.shape[1]:
            x_test = np.concatenate((x_test, np.zeros((x_test.shape[0], 30-x_test.shape[1]))))
        elif input.shape[1] < x_test.shape[1]:
            x_test = x_test[:, :input.shape[1], :]

        # Perform prediction
        pred, acc = sess.run([predict, accuracy],
                             feed_dict={input: x_test,
                                        target: y_test,
                                        seq_len: test_seq_len,
                                        dropout_keep_prob: 1})

    # Print results
    print('\nAccuracy: {0:.4f}\n'.format(acc))
    RESULT = 'Accuracy: {0:.4f}'.format(acc)
except Exception as e:
    print(e)
    RESULT = "TRAIN MODEL DOESN'T EXIST(TRAIN FAILED)"
finally:
    writer.writelines("[{0}]: {1}\n".format(TESTDIR, RESULT))
    writer.close()
    sys.exit(1)
    








