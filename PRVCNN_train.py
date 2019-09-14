import tensorflow as tf
import numpy as np
import PRVCNN_inference
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import matthews_corrcoef
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
LEARNING_RATE_BASE = 0.0001
LEARNING_RATE_DECAY = 0.99
BATCH_SIZE = 16
REGULARIZATION_RATE = 0.001
TRAINING_STEPS = 100000
MOVING_AVERAGE_DECAY = 0.99
NUM_EXAMPLE = 60977
num_epochs = 20


def train(args):
    LEARNING_RATE_BASE = args.learning_rate
    MOVING_AVERAGE_DECAY = args.learning_rate_decay
    drop_rate = args.dropout_rate
    x = tf.placeholder(tf.float32, [None, 74], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 2], name='y-input')
    y, y_softmax = PRVCNN_inference.inference4(x, args)
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    train_op = tf.train.AdamOptimizer(LEARNING_RATE_BASE).minimize(loss, global_step=global_step)
    # train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        saver = tf.train.Saver()

        # load data
        print("Loading data")
        f_dir = args.training_feature
        l_dir = args.training_label
        training_feature = np.loadtxt('%s' % (f_dir), delimiter=',')
        training_label = np.loadtxt('%s' % (l_dir), delimiter=',')

        training_feature = np.reshape(training_feature, [-1, 74])
        training_label = np.reshape(training_label, [-1, 2])

        print("train start")
        total_batch = int(60977 / BATCH_SIZE)
        for epoch in range(num_epochs):
            x_tmp, y_tmp = shuffle(training_feature, training_label)
            for i in range(total_batch - 1):
                xs = x_tmp[i * BATCH_SIZE:i * BATCH_SIZE + BATCH_SIZE]
                ys = y_tmp[i * BATCH_SIZE:i * BATCH_SIZE + BATCH_SIZE]
                xs = np.reshape(xs, [-1, 74])
                ys = np.reshape(ys, [-1, 2])
                _, y_pred, y_true, losses = sess.run([train_op, y, y_, loss], feed_dict={x: xs, y_: ys})
        model_name = args.model_name
        saver.save(sess, './log/%s' % (model_name))
        print("Model saved in path: ./log/%s" % (model_name))


def main(argv=None):
    train()


if __name__ == '__main__':
    main()
