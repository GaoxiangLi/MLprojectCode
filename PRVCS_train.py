import tensorflow as tf
import numpy as np
import PRVCS_inference
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.99
BATCH_SIZE = 50
REGULARIZATION_RATE = 0.01
TRAINING_STEPS = 200000
MOVING_AVERAGE_DECAY = 0.99
NUM_EXAMPLE = 60977


def train():
    # regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    x = tf.placeholder(tf.float32, [None, 81], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 2], name='y-input')
    drop_rate = tf.placeholder(tf.float32, name='drop_rate')
    y = PRVCS_inference.inference2(x, drop_rate)

    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(

        LEARNING_RATE_BASE,
        global_step,
        NUM_EXAMPLE / BATCH_SIZE, LEARNING_RATE_DECAY,
        staircase=True)

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # load data
        print("Loading data")
        training_feature = np.loadtxt('./training_feature2.csv', delimiter=',')
        training_label = np.loadtxt('./training_label2.csv', delimiter=',')
        test_feature = np.loadtxt('./test_feature2.csv', delimiter=',')
        test_label = np.loadtxt('./test_label2.csv', delimiter=',')

        training_feature = np.reshape(training_feature, [-1, 81])
        test_feature = np.reshape(test_feature, [-1, 81])
        training_label = np.reshape(training_label, [-1, 2])
        test_label = np.reshape(test_label, [-1, 2])

        TRAINING_EXAMPLE = len(training_feature)
        print("train start")
        for i in range(TRAINING_STEPS):
            xs = training_feature[i % TRAINING_EXAMPLE]
            ys = training_label[i % TRAINING_EXAMPLE]
            # Batch
            for j in range(BATCH_SIZE - 1):
                xs = np.append(xs, training_feature[(i + j + 1) % TRAINING_EXAMPLE])
                ys = np.append(ys, training_label[(i + j + 1) % TRAINING_EXAMPLE])
            xs = np.reshape(xs, [-1, 81])
            ys = np.reshape(ys, [-1, 2])
            _, y_pred, y_true, loss_value = sess.run([train_op, y, y_, loss],
                                                     feed_dict={x: xs, y_: ys, drop_rate: 1})

            if i % 5000 == 0:
                print(
                    "After %d training step(s), cross_entropy mean on training batch is %g." % (i, loss_value))
            # # testing
            # if i % 10000 == 0 and i > 50000:
            #     TEST_EXAMPLE = len(test_feature)
            #     for i in range(TEST_EXAMPLE):
            #         xs = test_feature[i]
            #         ys = test_label[i]
            #         xs = np.reshape(xs, [-1, 81])
            #         ys = np.reshape(ys, [-1, 2])
            #         y_pred, y_true, loss = sess.run([y, y_, cross_entropy_mean],
            #                                         feed_dict={x: xs, y_: ys, drop_rate: 1})
            #
            #     print("After test, cross_entropy is %g" % loss)

        # Validation
        print("train finished, validation start")
        # TEST_EXAMPLE = len(test_feature)
        # for i in range(TEST_EXAMPLE):
        #     xs = test_feature[i]
        #     ys = test_label[i]
        #     xs = np.reshape(xs, [-1, 81])
        #     ys = np.reshape(ys, [-1, 2])
        _, y_pred, y_true = sess.run([train_op, y, y_],
                                     feed_dict={x: test_feature, y_: test_label, drop_rate: 1})
        # acc = tf.metrics.accuracy(y_true, y_pred)
        # recall = tf.metrics.recall(y_true, y_pred)
        # precision = tf.metrics.precision(y_true, y_pred)
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        print("After test, acc in validation set is", acc)
        print("Recall in validation set is", recall)
        print("Precision in validation set is", precision)


def main(argv=None):
    train()


if __name__ == '__main__':
    main()
