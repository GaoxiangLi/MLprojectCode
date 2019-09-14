import tensorflow as tf
import numpy as np
import PRVCNN_inference
import matplotlib.pyplot as plt
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


def test4(args):
    x = tf.placeholder(tf.float32, [None, 74], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 2], name='y-input')
    y, y_softmax = PRVCNN_inference.inference4(x, args)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        saver = tf.train.Saver()
        test_name = args.test_name
        saver.restore(sess, "./log/%s" % (test_name))

        test_feature = np.loadtxt('./data/t4_validated_feature.csv', delimiter=',')
        test_label = np.loadtxt('./data/t4_validated_label.csv', delimiter=',')
        test_feature = np.reshape(test_feature, [-1, 74])
        test_label = np.reshape(test_label, [-1, 2])

        # Testing and validation
        print("Testing4 start")
        y_pred = sess.run(y_softmax, feed_dict={x: test_feature})

        # Calculate metrics
        y_pred = np.reshape(y_pred, [-1, 2])
        y_prob = y_pred[:, 1:2]
        np.savetxt("./result/score4.csv", y_prob, delimiter=",")
        print("training model prediction score for test4 saved in ./result/score4.cs")
        y_true = np.argmax(test_label, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
        acc = accuracy_score(y_true, y_pred)
        precision_2 = precision_score(y_true, y_pred, average='weighted')
        recall_2 = recall_score(y_true, y_pred, average='weighted')
        f1 = 2 * (precision_2 * recall_2) / (precision_2 + recall_2)
        matthews_correlation_coefficient1 = matthews_corrcoef(y_true, y_pred)
        print("After test, acc in validation set is", acc)
        print("Recall in validation set is", recall_2)
        print("Precision in validation set is", precision_2)
        print("F1_score in validation set is", f1)
        print(("Matthews correlation is", matthews_correlation_coefficient1))




def main(argv=None):
    # train()
    test4()


if __name__ == '__main__':
    main()
