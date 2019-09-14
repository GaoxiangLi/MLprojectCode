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
from sklearn.utils import shuffle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
LEARNING_RATE_BASE = 0.0001
LEARNING_RATE_DECAY = 0.99
BATCH_SIZE = 16
REGULARIZATION_RATE = 0.001
TRAINING_STEPS = 100000
MOVING_AVERAGE_DECAY = 0.99
NUM_EXAMPLE = 60977
num_epochs = 25


def test1(args):
    x = tf.placeholder(tf.float32, [None, 74], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 2], name='y-input')
    # drop_rate = tf.placeholder(tf.float32, name='drop_rate')
    y, y_softmax = PRVCNN_inference.inference4(x, args)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        saver = tf.train.Saver()
        test_name = args.test_name
        saver.restore(sess, "./log/%s" % (test_name))

        test_feature = np.loadtxt('./data/t1_imbanlance_feature.csv', delimiter=',')
        test_label = np.loadtxt('./data/t1_imbanlance_label.csv', delimiter=',')
        test_feature = np.reshape(test_feature, [-1, 74])
        test_label = np.reshape(test_label, [-1, 2])

        print("Testing1 start")
        y_pred = sess.run(y_softmax, feed_dict={x: test_feature})

        # Calculate metrics
        y_pred = np.reshape(y_pred, [-1, 2])
        y_prob = y_pred[:, 1:2]
        np.savetxt("./result/score1.csv", y_prob, delimiter=",")
        print("training model prediction score for test1 saved in ./result/score1.csv")
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

        # Compute fpr, tpr, thresholds and roc auc
        # fpr, tpr, thresholds = roc_curve(y_true, y_prob, pos_label=1)
        #
        # roc_auc = auc(fpr, tpr)
        # all_label = np.loadtxt("./data/t1_label.csv", delimiter=',')
        # y_pred2 = np.loadtxt('./data/t1_Fathmm_MKL.csv', delimiter=',')
        # y_pred3 = np.loadtxt('./data/t1_GWAS3D.csv', delimiter=',')
        # y_pred4 = np.loadtxt('./data/t1_Funseq.csv', delimiter=',')
        # y_pred5 = np.loadtxt('./data/t1_Funseq2.csv', delimiter=',')
        # y_pred6 = np.loadtxt('./data/t1_GWAVA_TSS.csv', delimiter=',')
        # y_pred7 = np.loadtxt('./data/t1_SuRFR.csv', delimiter=',')
        # y_pred8 = np.loadtxt('./data/t1_DANN.csv', delimiter=',')
        # y_pred9 = np.loadtxt('./data/t1_CADD.csv', delimiter=',')
        # y_pred10 = np.loadtxt('./data/t1_Composite.csv', delimiter=',')
        # #
        # fpr2, tpr2, thresholds2 = roc_curve(all_label, y_pred2, pos_label=1)
        # fpr3, tpr3, thresholds3 = roc_curve(all_label, y_pred3, pos_label=1)
        # fpr4, tpr4, thresholds4 = roc_curve(all_label, y_pred4, pos_label=1)
        # fpr5, tpr5, thresholds5 = roc_curve(all_label, y_pred5, pos_label=1)
        # fpr6, tpr6, thresholds6 = roc_curve(all_label, y_pred6, pos_label=1)
        # fpr7, tpr7, thresholds7 = roc_curve(all_label, y_pred7, pos_label=1)
        # fpr8, tpr8, thresholds8 = roc_curve(all_label, y_pred8, pos_label=1)
        # fpr9, tpr9, thresholds9 = roc_curve(all_label, y_pred9, pos_label=1)
        # fpr10, tpr10, thresholds9 = roc_curve(all_label, y_pred10, pos_label=1)
        # #
        # roc_auc2 = auc(fpr2, tpr2)
        # roc_auc3 = auc(fpr3, tpr3)
        # roc_auc4 = auc(fpr4, tpr4)
        # roc_auc5 = auc(fpr5, tpr5)
        # roc_auc6 = auc(fpr6, tpr6)
        # roc_auc7 = auc(fpr7, tpr7)
        # roc_auc8 = auc(fpr8, tpr8)
        # roc_auc9 = auc(fpr9, tpr9)
        # roc_auc10 = auc(fpr10, tpr10)
        #
        # # # Plot ROC curve
        # plt.plot(fpr, tpr, label='CNN (area = %0.3f)' % roc_auc)
        # plt.plot(fpr2, tpr2, label='Fathmm-MKL (area = %0.3f)' % roc_auc2)
        # plt.plot(fpr3, tpr3, label='GWAS3D (area = %0.3f)' % roc_auc3)
        # plt.plot(fpr4, tpr4, label='Funseq (area = %0.3f)' % roc_auc4)
        # plt.plot(fpr5, tpr5, label='Funseq2 (area = %0.3f)' % roc_auc5)
        # plt.plot(fpr6, tpr6, label='GWAVA_TSS (area = %0.3f)' % roc_auc6)
        # plt.plot(fpr7, tpr7, label='SuRFR (area = %0.3f)' % roc_auc7)
        # plt.plot(fpr8, tpr8, label='DANN (area = %0.3f)' % roc_auc8)
        # plt.plot(fpr9, tpr9, label='CADD (area = %0.3f)' % roc_auc9)
        # plt.plot(fpr10, tpr10, label='Composite (area = %0.3f)' % roc_auc10)
        # # plt.plot(fpr11, tpr11, label='Combined (area = %0.3f)' % roc_auc11)
        #
        # plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.0])
        # plt.xlabel('False Positive Rate or (1 - Specifity)')
        # plt.ylabel('True Positive Rate or (Sensitivity)')
        # plt.title('allelc imbalanced')
        # plt.legend(loc="lower right")
        # plt.show()
        # print("finished")


def main(argv=None):
    # train()
    test1()


if __name__ == '__main__':
    main()
