import tensorflow as tf
import numpy as np
import PRVCNN_inference
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import matthews_corrcoef
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def test(args):
    x = tf.placeholder(tf.float32, [None, 74], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 2], name='y-input')
    y, y_softmax = PRVCNN_inference.inference(x, args)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        saver = tf.train.Saver()
        test_name = args.test_name
        saver.restore(sess, "./log/%s" % (test_name))
        f_dir = args.testing_feature
        l_dir = args.testing_label
        test_feature = np.loadtxt('%s' % (f_dir), delimiter=',')
        test_label = np.loadtxt('%s' % (l_dir), delimiter=',')
        test_feature = np.reshape(test_feature, [-1, 74])
        test_label = np.reshape(test_label, [-1, 2])

        print("Testing1 start")
        y_pred = sess.run(y_softmax, feed_dict={x: test_feature})

        # Calculate metrics
        y_pred = np.reshape(y_pred, [-1, 2])
        y_prob = y_pred[:, 1:2]
        filename = args.result_score_file
        np.savetxt("./result/%s" % (filename), y_prob, delimiter=",")
        print("training model prediction score for test1 saved in ./result/%s" % (filename))
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
    test()


if __name__ == '__main__':
    main()
