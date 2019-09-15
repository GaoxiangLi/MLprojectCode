from sklearn import svm
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt


def SVM(args):
    # load data
    print("Loading data")
    f_dir = args.training_feature
    l_dir = args.training_label
    training_feature = np.loadtxt('%s' % (f_dir), delimiter=',')
    training_label = np.loadtxt('%s' % (l_dir), delimiter=',')
    f_dir2 = args.testing_feature
    l_dir2 = args.testing_label
    test_feature = np.loadtxt('%s' % (f_dir2), delimiter=',')
    test_label = np.loadtxt('%s' % (l_dir2), delimiter=',')

    print("Training")
    clf = svm.SVC(kernel='rbf')
    clf.fit(training_feature, training_label)
    result = clf.predict_proba(test_feature)
    result = result[:, 1:2]
    filename = args.result_score_file
    np.savetxt("./result/%s" % (filename), result, delimiter=",")
    print("training model prediction score for SVM saved in ./result/%s" % (filename))
    fpr, tpr, thresholds = roc_curve(test_label, result)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate or (1 - Specifity)')
    plt.ylabel('True Positive Rate or (Sensitivity)')
    plt.title('SVM')
    plt.legend(loc="lower right")
    plt.show()
    print("finished")



