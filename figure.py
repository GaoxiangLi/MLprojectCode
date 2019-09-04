import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import matthews_corrcoef
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
rf_result = np.loadtxt("./t4_RF.csv", delimiter=',')
cnn_result = np.loadtxt("./score4.csv", delimiter=',')
svm_result = np.loadtxt('./data/t4_CADD.csv', delimiter=',')

cadd_label = np.loadtxt("./data/t4_label.csv", delimiter=',')
test_label = np.loadtxt('./data/t4_ml_label.csv', delimiter=',')

fpr1, tpr1, thresholds1 = roc_curve(test_label, rf_result)
fpr2, tpr2, thresholds2 = roc_curve(test_label, cnn_result)
fpr3, tpr3, thresholds9 = roc_curve(cadd_label, svm_result, pos_label=1)
roc_auc1 = auc(fpr1, tpr1)
roc_auc2 = auc(fpr2, tpr2)
roc_auc3 = auc(fpr3, tpr3)
plt.plot(fpr2, tpr2, label='CNN (area = %0.3f)' % roc_auc2)
plt.plot(fpr3, tpr3, label='SVM (area = %0.3f)' % roc_auc3)
plt.plot(fpr1, tpr1, label='Random Forest (area = %0.3f)' % roc_auc1)
# plt.plot(fpr2, tpr2, label='CNN (area = %0.3f)' % roc_auc2)
plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate or (1 - Specifity)')
plt.ylabel('True Positive Rate or (Sensitivity)')
plt.title('Machine Learning Methods- validated regulatory ')
plt.legend(loc="lower right")
plt.show()
print("finished")
