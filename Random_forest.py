from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt

# load data
training_feature = np.loadtxt('./data/training_feature3.csv', delimiter=',')
training_label = np.loadtxt('./data/svm_train_label.csv', delimiter=',')
test_feature = np.loadtxt('./data/test_feature3.csv', delimiter=',')
test_label = np.loadtxt('./data/svm_test_label.csv', delimiter=',')

clf = RandomForestClassifier(n_estimators=500, max_depth=None, random_state=0)
clf.fit(training_feature, training_label)
# score_rbf = clf_rbf.score(test_feature, test_label)
# print("The score of rbf is : %f" % score_rbf)
result = clf.predict(test_feature)
fpr, tpr, thresholds = roc_curve(test_label, result)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate or (1 - Specifity)')
plt.ylabel('True Positive Rate or (Sensitivity)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
print("finished")
# # kernel = 'linear'
# clf_linear = svm.SVC(kernel='linear')
# clf_linear.fit(X_train,y_train)
# score_linear = clf_linear.score(X_test,y_test)
# print("The score of linear is : %f"%score_linear)
#
# # kernel = 'poly'
# clf_poly = svm.SVC(kernel='poly')
# clf_poly.fit(X_train,y_train)
# score_poly = clf_poly.score(X_test,y_test)
# print("The score of poly is : %f"%score_poly)
