import PRVCS_data_helper
import numpy as np

# features = np.loadtxt('./features.csv', delimiter=',')
# labels = np.loadtxt('./labels.csv', delimiter=',')
# training_feature, training_label, test_feature, test_label = PRVCS_data_helper.subsample(features, labels)
# np.savetxt("./training_feature2.csv", training_feature, delimiter=",")
# np.savetxt("./training_label2.csv", training_label, delimiter=",")
# np.savetxt("./test_feature2.csv", test_feature, delimiter=",")
# np.savetxt("./test_label2.csv", test_label, delimiter=",")
# print("done")

test_label = np.loadtxt('./test_label1.csv', delimiter=',')
test_label = np.argmax(test_label,axis=1)
print(test_label)