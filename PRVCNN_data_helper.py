import numpy as np

NUM_EXAMPLE = 60977


# padding
def data_helper():
    # features
    c = np.loadtxt('./data/whole_feature_inorder.csv', delimiter=',')
    # labels
    b = np.loadtxt('./data/whole_label_inorder.csv', delimiter=',')
    c.reshape([-1, 69])
    #
    b.reshape([-1, 2])
    features = []
    labels = b
    for i in range(NUM_EXAMPLE):
        features = np.append(features, np.pad(c[i], (6, 6), 'constant'))

    features = np.reshape(features, [-1, 81])
    # labels = np.reshape(labels, [-1, 2])
    np.savetxt("./features.csv", features, delimiter=",")
    np.savetxt("./labels.csv", labels, delimiter=",")
    print("finished")


# subsample
def subsample(features, labels):
    training_features = []
    training_labels = []
    test_features = []
    test_labels = []
    compare_set1 = []
    compare_set2 = []
    a = np.loadtxt('./fathmn-MKL.csv', delimiter=',')
    b = np.loadtxt('./GWAVA_matched.csv.csv', delimiter=',')
    for i in range(NUM_EXAMPLE):
        if i % 10 == 1 or i % 3 == 0:
            test_features = np.append(test_features, features[i])
            test_labels = np.append(test_labels, labels[i])
        else:
            training_features = np.append(training_features, features[i])
            training_labels = np.append(training_labels, labels[i])
            compare_set1 = np.append(compare_set1, a[i])
            compare_set2 = np.append(compare_set2, b[i])

    training_features = np.reshape(training_features, [-1, 81])
    test_features = np.reshape(test_features, [-1, 81])
    training_labels = np.reshape(training_labels, [-1, 2])
    test_labels = np.reshape(test_labels, [-1, 2])
    compare_set1 = np.reshape(compare_set1, [-1, 1])
    compare_set2 = np.reshape(compare_set2, [-1, 1])
    np.savetxt("./compare_set1.csv", compare_set1, delimiter=",")
    np.savetxt("./compare_set2.csv", compare_set2, delimiter=",")
    return training_features, training_labels, test_features, test_labels
