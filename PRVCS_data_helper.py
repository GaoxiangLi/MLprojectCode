import numpy as np

NUM_EXAMPLE = 5229


# padding
def data_helper():
    # features
    c = np.loadtxt('./data_xx1_1.csv', delimiter=',')
    # labels
    b = np.loadtxt('./data_xx1_1.csv', delimiter=',')
    c.reshape([-1, 65])
    #
    b.reshape([-1, 65])
    features = []
    labels = []
    for i in range(NUM_EXAMPLE):
        features = np.append(features, np.pad(c[i], (8, 8), 'constant'))
        labels = np.append(labels, np.pad(b[i], (8, 8), 'constant'))

    features = np.reshape(features, [-1, 81])
    labels = np.reshape(labels, [-1, 81])
    return features, labels


# subsample
def subsample(features, labels):
    training_features = []
    training_labels = []
    test_features = []
    test_labels = []
    for i in range(NUM_EXAMPLE):
        if i % 10 == 1 or i % 10 == 2:
            test_features = np.append(test_features, features[i])
            test_labels = np.append(test_labels, labels[i])
        else:
            training_features = np.append(training_features, features[i])
            training_labels = np.append(training_labels, labels[i])

    training_features = np.reshape(training_features, [-1, 81])
    test_features = np.reshape(test_features, [-1, 81])
    return training_features, training_labels, test_features, test_labels
