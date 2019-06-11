import numpy as np

NUM_EXAMPLE = 60977


# padding
def data_helper():
    # features
    c = np.loadtxt('./xx1_feature.csv', delimiter=',')
    # labels
    b = np.loadtxt('./xx1_label.csv', delimiter=',')
    c.reshape([-1, 65])
    #
    b.reshape([-1, 2])
    features = []
    labels = b
    for i in range(NUM_EXAMPLE):
        features = np.append(features, np.pad(c[i], (8, 8), 'constant'))

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
    for i in range(NUM_EXAMPLE):
        if i % 10 == 3 or i % 10 == 4:
            test_features = np.append(test_features, features[i])
            test_labels = np.append(test_labels, labels[i])
        else:
            training_features = np.append(training_features, features[i])
            training_labels = np.append(training_labels, labels[i])

    training_features = np.reshape(training_features, [-1, 81])
    test_features = np.reshape(test_features, [-1, 81])
    training_labels = np.reshape(training_labels, [-1, 2])
    test_labels = np.reshape(test_labels, [-1, 2])

    return training_features, training_labels, test_features, test_labels
