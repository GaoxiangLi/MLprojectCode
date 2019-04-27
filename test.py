import tensorflow as tf
import numpy as np
import inference_2
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.99
BATCH_SIZE = 25
REGULARIZATION_RATE = 0.01
TRAINING_STEPS = 5000
MOVING_AVERAGE_DECAY = 0.99
NUM_EXAMPLE = 308

features = np.load('./X.npy')
labels = np.load('./Y.npy')
features = features.reshape([NUM_EXAMPLE, -1])
labels = labels.reshape([NUM_EXAMPLE, -1])

training_feature = []
training_label = []
test_feature = []
test_label = []
# for i in range(NUM_EXAMPLE):
#     np.pad(features[i], (34, 35), 'constant')
# for i in range(NUM_EXAMPLE):
#     if i % 8 == 0 or i % 9 == 0:
#         test_feature = np.append(test_feature, np.pad(features[i], (34, 35), 'constant'))
#         test_label = np.append(test_label, labels[i])
#     else:
#         training_feature = np.append(training_feature, np.pad(features[i], (34, 35), 'constant'))
#         training_label = np.append(training_label, labels[i])
# training_feature = np.reshape(training_feature, [-1, 1156])
# test_feature = np.reshape(test_feature, [-1, 1156])
# print(len(training_label))
# print(len(training_feature))
# print(len(test_feature))
# print(len(test_label))
y1 = [[1.1234],[2]]
y2 = [[1],[2]]
acc = r2_score(y1, y2)
print(acc)
