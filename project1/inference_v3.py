import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
INPUT_NODE = 1156
OUTPUT_NODE = 1
LAYER1_NODE = 256
LAYER2_NODE = 128
REGULARIZATION_RATE = 0.01


def get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    tf.add_to_collection('losses', regularizer(weights))
    return weights


# decrease the num of layers and nodes
def inference(input_tensor, regularizer):
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
        layer1_dropout = tf.nn.dropout(layer1, 0.5)

    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1_dropout, weights) + biases
    return layer2


def inference2(input_tensor, drop_rate):
    input_tensor = tf.reshape(input_tensor, [-1, 34, 34, 1])
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable(
            "weight", [3, 3, 1, 2],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)(conv1_weights))
        conv1_biases = tf.get_variable("bias", 2, initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable(
            "weight", [3, 3, 2, 4],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)(conv2_weights))
        conv2_biases = tf.get_variable("bias", 4, initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        pool_shape = pool2.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        reshaped = tf.reshape(pool2, [-1, nodes])

    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, 512],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)(fc1_weights))
        fc1_biases = tf.get_variable("bias", [512], initializer=tf.constant_initializer(0.1))

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        fc1 = tf.nn.dropout(fc1, drop_rate)


    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable("weight", [512, 1],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)(fc2_weights))
        fc2_biases = tf.get_variable("bias", [1], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases
    logit = np.reshape(logit, [1])
    logit = logit[0]
    return logit
