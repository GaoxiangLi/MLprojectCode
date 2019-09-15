import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
OUTPUT_NODE = 2
LAYER1_NODE = 64
CONV1_SIZE = 3
CONV2_SIZE = 3
FC1_NODE = 64


def inference(input_tensor, args):
    input_tensor = tf.reshape(input_tensor, [-1, 1, 74, 1])
    kernel_size = args.kernel_size
    drop_rate = args.dropout_rate
    REGULARIZATION_RATE = args.L2_regularizer
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable(
            "weight", [1, kernel_size, 1, 5],
            initializer=tf.contrib.layers.xavier_initializer())
        conv1_biases = tf.get_variable("bias", 5, initializer=tf.contrib.layers.xavier_initializer())
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv1 = tf.nn.local_response_normalization(conv1)
        relu1 = tf.nn.leaky_relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding="SAME")

    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable(
            "weight", [1, 5, 5, 10],
            initializer=tf.contrib.layers.xavier_initializer())
        conv2_biases = tf.get_variable("bias", 10, initializer=tf.contrib.layers.xavier_initializer())
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.leaky_relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')
        pool_shape = pool2.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        reshaped = tf.reshape(pool2, [-1, nodes])

    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, FC1_NODE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)(fc1_weights))
        fc1_biases = tf.get_variable("bias", [FC1_NODE], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        fc1 = tf.nn.dropout(fc1, drop_rate)

    with tf.variable_scope('layer5-fc2'):
        fc2_weights = tf.get_variable("weight", [FC1_NODE, 32],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)(fc2_weights))
        fc2_biases = tf.get_variable("bias", [32], initializer=tf.constant_initializer(0.1))

        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
        fc2 = tf.nn.dropout(fc2, drop_rate)

    with tf.variable_scope('layer6-fc3'):
        fc3_weights = tf.get_variable("weight", [32, 2],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)(fc3_weights))
        fc3_biases = tf.get_variable("bias", [2], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc2, fc3_weights) + fc3_biases
        logit_softmax = tf.nn.softmax(logit)
    return logit, logit_softmax
