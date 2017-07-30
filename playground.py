import random
from operator import itemgetter

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle
import os
import pandas as pd
from sklearn.utils import shuffle

EPOCHS = 150
BATCH_SIZE = 256
random.seed(1234)


def get_data(path):
    with open(path, mode='rb') as f:
        return pickle.load(f)


directory = 'data'
training_file = os.path.join(directory, 'train.p')
validation_file = os.path.join(directory, 'valid.p')
testing_file = os.path.join(directory, 'test.p')

train = get_data(training_file)
valid = get_data(validation_file)
test = get_data(testing_file)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

sign_names = pd.read_csv('signnames.csv')

num_train = X_train.shape[0]
num_validation = X_valid.shape[0]
num_test = X_test.shape[0]
image_shape = X_train[0].shape
num_classes = np.unique(y_train).shape[0]

print("Number of training examples =", num_train)
print("Number of validation examples =", num_validation)
print("Number of testing examples =", num_test)
print("Image data shape =", image_shape)
import cv2

print(cv2.cvtColor(X_train[0], cv2.COLOR_RGB2GRAY).shape)
print("Number of classes =", num_classes)


def get_class_percents(y, keys):
    classes, counts = np.unique(y, return_counts=True)
    cls2percent = {cls: percent for cls, percent in zip(classes, 100 * counts / sum(counts))}
    return np.array([cls2percent[cls] for cls in keys])


keys = sign_names.ClassId.values
sign_names = sign_names.assign(train_percents=get_class_percents(y_train, keys))
sign_names = sign_names.assign(valid_percents=get_class_percents(y_valid, keys))
sign_names = sign_names.assign(test_percents=get_class_percents(y_test, keys))


# print(sign_names.sort_values(by=['train_percents', 'valid_percents', 'test_percents'], ascending=False))

# TODO: connect cnn layers directly to output
def model_gray(x, num_of_classes):
    mean = 0
    stddev = 0.1
    l1_depth = 16
    l1_filters = tf.Variable(tf.random_normal(shape=[5, 5, 1, l1_depth], mean=mean, stddev=stddev))
    l1_bias = tf.Variable(tf.zeros(l1_depth))
    l1_conv = tf.add(tf.nn.conv2d(x, l1_filters, strides=[1, 1, 1, 1], padding='VALID'), l1_bias)
    l1_out = tf.nn.relu(l1_conv)
    l1_maxpool = tf.nn.max_pool(l1_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # TODO:check AlexNet => l1_maxpool = tf.nn.max_pool(l1_out, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
    print(l1_maxpool.shape)

    l2_depth = 16
    l2_filters = tf.Variable(tf.random_normal(shape=[3, 3, l1_depth, l2_depth], mean=mean, stddev=stddev))
    l2_bias = tf.Variable(tf.zeros(l2_depth))
    l2_conv = tf.add(tf.nn.conv2d(l1_maxpool, l2_filters, strides=[1, 1, 1, 1], padding='VALID'), l2_bias)
    l2_out = tf.nn.relu(l2_conv)
    l2_maxpool = tf.nn.max_pool(l2_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    print(l2_maxpool.shape)

    flattened = tf.contrib.layers.flatten(l2_maxpool)
    print(flattened.shape)

    l3_out_size = 128
    l3_weights = tf.Variable(
        tf.truncated_normal(shape=[int(flattened.shape[1]), l3_out_size], mean=mean, stddev=stddev))
    l3_bias = tf.Variable(tf.zeros(l3_out_size))
    l3_out = tf.nn.relu(tf.add(tf.matmul(flattened, l3_weights), l3_bias))
    print(l3_out.shape)

    l4_out_size = 120
    l4_weights = tf.Variable(
        tf.truncated_normal(shape=[l3_out_size, l4_out_size], mean=mean, stddev=stddev))
    l4_bias = tf.Variable(tf.zeros(l4_out_size))
    l4_out = tf.nn.relu(tf.add(tf.matmul(l3_out, l4_weights), l4_bias))
    print(l4_out.shape)

    l6_out_size = 86
    l6_weights = tf.Variable(tf.random_normal(shape=[l4_out_size, l6_out_size], mean=mean, stddev=stddev))
    l6_bias = tf.Variable(tf.zeros(l6_out_size))
    l6_out = tf.nn.relu(tf.add(tf.matmul(l4_out, l6_weights), l6_bias))
    print(l6_out.shape)

    l5_weights = tf.Variable(tf.random_normal(shape=[l6_out_size, num_of_classes], mean=mean, stddev=stddev))
    l5_bias = tf.Variable(tf.zeros(num_of_classes))
    return tf.nn.relu(tf.add(tf.matmul(l6_out, l5_weights), l5_bias))


X_train = np.array(list(map(lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), X_train)))
X_valid = np.array(list(map(lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), X_valid)))
X_train = X_train.reshape(*X_train.shape, -1)
X_valid = X_valid.reshape(*X_valid.shape, -1)
print('gray', X_train.shape)

x = tf.placeholder(tf.float32, shape=[None, *X_train[0].shape])
y = tf.one_hot(tf.placeholder(tf.int32, (None)), num_classes)
logits = model_gray(x, num_of_classes=num_classes)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
mean_loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer()
training = optimizer.minimize(mean_loss)

correct_prediction = tf.equal(tf.argmax(logits, axis=1), tf.argmax(y, axis=1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: _to_one_hot(batch_y, num_classes)})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


def _to_one_hot(labels, depth):
    result = np.zeros(shape=(len(labels), depth))
    result[np.arange(len(labels)), labels] = 1
    return result.astype(np.int8)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training, feed_dict={x: batch_x, y: _to_one_hot(batch_y, num_classes)})

        training_accuracy = evaluate(X_train, y_train)
        validation_accuracy = evaluate(X_valid, y_valid)
        print("Epoch {}\tTraining accuracy = {:.3f}\tValidation accuracy = {:.3f}".format(
            i + 1, training_accuracy, validation_accuracy))
