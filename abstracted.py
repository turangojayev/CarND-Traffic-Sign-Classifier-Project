import random
from functools import partial
from operator import itemgetter

from augmentations import Rotator, HistogramEqualizer, Squeezer, Flipper, ContrastNormalization
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle
import os
import pandas as pd
from sklearn.utils import shuffle

EPOCHS = 200
BATCH_SIZE = 256
np.random.seed(12345)


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
class ConvolutionalLayer:
    def __init__(self,
                 kernel_size,
                 input_channels,
                 filters,
                 strides=None,
                 activation=tf.nn.relu,
                 padding='VALID'):
        self._filters = tf.get_variable("conv-{}-{}".format(input_channels, filters),
                                        shape=[*kernel_size, input_channels, filters],
                                        initializer=tf.contrib.layers.xavier_initializer())
        if strides is None:
            strides = [1, 1]
        self._strides = strides
        self._bias = tf.Variable(tf.zeros(filters))
        self._activation = activation
        self._padding = padding

    def __call__(self, input):
        conv = tf.add(
            tf.nn.conv2d(input, self._filters, strides=[1, *self._strides, 1], padding=self._padding), self._bias)
        return self._activation(conv)


class MaxPool:
    def __init__(self, kernel_size, strides):
        self._kernel_size = kernel_size
        self._strides = strides

    def __call__(self, input):
        return tf.nn.max_pool(input, ksize=[1, *self._kernel_size, 1], strides=[1, *self._strides, 1], padding='VALID')


class Dense:
    def __init__(self,
                 output_size,
                 input_size,
                 activation=tf.nn.relu):
        self._weights = tf.get_variable("weights{}-{}".format(input_size, output_size),
                                        shape=[input_size, output_size],
                                        initializer=tf.contrib.layers.xavier_initializer())
        self._bias = tf.Variable(tf.zeros(output_size))
        self._activation = activation

    def __call__(self, input):
        return tf.nn.relu(tf.add(tf.matmul(input, self._weights), self._bias))


# TODO:check AlexNet => l1_maxpool = tf.nn.max_pool(l1_out, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
def build_model2(x, num_of_classes, dense_keep_prob, conv_keep_prob):
    l1_depth = 32
    l2_depth = 32
    dense1_out_size = 300
    dense2_out_size = 256
    dense3_out_size = 86
    input_channels = int(x.shape[3])

    conv1 = ConvolutionalLayer(kernel_size=[5, 5], input_channels=input_channels, filters=l1_depth, mean=0, stddev=0.1)
    l1_maxpool = MaxPool(kernel_size=[2, 2], strides=[2, 2])
    conv2 = ConvolutionalLayer(kernel_size=[3, 3], input_channels=l1_depth, filters=l2_depth, mean=0, stddev=0.1)
    l2_maxpool = MaxPool(kernel_size=[2, 2], strides=[2, 2])

    conv1_out = conv1(x)
    l1_maxout = tf.nn.dropout(conv1_out, conv_keep_prob)
    l1_maxout = l1_maxpool(l1_maxout)

    conv2_out = conv2(l1_maxout)
    l2_maxout = tf.nn.dropout(conv2_out, conv_keep_prob)
    l2_maxout = l2_maxpool(l2_maxout)

    flattened = tf.contrib.layers.flatten(l2_maxout)
    l1_flattened = tf.contrib.layers.flatten(l1_maxpool(l1_maxout))
    concatenated = tf.concat([flattened, l1_flattened], axis=1)

    dense1 = Dense(dense1_out_size, int(concatenated.shape[1]), mean=0, stddev=0.1)
    dense1_out = dense1(concatenated)
    dense1_out = tf.nn.dropout(dense1_out, dense_keep_prob)

    # dense2 = Dense(dense2_out_size, dense1_out_size, mean=0, stddev=0.1)
    # dense2_out = dense2(dense1_out)
    # dense2_out = tf.nn.dropout(dense2_out, dense_keep_prob)
    #
    # dense3 = Dense(dense3_out_size, dense2_out_size, mean=0, stddev=0.1)
    # dense3_out = dense3(dense2_out)
    # dense3_out = tf.nn.dropout(dense3_out, dense_keep_prob)

    # dense3 = Dense(dense3_out_size, dense2_out_size, mean=0, stddev=0.1)
    # dense3_out = dense3(dense2_out)

    dense4 = Dense(num_of_classes, dense1_out_size, mean=0, stddev=0.1)
    dense4_out = dense4(dense1_out)

    print(conv1_out.shape)
    print(l1_maxout.shape)
    print(conv2_out.shape)
    print(l2_maxout.shape)
    print(l1_flattened.shape)
    print(flattened.shape)
    print(concatenated.shape)
    print(dense1_out.shape)
    # print(dense2_out.shape)
    print(dense4_out.shape)

    return dense4_out


def build_model3(x, num_of_classes, dense_keep_prob, conv_keep_prob):
    l1_depth = 50
    l2_depth = 75
    dense1_out_size = 300
    dense2_out_size = 256
    dense3_out_size = 86
    input_channels = int(x.shape[3])

    conv1 = ConvolutionalLayer(kernel_size=[5, 5], input_channels=input_channels, filters=l1_depth)
    l1_maxpool = MaxPool(kernel_size=[2, 2], strides=[2, 2])
    conv2 = ConvolutionalLayer(kernel_size=[3, 3], input_channels=l1_depth, filters=l2_depth)
    l2_maxpool = MaxPool(kernel_size=[2, 2], strides=[2, 2])

    conv1_out = conv1(x)
    l1_maxout = tf.nn.dropout(conv1_out, conv_keep_prob)
    l1_maxout = l1_maxpool(l1_maxout)

    conv2_out = conv2(l1_maxout)
    l2_maxout = tf.nn.dropout(conv2_out, conv_keep_prob)
    l2_maxout = l2_maxpool(l2_maxout)

    flattened = tf.contrib.layers.flatten(l2_maxout)
    # l1_flattened = tf.contrib.layers.flatten(l1_maxpool(l1_maxout))
    # concatenated = tf.concat([flattened, l1_flattened], axis=1)

    dense1 = Dense(dense1_out_size, int(flattened.shape[1]))
    dense1_out = dense1(flattened)
    dense1_out = tf.nn.dropout(dense1_out, dense_keep_prob)

    # dense2 = Dense(dense2_out_size, dense1_out_size, mean=0, stddev=0.1)
    # dense2_out = dense2(dense1_out)
    # dense2_out = tf.nn.dropout(dense2_out, dense_keep_prob)
    #
    # dense3 = Dense(dense3_out_size, dense2_out_size, mean=0, stddev=0.1)
    # dense3_out = dense3(dense2_out)
    # dense3_out = tf.nn.dropout(dense3_out, dense_keep_prob)

    # dense3 = Dense(dense3_out_size, dense2_out_size, mean=0, stddev=0.1)
    # dense3_out = dense3(dense2_out)

    dense4 = Dense(num_of_classes, dense1_out_size)
    dense4_out = dense4(dense1_out)

    print(conv1_out.shape)
    print(l1_maxout.shape)
    print(conv2_out.shape)
    print(l2_maxout.shape)
    # print(l1_flattened.shape)
    print(flattened.shape)
    # print(concatenated.shape)
    print(dense1_out.shape)
    # print(dense2_out.shape)
    print(dense4_out.shape)

    return dense4_out


def build_model4(x, num_of_classes, dense_keep_prob, conv_keep_prob):
    l1_depth = 50
    l2_depth = 75
    l3_depth = 100

    dense1_out_size = 300
    dense2_out_size = 256
    dense3_out_size = 86
    input_channels = int(x.shape[3])

    conv1 = ConvolutionalLayer(kernel_size=[7, 7], input_channels=input_channels, filters=l1_depth)
    l1_maxpool = MaxPool(kernel_size=[2, 2], strides=[2, 2])
    conv2 = ConvolutionalLayer(kernel_size=[4, 4], input_channels=l1_depth, filters=l2_depth)
    l2_maxpool = MaxPool(kernel_size=[2, 2], strides=[2, 2])
    conv3 = ConvolutionalLayer(kernel_size=[2, 2], input_channels=l2_depth, filters=l3_depth)
    l3_maxpool = MaxPool(kernel_size=[2, 2], strides=[2, 2])

    conv1_out = conv1(x)
    l1_maxout = tf.nn.dropout(conv1_out, conv_keep_prob)
    l1_maxout = l1_maxpool(l1_maxout)

    conv2_out = conv2(l1_maxout)
    l2_maxout = tf.nn.dropout(conv2_out, conv_keep_prob)
    l2_maxout = l2_maxpool(l2_maxout)


    conv3_out = conv3(l2_maxout)
    l3_maxout = tf.nn.dropout(conv3_out, conv_keep_prob)
    l3_maxout = l3_maxpool(l3_maxout)

    flattened = tf.contrib.layers.flatten(l3_maxout)
    # flattened = tf.contrib.layers.flatten(l2_maxout)
    # l1_flattened = tf.contrib.layers.flatten(l1_maxpool(l1_maxout))
    # concatenated = tf.concat([flattened, l1_flattened], axis=1)

    dense1 = Dense(dense1_out_size, int(flattened.shape[1]))
    dense1_out = dense1(flattened)
    dense1_out = tf.nn.dropout(dense1_out, dense_keep_prob)

    output = Dense(num_of_classes, dense1_out_size)
    output_out = output(dense1_out)

    print(conv1_out.shape)
    print(l1_maxout.shape)
    print(conv2_out.shape)
    print(l2_maxout.shape)
    # print(l1_flattened.shape)
    print(flattened.shape)
    # print(concatenated.shape)
    print(dense1_out.shape)
    print(output_out.shape)
    return output_out


def _iterate_over_batches(X_data, y_data, class_weights=None, dense_keep_probability=1., conv_keep_probability=1.):
    num_examples = len(X_data)
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        weights = np.ones(len(batch_x))
        if class_weights is not None:
            weights *= class_weights[batch_y]
        yield {
                  x: batch_x,
                  y: _to_one_hot(batch_y, num_classes),
                  w: weights,
                  dense_keep_prob: dense_keep_probability,
                  conv_keep_prob: conv_keep_probability
              }, \
              len(batch_x)


def evaluate(X_data, y_data, operation, class_weights=None):
    accuracies_and_batch_sizes = list(
        do_epoch(
            tf.get_default_session(),
            _iterate_over_batches(X_data, y_data, class_weights),
            operation
        ))

    accuracies = np.array(list(map(itemgetter(0), accuracies_and_batch_sizes)))
    batch_sizes = np.array(list(map(itemgetter(1), accuracies_and_batch_sizes)))
    return np.dot(accuracies, batch_sizes) / len(X_data)


def do_epoch(session, iterable, *operation):
    for i in iterable:
        yield session.run(*operation, feed_dict={**(i[0])}), i[1]


def _to_one_hot(labels, depth):
    result = np.zeros(shape=(len(labels), depth))
    result[np.arange(len(labels)), labels] = 1
    return result.astype(np.int8)


def _get_default_model_name():
    from datetime import datetime
    now = str(datetime.now())
    return 'model-{}'.format(now)


def _create_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


class EarlyStopper:
    def __init__(self, saver, session, which=min, patience=5, model_description=None):
        self._saver = saver
        self._session = session
        self._method = which
        self._last_value = None
        self._current_epoch = 0
        self._best_epoch = 0
        self._patience = patience
        self._best_path = None

        directory = os.path.join(os.getcwd(), 'models')
        _create_if_not_exists(directory)
        if model_description is None:
            model_description = _get_default_model_name()
        self._basename = os.path.join(directory, model_description)

    def should_stop(self, value):
        self._current_epoch += 1
        stop = False
        if self._last_value is None or self._last_value != self._method(value, self._last_value):
            self._last_value = value
            self._best_epoch = self._current_epoch
            self._best_path = os.path.join('{}-{:.3f}-{}'.format(self._basename, value, self._current_epoch))
            self._saver.save(self._session, self._best_path)
        elif self._current_epoch - self._best_epoch > self._patience:
            stop = True
        return stop

    def load(self):
        if self._best_path is not None:
            self._saver.restore(self._session, self._best_path)
        else:
            raise ValueError('No path to the model to load, train first!')


def train(X_train,
          y_train,
          X_valid,
          y_valid,
          accuracy,
          train_class_weights,
          valid_class_weights,
          dense_dropout,
          conv_dropout):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        early_stopper = EarlyStopper(saver, sess, which=max, patience=50, model_description='mirt')
        sess.run(tf.global_variables_initializer())
        for i in range(EPOCHS):
            X_train, y_train = shuffle(X_train, y_train)

            results = list(do_epoch(sess,
                                    _iterate_over_batches(X_train, y_train, train_class_weights,
                                                          dense_keep_probability=dense_dropout,
                                                          conv_keep_probability=conv_dropout),
                                    [training,
                                     mean_loss,
                                     accuracy]))
            evaluations = list(map(itemgetter(0), results))
            losses = np.array(list(map(itemgetter(1), evaluations)))
            accuracies = np.array(list(map(itemgetter(2), evaluations)))
            batch_sizes = np.array(list(map(itemgetter(1), results)))

            train_loss = np.dot(losses, batch_sizes) / np.sum(batch_sizes)
            train_accuracy = np.dot(accuracies, batch_sizes) / np.sum(batch_sizes)

            validation_accuracy = evaluate(X_valid, y_valid, accuracy)
            validation_loss = evaluate(X_train, y_train, mean_loss, valid_class_weights)
            print(
                "Epoch {} train loss={:.3f}, acc.={:.3f}\tvalid loss = {:.3f}, acc. = {:.3f}".format(
                    i + 1, train_loss, train_accuracy, validation_loss, validation_accuracy))
            test_accuracy = evaluate(X_test, y_test, accuracy)
            print('test_accuracy', test_accuracy)

            if early_stopper.should_stop(validation_accuracy):
                break

        early_stopper.load()
        train_accuracy = evaluate(X_train, y_train, accuracy)
        valid_accuracy = evaluate(X_valid, y_valid, accuracy)
        test_accuracy = evaluate(X_test, y_test, accuracy)
        print('train_accuracy', train_accuracy)
        print('best validation accuracy', valid_accuracy)
        print('test_accuracy', test_accuracy)


if __name__ == "__main__":
    dense_dropout = 0.7
    conv_dropout = 0.7
    # X_train = 255 - X_train
    # X_valid = 255- X_valid
    # X_test = 255 - X_test
    X_train = np.array(list(map(lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), X_train)))
    X_valid = np.array(list(map(lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), X_valid)))
    X_test = np.array(list(map(lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), X_test)))
    X_train = X_train.reshape(*X_train.shape, -1)
    X_valid = X_valid.reshape(*X_valid.shape, -1)
    X_test = X_test.reshape(*X_test.shape, -1)

    # X_train, y_train = augment(X_train, y_train, equalize_histogram, size=20000)
    rows, columns, *_ = X_train[0].shape

    # random_indices = np.random.choice(len(X_train), size=20000, replace=False)
    # augmented = list(map(lambda image:255-image, X_train[random_indices]))
    # X_additional = np.array(augmented).reshape(20000, *X_train.shape[1:])
    # X_train = np.vstack((X_train, X_additional))
    # y_train = np.concatenate((y_train, y_train[random_indices]))
    # X_train = 255 - X_train
    # X_valid = 255- X_valid
    # X_test = 255 - X_test

    # equalizer = HistogramEqualizer()
    # X_train, y_train, _ = equalizer(X_train, y_train, 30000)
    #
    # # rotator = Rotator(columns=columns, rows=rows, stddev_rotation_angle=10)
    # rotator = Rotator(columns=columns, rows=rows, prob_distr=partial(np.random.uniform, low=-20, high=20))
    # X_train, y_train, _ = rotator(X_train, y_train, size=40000)
    #
    # # squeezer = Squeezer(columns, rows, stddev_horizontal_scale_coef=0.05, stddev_vertical_scale_coef=0.05)
    # squeezer = Squeezer(columns, rows, prob_distr=partial(np.random.uniform, low=-0.1, high=0.1))
    # X_train, y_train, _ = squeezer(X_train, y_train, size=60000)
    #
    #
    # contraster = ContrastNormalization()
    # X_train, y_train, _ = contraster(X_train, y_train, size=60000)


    equalizer = HistogramEqualizer()
    X_train, y_train, _ = equalizer(X_train, y_train, 34799)
    X_train = X_train[34799:]
    y_train = y_train[34799:]

    X_valid, y_valid, _ = equalizer(X_valid, y_valid, 4410)
    X_valid = X_valid[4410:]
    y_valid = y_valid[4410:]

    X_test, y_test, _ = equalizer(X_test, y_test, 12630)
    X_test = X_test[12630:]
    y_test = y_test[12630:]


    # rotator = Rotator(columns=columns, rows=rows, stddev_rotation_angle=10)
    rotator = Rotator(columns=columns, rows=rows, prob_distr=partial(np.random.uniform, low=-20, high=20))
    X_train, y_train, _ = rotator(X_train, y_train, size=30000)

    # squeezer = Squeezer(columns, rows, stddev_horizontal_scale_coef=0.05, stddev_vertical_scale_coef=0.05)
    squeezer = Squeezer(columns, rows, prob_distr=partial(np.random.uniform, low=-0.1, high=0.1))
    X_train, y_train, _ = squeezer(X_train, y_train, size=60000)


    # contraster = ContrastNormalization()
    # X_train, y_train, _ = contraster(X_train, y_train, size=60000)



    # flipper = Flipper()
    # X_train, y_train, _ = flipper(X_train, y_train, size=50000)

    print(X_train.shape)
    print('gray', X_train.shape)

    x = tf.placeholder(tf.float32, shape=[None, *X_train[0].shape])
    x = tf.placeholder(tf.float32, shape=[None, *X_train[0].shape])
    y = tf.one_hot(tf.placeholder(tf.int32, (None)), num_classes)
    w = tf.placeholder(tf.float32, shape=[None])
    dense_keep_prob = tf.placeholder(tf.float32)
    conv_keep_prob = tf.placeholder(tf.float32)

    # logits = build_model3(x, num_of_classes=num_classes, dense_keep_prob=dense_keep_prob, conv_keep_prob=conv_keep_prob)
    logits = build_model4(x, num_of_classes=num_classes, dense_keep_prob=dense_keep_prob, conv_keep_prob=conv_keep_prob)
    cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits, weights=w)
    mean_loss = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    training = optimizer.minimize(mean_loss)

    correct_prediction = tf.equal(tf.argmax(logits, axis=1), tf.argmax(y, axis=1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    train_class_weights = get_class_percents(y_train, keys)
    train_class_weights = np.sqrt(train_class_weights)
    valid_class_weights = sign_names.valid_percents
    valid_class_weights = np.sqrt(valid_class_weights)

    train(X_train,
          y_train,
          X_valid,
          y_valid,
          accuracy_operation,
          train_class_weights=None,
          valid_class_weights=None,
          dense_dropout=dense_dropout,
          conv_dropout=conv_dropout)
