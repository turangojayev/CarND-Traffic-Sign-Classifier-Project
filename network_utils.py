import os
import math
from operator import itemgetter

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from tqdm import tqdm


def conv2d(input,
           kernel_size,
           num_of_filters,
           strides=None,
           activation=tf.nn.relu,
           padding='VALID',
           name=None):
    input_channels = int(input.shape[3])
    if name is None:
        name = "conv-{}-{}".format(input_channels, num_of_filters)
    filters = tf.get_variable(name,
                              shape=[*kernel_size, input_channels, num_of_filters],
                              initializer=tf.contrib.layers.xavier_initializer())

    bias = tf.Variable(tf.zeros(num_of_filters))
    if strides is None:
        strides = [1, 1]

    conv = tf.add(tf.nn.conv2d(input, filters, strides=[1, *strides, 1], padding=padding), bias)
    return activation(conv)


def maxpool(input, kernel_size, strides, padding='VALID'):
    return tf.nn.max_pool(input, ksize=[1, *kernel_size, 1], strides=[1, *strides, 1], padding=padding)


def avgpool(input, kernel_size, strides, padding='VALID'):
    return tf.nn.avg_pool(input, ksize=[1, *kernel_size, 1], strides=[1, *strides, 1], padding=padding)


def dense(input, output_size, use_relu=True, name=None):
    input_size = int(input.shape[1])
    if name is None:
        name = "weights{}-{}".format(input_size, output_size)
    weights = tf.get_variable(name,
                              shape=[input_size, output_size],
                              initializer=tf.contrib.layers.xavier_initializer())

    bias = tf.Variable(tf.zeros(output_size))
    result = tf.add(tf.matmul(input, weights), bias)
    if use_relu:
        result = tf.nn.relu(result)
    return result


def _iterate_over_batches(input,
                          output,
                          weights,
                          dense_keep,
                          conv_keep,
                          X_data,
                          y_data,
                          class_weights=None,
                          dense_keep_probability=1.,
                          conv_keep_probability=1.,
                          batch_size=256,
                          progress_bar=False):
    num_classes = np.unique(y_data).shape[0]

    batch_count = int(math.ceil(len(X_data) / batch_size))
    if progress_bar:
        batches_pbar = tqdm(range(batch_count), unit='batches')
    else:
        batches_pbar = range(batch_count)

    for batch in batches_pbar:
        batch_start = batch * batch_size
        batch_x, batch_y = X_data[batch_start:batch_start + batch_size], y_data[batch_start:batch_start + batch_size]

        batch_weights = np.ones(len(batch_x))
        if class_weights is not None:
            batch_weights *= class_weights[batch_y]
        yield {
                  input: batch_x,
                  output: _to_one_hot(batch_y, num_classes),
                  weights: batch_weights,
                  dense_keep: dense_keep_probability,
                  conv_keep: conv_keep_probability
              }, \
              len(batch_x)


def _to_one_hot(labels, depth):
    result = np.zeros(shape=(len(labels), depth))
    result[np.arange(len(labels)), labels] = 1
    return result.astype(np.int8)


def evaluate(input,
             output,
             weights,
             operation,
             dense_keep,
             conv_keep,
             X_data,
             y_data,
             class_weights=None,
             dense_keep_probability=1.,
             conv_keep_probability=1.,
             batch_size=256):
    accuracies_and_batch_sizes = list(
        do_epoch(
            tf.get_default_session(),
            _iterate_over_batches(input, output, weights,
                                  dense_keep, conv_keep,
                                  X_data, y_data,
                                  class_weights=class_weights,
                                  dense_keep_probability=dense_keep_probability,
                                  conv_keep_probability=conv_keep_probability,
                                  batch_size=batch_size),
            operation
        ))

    accuracies = np.array(list(map(itemgetter(0), accuracies_and_batch_sizes)))
    batch_sizes = np.array(list(map(itemgetter(1), accuracies_and_batch_sizes)))
    return np.dot(accuracies, batch_sizes) / len(X_data)


def do_epoch(session, iterable, *operation):
    for i in iterable:
        yield session.run(*operation, feed_dict={**(i[0])}), i[1]


def train(session, input, output, weights, dense_keep, conv_keep, training,
          loss, X_train, y_train, X_valid, y_valid, accuracy,
          train_class_weights, valid_class_weights, dense_dropout, conv_dropout,
          early_stopper, epochs=200, batch_size=256):
    with session.as_default():
        for i in range(epochs):
            X_train, y_train = shuffle(X_train, y_train)

            results = list(do_epoch(session,
                                    _iterate_over_batches(input, output, weights,
                                                          dense_keep, conv_keep,
                                                          X_train, y_train, train_class_weights,
                                                          dense_keep_probability=dense_dropout,
                                                          conv_keep_probability=conv_dropout,
                                                          batch_size=batch_size),
                                    [training,
                                     loss,
                                     accuracy]))
            evaluations = list(map(itemgetter(0), results))
            losses = np.array(list(map(itemgetter(1), evaluations)))
            accuracies = np.array(list(map(itemgetter(2), evaluations)))
            batch_sizes = np.array(list(map(itemgetter(1), results)))

            train_loss = np.dot(losses, batch_sizes) / np.sum(batch_sizes)
            train_accuracy = np.dot(accuracies, batch_sizes) / np.sum(batch_sizes)

            validation_accuracy = evaluate(input, output, weights,
                                           accuracy, dense_keep, conv_keep,
                                           X_valid, y_valid, batch_size=batch_size)

            validation_loss = evaluate(input, output, weights,
                                       loss, dense_keep, conv_keep,
                                       X_valid, y_valid,
                                       class_weights=valid_class_weights, batch_size=batch_size)

            print(
                "Epoch {} train loss={:.3f}, acc.={:.3f}\tvalid loss = {:.3f}, acc. = {:.3f}".format(
                    i + 1, train_loss, train_accuracy, validation_loss, validation_accuracy))

            if early_stopper.should_stop(validation_loss):
                break
    return early_stopper._best_path


class EarlyStopper:
    def __init__(self, saver, session, which=min, patience=5, model_description=None, best_only=True):
        self._saver = saver
        self._session = session
        self._method = which
        self._last_value = None
        self._current_epoch = 0
        self._best_epoch = 0
        self._patience = patience
        self._best_path = None
        self._best_only = best_only

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

            if self._best_only:
                self._best_path = self._basename
            else:
                self._best_path = '{}-{:.3f}-{}'.format(self._basename, value, self._current_epoch)

            self._saver.save(self._session, self._best_path)
        elif self._current_epoch - self._best_epoch > self._patience:
            stop = True
        return stop



def _get_default_model_name():
    from datetime import datetime
    now = str(datetime.now())
    return 'model-{}'.format(now)


def _create_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

