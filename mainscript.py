import os
import pickle
from functools import partial
from operator import itemgetter

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from augmentations import Rotator, HistogramEqualizer, Squeezer, NoiseAdder, ContrastNormalization, StandardScaling
from network_utils import conv2d, maxpool, dense, train, EarlyStopper, evaluate, do_epoch, _iterate_over_batches

EPOCHS = 50
RUNS = 5
batch_size = 256


def apply_feedforward_model(x, num_of_classes, dense_keep_prob, conv_keep_prob):
    l1_depth = 100
    l2_depth = 150
    l3_depth = 250
    dense1_out_size = 300

    conv1 = conv2d(x, kernel_size=[7, 7], num_of_filters=l1_depth)
    l1_maxout = tf.nn.dropout(conv1, conv_keep_prob)
    l1_maxout = maxpool(l1_maxout, kernel_size=[2, 2], strides=[2, 2])

    conv2 = conv2d(l1_maxout, kernel_size=[4, 4], num_of_filters=l2_depth)
    l2_maxout = tf.nn.dropout(conv2, conv_keep_prob)
    l2_maxout = maxpool(l2_maxout, kernel_size=[2, 2], strides=[2, 2])

    conv3 = conv2d(l2_maxout, kernel_size=[2, 2], num_of_filters=l3_depth)
    l3_maxout = tf.nn.dropout(conv3, conv_keep_prob)
    l3_maxout = maxpool(l3_maxout, kernel_size=[2, 2], strides=[2, 2])

    flattened = tf.contrib.layers.flatten(l3_maxout)

    dense1_out = dense(flattened, dense1_out_size)
    dense1_out = tf.nn.dropout(dense1_out, dense_keep_prob)

    output = dense(dense1_out, num_of_classes)

    print(conv1.shape)
    print(l1_maxout.shape)
    print(conv2.shape)
    print(l2_maxout.shape)
    print(conv3.shape)
    print(l3_maxout.shape)
    print(flattened.shape)
    print(dense1_out.shape)
    print(output.shape)
    return output


def get_data(path):
    with open(path, mode='rb') as f:
        return pickle.load(f)


def preprocess(data, preprocessor):
    X, y = data['features'], data['labels']
    X_grayscale = np.array(list(map(lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), X)))
    X_reshaped = X_grayscale.reshape(*X_grayscale.shape, -1)
    X = preprocessor(X_reshaped)
    return X, y


if __name__ == "__main__":
    seed = 43512
    np.random.seed(seed)
    tf.set_random_seed(seed)
    directory = 'data'


    def get_class_percents(y, keys):
        classes, counts = np.unique(y, return_counts=True)
        cls2percent = {cls: percent for cls, percent in zip(classes, 100 * counts / sum(counts))}
        return np.array([cls2percent[cls] for cls in keys])


    test_probabilities_over_runs = []
    test_accuracies = 0
    # preprocessings = [HistogramEqualizer, ContrastNormalization]
    preprocessings = [HistogramEqualizer]
    overall_runs = RUNS * len(preprocessings)

    for preprocessing in preprocessings:
        for run in range(RUNS):
            train_data = get_data(os.path.join(directory, 'train.p'))
            valid = get_data(os.path.join(directory, 'valid.p'))
            test = get_data(os.path.join(directory, 'test.p'))

            preprocessor = preprocessing()
            X_train, y_train = preprocess(train_data, preprocessor)
            X_valid, y_valid = preprocess(valid, preprocessor)
            X_test, y_test = preprocess(test, preprocessor)
            num_classes = np.unique(y_train).shape[0]

            instances, rows, columns, *_ = X_train.shape

            sign_names = pd.read_csv('signnames.csv')
            keys = sign_names.ClassId.values
            sign_names = sign_names.assign(train_percents=get_class_percents(y_train, keys))
            sign_names = sign_names.assign(valid_percents=get_class_percents(y_valid, keys))
            sign_names = sign_names.assign(test_percents=get_class_percents(y_test, keys))

            dense_dropout = 0.7
            conv_dropout = 0.7

            rotator = Rotator(columns=columns, rows=rows, prob_distr=partial(np.random.uniform, low=-20, high=20))
            X_train, y_train, _ = rotator(X_train, y_train, len(X_train))

            squeezer = Squeezer(columns, rows, prob_distr=partial(np.random.uniform, low=-0.1, high=0.1))
            X_train, y_train, _ = squeezer(X_train, y_train, len(X_train))

            # noise_adder = NoiseAdder(prob_distr=partial(np.random.normal, loc=0, scale=20))
            # noise_adder = NoiseAdder(prob_distr=partial(np.random.uniform, low=-10, high=10))
            # X_train, y_train, _ = noise_adder(X_train, y_train, 60000)

            print('Training data shape = {}'.format(X_train.shape))

            train_class_weights = get_class_percents(y_train, keys)
            train_class_weights = 1 / np.sqrt(train_class_weights)
            valid_class_weights = sign_names.valid_percents
            valid_class_weights = 1 / np.sqrt(valid_class_weights)

            tf.reset_default_graph()
            x = tf.placeholder(tf.float32, shape=[None, *X_train[0].shape])
            y = tf.one_hot(tf.placeholder(tf.int32, (None)), num_classes)
            w = tf.placeholder(tf.float32, shape=[None])
            dense_keep_prob = tf.placeholder(tf.float32)
            conv_keep_prob = tf.placeholder(tf.float32)

            logits = apply_feedforward_model(x, num_of_classes=num_classes,
                                             dense_keep_prob=dense_keep_prob,
                                             conv_keep_prob=conv_keep_prob)
            # weights = [var for var in tf.global_variables() if var.name.startswith('conv') or var.name.startswith('weights')]

            cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits, weights=w)
            mean_loss = tf.reduce_mean(cross_entropy)
            # regularizer = [tf.nn.l2_loss(var) for var in weights]
            # mean_loss = tf.reduce_mean(loss + 0.0001 * sum(regularizer))
            optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
            training = optimizer.minimize(mean_loss)

            correct_prediction = tf.equal(tf.argmax(logits, axis=1), tf.argmax(y, axis=1))
            accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            session = tf.Session()
            saver = tf.train.Saver()
            early_stopper = EarlyStopper(saver, session, which=min, patience=20, model_description='mirt')

            with session.as_default():
                session.run(tf.global_variables_initializer())

            path_to_model = train(session, x, y, w,
                                  dense_keep_prob,
                                  conv_keep_prob,
                                  training, mean_loss,
                                  X_train, y_train, X_valid, y_valid,
                                  accuracy_operation,
                                  train_class_weights=None, valid_class_weights=None,
                                  dense_dropout=dense_dropout, conv_dropout=conv_dropout,
                                  early_stopper=early_stopper, epochs=EPOCHS, batch_size=batch_size)

            tf.reset_default_graph()

            with session.as_default():
                saver.restore(session, path_to_model)
                test_accuracy = evaluate(x, y, w, accuracy_operation, dense_keep_prob, conv_keep_prob, X_test, y_test,
                                         batch_size=4096)

                print('test accuracy = {}'.format(test_accuracy))
                test_accuracies += test_accuracy

            probs = tf.nn.softmax(logits)
            with session.as_default():
                probabilities_and_batch_sizes = list(
                    do_epoch(
                        session,
                        _iterate_over_batches(x, y, w,
                                              dense_keep_prob, conv_keep_prob,
                                              X_test, y_test,
                                              batch_size=4096),
                        probs
                    ))

                probabilities = np.array(list(map(itemgetter(0), probabilities_and_batch_sizes)))
                probabilities = np.concatenate(probabilities, axis=0)
                test_probabilities_over_runs.append(probabilities)
                # predictions = np.argmax(probabilities, axis=1)
                # from sklearn import metrics
                # print(metrics.classification_report(y_test, predictions))
    print(test_accuracies / overall_runs)
    end_probabilities = sum(test_probabilities_over_runs) / overall_runs
    predictions = np.argmax(end_probabilities, axis=1)
    print(np.mean(predictions == y_test))


    # print(np.array(*probabilities).shape)
    # predictions = session.run(probs, feed_dict={x: X_test, y: y_test, })
