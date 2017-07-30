from operator import is_not

import tensorflow as tf


class Layer:
    def __init__(self, output_size, inbound_layers=None):
        self._output_size = output_size
        self._outbound_layers = []

        nones_removed = list(filter(lambda x: x is not None, inbound_layers))

        if inbound_layers == nones_removed:
            self._inbound_layers = inbound_layers
        else:
            self._inbound_layers = []

        for layer in self._inbound_layers:
            layer._outbound_layers.append(self)

    @property
    def input_size(self):
        return sum(map(lambda layer: layer._output_size, self._inbound_layers))


class ConvolutionalLayer(Layer):
    def __init__(self,
                 kernel_size,
                 filters,
                 strides=None,
                 mean=0,
                 stddev=0.1,
                 activation=tf.nn.relu,
                 inbound_layer=None,
                 channels=None):
        super(ConvolutionalLayer, self).__init__(filters, [inbound_layer])

        if inbound_layer is None and channels is None:
            # First layer, channels not given
            raise ValueError('Either inbound_layers or channels must be different from None')
        elif channels is not None:
            input_size = channels  # First layer, channels given
        else:
            input_size = self.input_size  # Not first layer, get input size from inputs

        self._filters = tf.Variable(
            tf.random_normal(shape=[*kernel_size, input_size, filters], mean=mean, stddev=stddev))
        if strides is None:
            strides = [1, 1]
        self._strides = strides
        self._bias = tf.Variable(tf.zeros(filters))
        self._activation = activation

    def __call__(self, input):
        conv = tf.add(tf.nn.conv2d(input, self._filters, strides=[1, *self._strides, 1], padding='VALID'), self._bias)
        return self._activation(conv)


class MaxPool(Layer):
    def __init__(self, kernel_size, strides, inbound_layer):
        super(MaxPool, self).__init__(inbound_layer._output_size, [inbound_layer])
        self._kernel_size = kernel_size
        self._strides = strides

    def __call__(self, input):
        return tf.nn.max_pool(input, ksize=[1, *self._kernel_size, 1], strides=[1, *self._strides, 1], padding='VALID')


class Dense(Layer):
    def __init__(self,
                 output_size,
                 mean=0,
                 stddev=0.1,
                 activation=tf.nn.relu,
                 inbound_layer=None,
                 input_size=None):
        super(Dense, self).__init__(output_size, [inbound_layer])
        self._weights = tf.Variable(tf.truncated_normal(shape=[input_size, output_size], mean=mean, stddev=stddev))
        self._bias = tf.Variable(tf.zeros(output_size))
        self._activation = activation

    def __call__(self, input):
        return tf.nn.relu(tf.add(tf.matmul(input, self._weights), self._bias))


class Flatten(Layer):
    def __init__(self, inbound_layer=None):
        super(Flatten, self).__init__()

    def __call__(self, input):
        return tf.contrib.layers.flatten(input)

if __name__ == '__main__':
    x = tf.placeholder(tf.float32, shape=[None, *[32, 32, 1]])
    conv1 = ConvolutionalLayer(kernel_size=[3, 3], filters=16, channels=1)
    l1_maxpool = MaxPool(kernel_size=[2, 2], strides=[2, 2], inbound_layer=conv1)
    conv1_out = conv1(x)
    print(conv1_out.shape)
    l1_maxout = l1_maxpool(conv1_out)
    print(l1_maxout.shape)

    l2_depth = 16
    conv2 = ConvolutionalLayer(kernel_size=[2, 2], filters=l2_depth, inbound_layer=l1_maxpool)
    l2_maxpool = MaxPool(kernel_size=[2, 2], strides=[2, 2], inbound_layer=conv2)

    conv2_out = conv2(l1_maxout)
    l2_maxout = l2_maxpool(conv2_out)
    print(l2_maxout.shape)

    conv3 = ConvolutionalLayer(kernel_size=[2, 2], filters=10, inbound_layer=l2_maxpool)
    conv3_out = conv3(l2_maxout)
    print(conv3_out.shape)
