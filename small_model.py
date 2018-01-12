import functools
import tensorflow as tf
from tensorflow.python.ops import init_ops
import numpy as np
from rcn import dynamic_rcn, GruRcnCell


VGG_MEAN = [103.939, 116.779, 123.68]

def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


class RcnVgg16:

    def __init__(self, data, seq_length, target, train_mode):
        self.data = data
        self.seq_length = seq_length
        self.target = target
        self.train_mode = train_mode
        self.data_dict = None
        self.var_dict = {}


    @lazy_property
    def prediction(self):
        """
        load variable from npy to build the VGG
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        """

        # Convert RGB to BGR
        red, green, blue = tf.unstack(self.data, axis=4)
        bgr = tf.stack([
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ], 4)

        self.conv1 = self.conv_layer(bgr, 3, 8, "conv1")
        self.pool1 = self.max_pool(self.conv1, 'pool1')

        self.conv2 = self.conv_layer(self.pool1, 8, 16, "conv2")
        self.pool2 = self.max_pool(self.conv2, 'pool2')

        self.conv3 = self.conv_layer(self.pool2, 16, 32, "conv3")
        self.pool3 = self.max_pool(self.conv3, 'pool3')

        self.rcn4 = self.rcn_layer(self.pool3, 32, 32, "rcn4")
        self.pool4 = self.max_pool(self.rcn4, 'pool4')

        self.rcn5 = self.rcn_layer(self.pool4, 32, 32, "rcn5")
        self.rcn5_lastframe = self.last_frame_layer(self.rcn5, "rcn5_lastframe")
        self.pool5 = self.max_single_pool(self.rcn5_lastframe, 'pool5')

        self.fc6 = self.fc_layer(self.pool5, 7*7*32, 512, "fc6")
        self.relu6 = tf.nn.relu(self.fc6)
        self.relu6 = tf.cond(self.train_mode, lambda: tf.nn.dropout(self.relu6, 0.5), lambda: self.relu6)

        self.fc_7 = self.fc_layer(self.relu6, 512, 101, "fc_7")

        self.prob = tf.nn.softmax(self.fc_7, name="prob")

        del self.data_dict
        return self.prob

    @lazy_property
    def error(self):
        number = tf.range(0, tf.shape(self.seq_length)[0])
        indexs = tf.stack([number, self.target], axis=1)
        self.cross_entropy = -tf.reduce_sum(tf.log(tf.gather_nd(self.prediction, indexs)))
        return self.cross_entropy

    @lazy_property
    def accuracy(self):
        correct_prediction = tf.equal(tf.cast(tf.argmax(self.prediction, 1), tf.int32), self.target)
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def last_frame_layer(self, bottom, name):
        number = tf.range(0, tf.shape(self.seq_length)[0])
        indexs = tf.stack([self.seq_length - 1, number], axis=1)
        return tf.gather_nd(bottom, indexs, name)

    def max_pool(self, bottom, name):
        with tf.variable_scope(name):
            def _inner_max_pool(bott):
                return tf.nn.max_pool(bott,
                                      ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1],
                                      padding='SAME',
                                      name=name)

            bottoms = tf.unstack(bottom, axis=0)
            output = tf.stack([_inner_max_pool(bott) for bott in bottoms], axis=0)

            return output

    def max_single_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)

            def _inner_conv(bott):
                conv = tf.nn.conv2d(bott, filt, [1, 1, 1, 1], padding='SAME')
                bias = tf.nn.bias_add(conv, conv_biases)
                relu = tf.nn.relu(bias)
                return relu

            bottoms = tf.unstack(bottom, axis=0)
            output = tf.stack([_inner_conv(bott) for bott in bottoms], axis=0)

            return output

    def conv_single_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def rcn_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            _, _, N, H, C = bottom.get_shape().as_list()
            input_size = (N, H, C)
            nb_filter = out_channels
            dict_name = name.replace("rcn", "conv")
            weight_initializers = {}
            if self.data_dict is not None and dict_name in self.data_dict:
                filters = self.data_dict[dict_name][0]
                biases = self.data_dict[dict_name][1]
                weight_initializers['WConv'] = init_ops.constant_initializer(filters)
                weight_initializers['c_biases'] = init_ops.constant_initializer(biases)
            cell = GruRcnCell(input_size, nb_filter, 3, [1, 1, 1, 1], "SAME", 3, weight_initializers=weight_initializers)
            output, _ = dynamic_rcn(cell, bottom, sequence_length=self.seq_length, dtype=tf.float32)
            return output

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        if self.data_dict is not None and name in self.data_dict:
            filters = self.get_var(self.data_dict[name][0], name + "_filters", False)
            biases = self.get_var(self.data_dict[name][1], name + "_biases", False)
        else:
            initial_filter = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.01)
            initial_bias = tf.ones([out_channels], dtype=tf.float32)
            filters = self.get_var(initial_filter, name + "_filters", True)
            biases = self.get_var(initial_bias, name + "_biases", True)
        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        if self.data_dict is not None and name in self.data_dict:
            weights = self.get_var(self.data_dict[name][0], name + "_weights", True)
            biases = self.get_var(self.data_dict[name][1], name + "_biases", True)
        else:
            initial_weight = tf.truncated_normal([in_size, out_size], 0.0, 0.01)
            weights = self.get_var(initial_weight, name + "_weights", True)
            initial_bias = tf.ones([out_size], dtype=tf.float32)
            biases = self.get_var(initial_bias, name + "_biases", True)
        return weights, biases

    def get_var(self, initial_value, var_name, trainable):
        if trainable:
            var = tf.Variable(initial_value, name=var_name)
        else:
            var = tf.constant(initial_value, dtype=tf.float32, name=var_name)
        return var

'''
    def get_var_count(self):
        count = 0
        for v in self.var_dict.values():
            count += functools.reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count
'''
