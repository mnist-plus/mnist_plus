import struct
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import tensorflow as tf
import tensorflow.contrib.slim as slim
from imgaug import augmenters as iaa

class Classifier:
    def __init__(self):
        self.X = tf.placeholder(tf.float32, [None, 28, 28, 3])

        with tf.variable_scope('classifier', reuse=tf.AUTO_REUSE) as scope:
            y_ = slim.conv2d(inputs=self.X, num_outputs=32, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
            y_ = slim.pool(y_, kernel_size=2, pooling_type='MAX')
            y_ = slim.conv2d(y_, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
            y_ = slim.pool(y_, kernel_size=2, pooling_type='MAX')
            y_ = slim.flatten(y_)
            y_ = tf.layers.dense(y_, 512, activation=tf.nn.relu)
            y_ = tf.layers.dense(y_, 512, activation=tf.nn.relu)
            self.y_pred = tf.layers.dense(y_, 14)

    def predict(self, imgs):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        saver = tf.train.Saver()
        with tf.Session(config=config) as sess:
            ckpt =  tf.train.get_checkpoint_state(os.path.dirname('checkpoints/mnist_plus'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('successfully restore')
            else:
                sess.run(tf.global_variables_initializer())
                print('no pre-trained model')

            y = sess.run(self.y_pred, feed_dict={self.X: imgs})

        y = np.argmax(y, axis=1)
        first, second, symbol = 0, 0, 0
        for i in range(len(y)):
            if y[i] >= 10:
                symbol = y[i]
                continue
            if symbol == 0:
                first = first * 10 + y[i]
            if symbol != 0:
                second = second * 10 + y[i]
        print('expression')
        print(first, symbol, second)
        if symbol == 10:
            return first + second
        elif symbol == 11:
            return first - second
        elif symbol == 12:
            return first * second
        elif symbol == 13:
            return first / second
        else:
            return "can't recognize"