import struct
import pickle
import inspect
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import tensorflow as tf
import tensorflow.contrib.slim as slim
import logz
from imgaug import augmenters as iaa
from data_process import read_dataset, shuffle_dataset

def run_model(session, predict, loss, train_step, saver, images, labels, X, y,
              epochs=1, batch_size=64, print_every=100, is_test=False):
    if not is_test:
        # Configure output directory for logging
        logz.configure_output_dir('logs')

        # Log experimental parameters
        args = inspect.getargspec(main)[0] # Get the names and default values of a function's parameters.
        locals_ = locals() # Return a dictionary containing the current scope's local variables
        params = {k: locals_[k] if k in locals_ else None for k in args}
        logz.save_params(params)

    # have tensorflow compute accuracy
    correct_prediction = tf.equal(tf.argmax(predict, axis=1), tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # counter
    iter_cnt = 0
    iters_each_epoch = len(images)//batch_size - 1
    for e in range(epochs):
        # keep track of losses and accuracy
        correct = 0
        losses = []
        # make sure we iterate over the dataset once
        images, labels = shuffle_dataset(images, labels)
        for i in range(iters_each_epoch):
            current_iter = i+1
            
            batch_X, batch_y = images[current_iter*batch_size:(current_iter+1)*batch_size], labels[current_iter*batch_size:(current_iter+1)*batch_size]
            feed_dict = {X: batch_X, y: batch_y}
            
            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
            l, corr, _ = session.run([loss, correct_prediction, train_step],feed_dict=feed_dict)

            # aggregate performance stats
            losses.append(l*batch_size)
            correct += np.sum(corr)
            
            # print every now and then
            if (iter_cnt % print_every) == 0 and not is_test:
                logz.log_tabular("Iteration", iter_cnt)
                logz.log_tabular("minibatch_loss", l)
                logz.log_tabular("minibatch_accuracy", np.sum(corr)/batch_size)
                logz.dump_tabular()
                logz.pickle_tf_vars()

            iter_cnt += 1
        if is_test:
            total_correct = correct/len(images)
            total_loss = np.sum(losses)/len(images)
            print('acc:', total_correct)
            print('los:', total_loss)
        else:
            saver.save(session, 'checkpoints/mnist_plus', iter_cnt)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=bool, default=False)
    args = parser.parse_args()

    batch_size = 32
    images, labels, images_test, labels_test = read_dataset()

    X = tf.placeholder(tf.float32, [None, 28, 28, 3])
    y = tf.placeholder(tf.int64, [None, 14])

    with tf.variable_scope('classifier', reuse=tf.AUTO_REUSE) as scope:
        y_ = slim.conv2d(inputs=X, num_outputs=32, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        y_ = slim.pool(y_, kernel_size=2, pooling_type='MAX')
        y_ = slim.conv2d(y_, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        y_ = slim.pool(y_, kernel_size=2, pooling_type='MAX')
        y_ = slim.flatten(y_)
        y_ = tf.layers.dense(y_, 512, activation=tf.nn.relu)
        y_ = tf.layers.dense(y_, 512, activation=tf.nn.relu)
        y_pred = tf.layers.dense(y_, 14)
    
    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(y, logits=y_pred))
    train_step = tf.train.AdamOptimizer(5e-4).minimize(loss)

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
        if args.test:
            run_model(sess, y_pred, loss, train_step, saver, images_test, labels_test, X, y, 1, batch_size, 100, args.test)
        else:
            run_model(sess, y_pred, loss, train_step, saver, images, labels, X, y, 10, batch_size, 100, args.test)
            


if __name__ == '__main__':
    main()