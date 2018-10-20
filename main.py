import pandas as pd
import numpy as np
import tensorflow as tf
import random


PATH = "data/Classification/data.three_gauss.train.10000.csv"
TRAIN_SIZE_PERCENT = 0.75


def read_csv(path):
    try:
        return pd.read_csv(path)
    except IOError as e:
        print(e)
        exit(-1)


def divide_dataset(dataset):
    d = dataset.values
    np.random.shuffle(d)
    size = int(d.shape[0] * TRAIN_SIZE_PERCENT)
    return d[:size, :], d[size:, :]


def norm(data):
    return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))


def one_hot_encode(labels, num_classes):
    """only for classification"""
    return (np.arange(num_classes) + 1 == labels).astype(np.float32)


def batch_data(data, labels):
    """
    Divide train data in batches made of 100 samples randomly
    """
    rand_indx = random.randint(0, 224)
    size = rand_indx * 100
    return data[size:size+100], labels[size:size+100]


def main():
    df = read_csv(PATH)
    print("Dataframe size", df.shape)
    train, valid = divide_dataset(df)
    print("Train size {}, Validation size {}".format(train.shape, valid.shape))
    num_classes = np.unique(train[:, -1:]).shape[0]
    print("Number of classes", num_classes)
    print("Min {}, Max {}".format(np.min(train[:, :-1]), np.max(train[:, :-1])))
    train_data = norm(train[:, :-1])
    print("After norm Min {}, Max {}".format(np.min(train_data), np.max(train_data)))
    train_labels = one_hot_encode(train[:, -1:], num_classes)
    print("Train label not encoded", train[0, -1:])
    print("Train label encoded", train_labels[0])
    valid_data = norm(valid[:, :-1])
    valid_labels = one_hot_encode(valid[:, -1:], num_classes)

    X = tf.placeholder(tf.float32, [None, 2])
    # weights
    W = tf.Variable(tf.truncated_normal([2, 700], stddev=0.1, name="W"))
    W1 = tf.Variable(tf.truncated_normal([700, 500], stddev=0.1, name="W1"))
    W2 = tf.Variable(tf.truncated_normal([500, 200], stddev=0.1, name="W2"))
    W3 = tf.Variable(tf.truncated_normal([200, 3], stddev=0.1, name="W3"))
    # biases
    b0 = tf.Variable(tf.zeros([700]))
    b1 = tf.Variable(tf.zeros([500]))
    b2 = tf.Variable(tf.zeros([200]))
    b3 = tf.Variable(tf.zeros([3]))
    # input reshaped
    XX = tf.reshape(X, [-1, 2])

    # correct answers will go here
    Y_ = tf.placeholder(tf.float32, [None, 3])
    # The model
    Y0 = tf.nn.relu(tf.matmul(XX, W) + b0)
    Y1 = tf.nn.relu(tf.matmul(Y0, W1) + b1)
    Y2 = tf.nn.relu(tf.matmul(Y1, W2) + b2)
    Y = tf.nn.softmax(tf.matmul(Y2, W3) + b3)

    cross_entropy = tf.reduce_mean(
        -tf.reduce_sum(Y_ * tf.log(Y) * 100.0, reduction_indices=[1]))

    # maybe for visualisation
    # is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    # accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

    optimizer = tf.train.GradientDescentOptimizer(0.001)
    train_step = optimizer.minimize(cross_entropy)

    init = tf.initialize_all_variables()

    with tf.Session() as sees:
        sees.run(init)
        for i in range(100):
            train_batch, train_batch_labels = batch_data(train_data, train_labels)
            sees.run(train_step, feed_dict={X: train_batch, Y_: train_batch_labels})
            correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print(sees.run(accuracy, feed_dict={
                X: valid_data, Y_: valid_labels}))


if __name__ == '__main__':
    main()
