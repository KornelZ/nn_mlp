import pandas as pd
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import itertools

CLASSIFICATION = 0
REGRESSION = 1
TEST_PATH = "data/Classification/data.three_gauss.test.10000.csv"
TRAIN_PATH = "data/Classification/data.three_gauss.train.10000.csv"
TRAIN_SIZE_PERCENT = 0.75
LEARNING_RATE = 0.001
MOMENTUM = 0.2
EPOCHS = 1000
BATCH_SIZE = 32
MODE = CLASSIFICATION
ACTIVATION = tf.nn.relu
LAYERS = [700, 500, 200]
RANDOM_SEED = None

def get_file_name():
    dataset = TRAIN_PATH.split("/")
    dataset = dataset[-1][:-4]
    if MODE == CLASSIFICATION:
        params = "cls_"
    else:
        params = "reg_"

    act = { tf.nn.relu:"relu", tf.nn.sigmoid:"sigmoid", tf.nn.softmax:"softmax", tf.nn.tanh:"tanh"}
    if ACTIVATION in act:
        fun = act[ACTIVATION]
    else:
        fun = "unknown"
    params += "lr-" + str(LEARNING_RATE) + "_mom-" + str(MOMENTUM) + "_epochs-" + str(EPOCHS) \
        + "_batch-" + str(BATCH_SIZE) + "_act-" + fun

    for layer in LAYERS:
        params += "_l-" + str(layer)
    return dataset + "_" + params + ".png"

def plot_loss(train_loss, validation_loss):
    fig, ax = plt.subplots()
    ax.plot(np.arange(EPOCHS), train_loss, color='black',
            label='train loss')
    ax.plot(np.arange(EPOCHS), validation_loss, color='red', linestyle='dashed',
            label='valid loss')
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Error")
    ax.legend()
    plt.show()
    fig.savefig("loss_" + get_file_name())

def plot_regression(test_data, test_labels, predictions, test_loss):
    fig, ax = plt.subplots()
    ax.plot(test_data, predictions, color='red', linewidth=2,
            label='predictions')
    ax.plot(test_data, test_labels, color='green', linewidth=1,
            label='function')
    ax.set_xlabel("X  loss: " + "{0:.4f}".format(test_loss))
    ax.set_ylabel("Y")
    ax.legend()
    plt.show()
    fig.savefig("test_result_" + get_file_name())


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
    if ACTIVATION == tf.nn.relu:
        return data
    return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))

def denorm(data, non_norm_data):
    min = np.min(non_norm_data, axis=0)
    max = np.max(non_norm_data, axis=0)
    return data * (max - min) + min

def norm_labels(labels, num_classes):
    """only for classification"""
    if MODE == CLASSIFICATION:
        return (np.arange(num_classes) + 1 == labels).astype(np.float32)
    return labels


def batch_data(data, labels):
    """
    Divide train data in batches made of 100 samples randomly
    """
    rand_indx = random.randint(0, 224)
    size = rand_indx * BATCH_SIZE % (data.shape[0] - BATCH_SIZE)
    return data[size:size+BATCH_SIZE], labels[size:size+BATCH_SIZE]


def main():
    df = read_csv(TRAIN_PATH)
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)
    print("Dataframe size", df.shape)
    train, valid = divide_dataset(df)
    print("Train size {}, Validation size {}".format(train.shape, valid.shape))
    num_classes = np.unique(train[:, -1:]).shape[0]
    num_inputs = train.shape[1] - 1
    num_outputs = num_classes if MODE == CLASSIFICATION else 1
    print("Number of classes", num_classes)
    print("Min {}, Max {}".format(np.min(train[:, :-1]), np.max(train[:, :-1])))
    train_data = norm(train[:, :-1])
    print("After norm Min {}, Max {}".format(np.min(train_data), np.max(train_data)))
    train_labels = norm_labels(train[:, -1:], num_classes)
    print("Train label not encoded", train[0, -1:])
    print("Train label encoded", train_labels[0])
    valid_data = norm(valid[:, :-1])
    valid_labels = norm_labels(valid[:, -1:], num_classes)
    if TEST_PATH is not None:
        test = read_csv(TEST_PATH).values
        test_data = norm(test[:, :-1])
        test_labels = norm_labels(test[:, -1:], num_classes)

    def get_weights_and_biases():
        w = []
        b = []
        for i in range(len(LAYERS) - 1):
            w.append(tf.Variable(
                tf.truncated_normal([LAYERS[i], LAYERS[i + 1]], stddev=0.1)
            ))
            b.append(tf.Variable(tf.zeros([LAYERS[i + 1]])))
        return w, b

    def model(w, b, x):
        y = x
        for i in range(len(LAYERS) - 2):
            y = ACTIVATION(tf.matmul(y, w[i]) + b[i])
        y = tf.matmul(y, w[-1]) + b[-1]
        if MODE == CLASSIFICATION:
            return tf.nn.softmax(y)
        else:
            return y
    # input reshaped
    X = tf.placeholder(tf.float32, [None, num_inputs])
    # weights
    LAYERS.insert(0, num_inputs)
    LAYERS.append(num_outputs)
    XX = tf.reshape(X, [-1, num_inputs])

    # correct answers will go here
    Y_ = tf.placeholder(tf.float32, [None, num_outputs])
    # The model
    weights, biases = get_weights_and_biases()
    Y = model(weights, biases, XX)

    def cross_entropy():
        return tf.reduce_mean(
            -tf.reduce_sum(Y_ * tf.log(Y) * 100.0, reduction_indices=[1]))

    def mean_squared_error():
        return tf.reduce_mean(tf.abs(Y - Y_))

    if MODE == CLASSIFICATION:
        loss = cross_entropy()
    else:
        loss = mean_squared_error()
    # maybe for visualisation
    # is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    # accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

    optimizer = tf.train.MomentumOptimizer(LEARNING_RATE, MOMENTUM)
    train_step = optimizer.minimize(loss)

    init = tf.initialize_all_variables()

    def classification_session():
        with tf.Session() as sees:
            sees.run(init)
            for i in range(EPOCHS):
                train_batch, train_batch_labels = batch_data(train_data, train_labels)
                sees.run(train_step, feed_dict={X: train_batch, Y_: train_batch_labels})
                correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                print("Validation accuracy:", sees.run(accuracy, feed_dict={
                    X: valid_data, Y_: valid_labels}))
            if TEST_PATH is not None:
                correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                print("Test accuracy:", sees.run(accuracy, feed_dict={
                    X: test_data, Y_: test_labels}))

    def regression_session():
        train_loss_history = []
        valid_loss_history = []
        with tf.Session() as sees:
            sees.run(init)
            for i in range(EPOCHS):
                total_loss = 0
                batches = 0
                for offset in range(0, train_data.shape[0], BATCH_SIZE):
                    train_batch = train_data[offset:offset+BATCH_SIZE]
                    train_batch_labels = train_labels[offset:offset+BATCH_SIZE]
                    _, computed_loss = sees.run([train_step, loss], feed_dict={X: train_batch, Y_: train_batch_labels})
                    batches += 1
                    total_loss += computed_loss
                print("Training epoch: %d, loss: %f" % (i, total_loss / batches))
                pred_valid = sees.run(Y, feed_dict={X: valid_data})
                valid_loss = sees.run(tf.reduce_mean(tf.abs(pred_valid - valid_labels)))
                print("Validation loss:", valid_loss)
                train_loss_history.append(total_loss / batches)
                valid_loss_history.append(valid_loss)

            plot_loss(train_loss_history, valid_loss_history)
            if TEST_PATH is not None:
                pred_test = sees.run(Y, feed_dict={X: test_data})
                test_loss = sees.run(tf.reduce_mean(tf.abs(pred_test - test_labels)))
                print("Test loss", test_loss)
                plot_regression(test_data, test_labels, pred_test, test_loss)

    if MODE == CLASSIFICATION:
        classification_session()
    else:
        regression_session()


if __name__ == '__main__':
    main()


