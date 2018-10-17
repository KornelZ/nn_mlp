import pandas as pd
import numpy as np


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
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def one_hot_encode(labels, num_classes):
    """only for classification"""
    return (np.arange(num_classes) + 1 == labels).astype(np.float32)


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


if __name__ == '__main__':
    main()
