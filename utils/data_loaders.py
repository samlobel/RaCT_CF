from scipy import sparse
import pandas as pd
import numpy as np
import os

import tensorflow as tf


def load_train_data(csv_file, n_items):
    tp = pd.read_csv(csv_file)
    n_users = tp['uid'].max() + 1

    rows, cols = tp['uid'], tp['sid']
    data = sparse.csr_matrix((np.ones_like(rows), (rows, cols)),
                             dtype='float64',
                             shape=(n_users, n_items))
    return data


def load_tr_te_data(csv_file_tr, csv_file_te, n_items):
    tp_tr = pd.read_csv(csv_file_tr)
    tp_te = pd.read_csv(csv_file_te)

    start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
    end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

    rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
    rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

    data_tr = sparse.csr_matrix((np.ones_like(rows_tr), (rows_tr, cols_tr)),
                                dtype='float64',
                                shape=(end_idx - start_idx + 1, n_items))
    data_te = sparse.csr_matrix((np.ones_like(rows_te), (rows_te, cols_te)),
                                dtype='float64',
                                shape=(end_idx - start_idx + 1, n_items))
    return data_tr, data_te


def tr_te_dataset(data_tr, data_te, batch_size):
    # https://www.tensorflow.org/performance/performance_guide makes me think that I'm doing
    # something wrong, because my GPU usage hovers near 0 usually. That's v disappointing. I hope
    # I can speed it up hugely...
    # This is going to take in the output of data_tr and data_te, and turn them into
    # things we can sample from.

    # The only worry I have is, I don't know exactly how to do the whole "masking" part in here..

    # The way it works is, load_train_data just loads in training data, while load_tr_te_data
    # has goal-vectors as well. These are the ones that you drop-out. So, this really should be fine.

    data_tr = data_tr.astype(np.float32)
    data_tr_coo = data_tr.tocoo()

    n_items = data_tr_coo.shape[1]

    indices = np.mat([data_tr_coo.row, data_tr_coo.col]).transpose()
    sparse_data_tr = tf.SparseTensor(indices, data_tr_coo.data, data_tr_coo.shape)

    data_te = data_te.astype(np.float32)
    data_te_coo = data_te.tocoo()

    indices = np.mat([data_te_coo.row, data_te_coo.col]).transpose()
    sparse_data_te = tf.SparseTensor(indices, data_te_coo.data, data_te_coo.shape)

    samples_tr = tf.data.Dataset.from_tensor_slices(sparse_data_tr)
    samples_te = tf.data.Dataset.from_tensor_slices(sparse_data_te)

    # 10000 might be too big to sample from... Not sure how that's supposed to work with batch anyways.
    dataset = tf.data.Dataset.zip((samples_tr, samples_te)).shuffle(10000).batch(
        batch_size, drop_remainder=True)

    dataset = dataset.map(lambda x, y: (tf.sparse_tensor_to_dense(x), tf.sparse_tensor_to_dense(y)))

    expected_shape = tf.TensorShape([batch_size, n_items])
    dataset = dataset.apply(tf.contrib.data.assert_element_shape((expected_shape, expected_shape)))

    # dataset = dataset.skip(15)

    return dataset
    # dataset = dataset.map()


def train_dataset(data_tr, batch_size):

    # Note: I'm going to do the most heinous of things: I'm going to add in a fake operation here,
    # so that it has the same form as the other guy.
    # That will let us swap them out.

    data_tr = data_tr.astype(np.float32)

    data_tr_coo = data_tr.tocoo()

    n_items = data_tr_coo.shape[1]

    indices = np.mat([data_tr_coo.row, data_tr_coo.col]).transpose()
    sparse_data = tf.SparseTensor(indices, data_tr_coo.data, data_tr_coo.shape)

    samples_tr = tf.data.Dataset.from_tensor_slices(sparse_data)


    dataset = samples_tr.shuffle(10000).batch(batch_size, drop_remainder=True)#.map(tf.sparse_to_dense)
    dataset = dataset.map(tf.sparse_tensor_to_dense)

    expected_shape = tf.TensorShape([batch_size, n_items])
    dataset = dataset.apply(tf.contrib.data.assert_element_shape(expected_shape))

    dataset = dataset.zip((dataset, dataset))
    # dataset.apply(tf.contrib.data.assert_element_shape([expected_shape, expected_shape]))

    # dataset = dataset.skip(200)

    return dataset


def get_batch_from_list(idxlist, batch_size, batch_num, data):
    disc_training_indices = idxlist[(batch_size * batch_num):(batch_size * (batch_num + 1))]
    X_train = data[disc_training_indices]
    if sparse.isspmatrix(X_train):
        X_train = X_train.toarray()
    X_train = X_train.astype('float32')
    return X_train


def get_num_items(pro_dir):
    unique_sid = list()
    with open(os.path.join(pro_dir, 'unique_sid.txt'), 'r') as f:
        for line in f:
            unique_sid.append(line.strip())

    n_items = len(unique_sid)
    print("n_items: {}".format(n_items))
    return n_items
