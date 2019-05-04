"""
This file compares our method against NCF on the smaller datasets. This uses the datasets from the
original NCF paper, as well as their evaluation method. The evaluation is a bit different -- instead
of comparing the scores of ALL items, it samples one "seen" item and 99 "unseen" item, and measures the
frequency of it being in the top 10.

NOTE: For this one, manually copy the data from this repository into `data/NCF_DATA`
https://github.com/hexiangnan/neural_collaborative_filtering/tree/master/Data

"""

import sys
import os
UTILS_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'utils')
sys.path.insert(1, UTILS_DIR)

import tensorflow as tf
import numpy as np

import scipy.sparse as sp
import numpy as np

import time


class NCFDataset(object):
    '''
    classdocs
    '''

    def __init__(self, path):
        '''
        Constructor
        '''
        try:
            self.trainMatrix = self.load_rating_file_as_matrix(path + ".train.rating")
            self.testRatings = self.load_rating_file_as_list(path + ".test.rating")
            self.testNegatives = self.load_negative_file(path + ".test.negative")
        except Exception as e:
            print("Maybe you didn't add the data? Manually copy data from "
                  "https://github.com/hexiangnan/neural_collaborative_filtering/tree/master/Data "
                  "into 'data/NCF_DATA`")
        assert len(self.testRatings) == len(self.testNegatives)
        
        self.num_users, self.num_items = self.trainMatrix.shape
        
    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList
    
    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1: ]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList
    
    def load_rating_file_as_matrix(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if (rating > 0):
                    mat[user, item] = 1.0
                line = f.readline()    
        return mat



his_dataset = NCFDataset('../data/NCF_DATA/ml-1m')

his_pinterests_dataset = NCFDataset('../data/NCF_DATA/pinterest-20')

# print(his_dataset.trainMatrix.shape)
# print(len(his_dataset.testRatings))
# print(len(his_dataset.testNegatives))


# First order of business: get a tf.Dataset working!

# train_data_csr = dataset.trainMatrix.tocsr()
# print(train_data)

def csr_to_tfdataset(dataset, batch_size):
    print('bing')
    data_tr = dataset.trainMatrix.tocsr().astype(np.float32)


    data_tr_coo = data_tr.tocoo()

    n_items = data_tr_coo.shape[1]

    indices = np.mat([data_tr_coo.row, data_tr_coo.col]).transpose()


    sorted_indices = indices.tolist()
    sorted_indices.sort()

    # print(sorted_indices)

    indices = np.matrixlib.defmatrix.matrix(sorted_indices, dtype=np.int32)



    # import ipdb; ipdb.set_trace()

    sparse_data = tf.SparseTensor(indices, data_tr_coo.data, data_tr_coo.shape)


    print('bang')
    # import ipdb; ipdb.set_trace()

    # import ipdb; ipdb.set_trace()

    samples_tr = tf.data.Dataset.from_tensor_slices(sparse_data)



    # stupid_rating_extraction = np.asarray([[a[1]] for a in dataset.testRatings], dtype=np.int32)
    # negative_extraction = np.asarray(dataset.testNegatives, dtype=np.int32)
    # stupid_rating_extraction = tf.data.Dataset.from_tensor_slices(stupid_rating_extraction)
    # negative_extraction = tf.data.Dataset.from_tensor_slices(negative_extraction)

    # samples_batched = samples_tr.batch(batch_size, drop_remainder=True)
    # samples_batched = samples_batched.map(tf.sparse_tensor_to_dense)

    # rating_batched = stupid_rating_extraction.batch(batch_size, drop_remainder=True)
    # negative_extraction_batched = negative_extraction.batch(batch_size, drop_remainder=True)

    # dataset = tf.data.Dataset.zip((samples_batched, rating_batched, negative_extraction_batched))
    # expected_shape = (tf.TensorShape([batch_size, n_items]), tf.TensorShape([batch_size,1]), tf.TensorShape([batch_size, 99]))
    # dataset = dataset.apply(tf.contrib.data.assert_element_shape(expected_shape))

    # return dataset




    # samples_tr = tf.data.Dataset.from_sparse_tensor_slices(sparse_data)

    print('malmstein')
    # print(samples_tr)


    stupid_rating_extraction = np.asarray([[a[1]] for a in dataset.testRatings], dtype=np.int32)
    negative_extraction = np.asarray(dataset.testNegatives, dtype=np.int32)

    print(stupid_rating_extraction.shape)
    print(negative_extraction.shape)

    stupid_rating_extraction = tf.data.Dataset.from_tensor_slices(stupid_rating_extraction)
    negative_extraction = tf.data.Dataset.from_tensor_slices(negative_extraction)
    
    
    dataset = tf.data.Dataset.zip((samples_tr, stupid_rating_extraction, negative_extraction))

    dataset = dataset.shuffle(10000).batch(batch_size, drop_remainder=True)


    # def check_out_thing(x):
    #     import ipdb; ipdb.set_trace()
    #     return tf.sparse_tensor_to_dense(x)

    dataset = dataset.map(lambda x, y, z: (tf.sparse_tensor_to_dense(x), y, z))
    # dataset = dataset.map(lambda x, y, z: (check_out_thing(x), y, z))


    #     dataset = samples_tr.shuffle(10000).batch(batch_size, drop_remainder=True)#.map(tf.sparse_to_dense)
    #     dataset = dataset.map(tf.sparse_tensor_to_dense)

    # expected_shape = tf.TensorShape([batch_size, n_items])
    # expected_shape = (tf.TensorShape([batch_size, n_items]), tf.TensorShape([batch_size,]), tf.TensorShape([batch_size, 99]))
    expected_shape = (tf.TensorShape([batch_size, n_items]), tf.TensorShape([batch_size,1]), tf.TensorShape([batch_size, 99]))
    dataset = dataset.apply(tf.contrib.data.assert_element_shape(expected_shape))
    
    return dataset

def dense_csr_to_tfdataset(dataset, batch_size):
    print('bing')
    data_tr = dataset.trainMatrix.tocsr().astype(np.float32)


    data_tr_coo = data_tr.tocoo()

    print("This step might be slow")
    data_tr_dense = data_tr.todense()
    print("How slow?")

    print("Mean density: {}".format(data_tr_dense.sum(axis=1).mean()))

    # import ipdb; ipdb.set_trace()

    n_items = data_tr_coo.shape[1]

    # data = tf.Tensor(data_tr_dense)

    # indices = np.mat([data_tr_coo.row, data_tr_coo.col]).transpose()
    # sparse_data = tf.SparseTensor(indices, data_tr_coo.data, data_tr_coo.shape)

    print('bang')

    # import ipdb; ipdb.set_trace()

    samples_tr = tf.data.Dataset.from_tensor_slices(data_tr_dense)

    # samples_tr = tf.data.Dataset.from_tensor_slices(sparse_data)

    stupid_rating_extraction = np.asarray([a[1] for a in dataset.testRatings], dtype=np.int32)
    negative_extraction = np.asarray(dataset.testNegatives, dtype=np.int32)

    print(stupid_rating_extraction.shape)
    print(negative_extraction.shape)

    stupid_rating_extraction = tf.data.Dataset.from_tensor_slices(stupid_rating_extraction)
    negative_extraction = tf.data.Dataset.from_tensor_slices(negative_extraction)
    
    
    dataset = tf.data.Dataset.zip((samples_tr, stupid_rating_extraction, negative_extraction))

    dataset = dataset.shuffle(10000).batch(batch_size, drop_remainder=True)
    # dataset = dataset.map(lambda x, y, z: (tf.sparse_tensor_to_dense(x), y, z))
    
    #     dataset = samples_tr.shuffle(10000).batch(batch_size, drop_remainder=True)#.map(tf.sparse_to_dense)
    #     dataset = dataset.map(tf.sparse_tensor_to_dense)

    # expected_shape = tf.TensorShape([batch_size, n_items])
    expected_shape = (tf.TensorShape([batch_size, n_items]), tf.TensorShape([batch_size,]), tf.TensorShape([batch_size, 99]))
    dataset = dataset.apply(tf.contrib.data.assert_element_shape(expected_shape))
    
    return dataset

# tf_dataset = csr_to_tfdataset(dataset, 100)

# print(tf_dataset)
# print(tf_dataset.output_types)


# qwer = np.asarray([len(qwer) for qwer in dataset.testNegatives], dtype=np.int32)
# zxcv= np.asarray(dataset.testNegatives)
# print(zxcv)


import sys

print(sys.path)
if ".." not in sys.path:
    sys.path.insert(0, "..")

from models import OldMultiDAE, MultiVAE, WarpEncoder, VariationalWarpEncoder

from evaluation_functions import Recall_at_k_batch, NDCG_binary_at_k_batch


class FunkyMagicMixin(object):

    # def _return_ndcg_given_args(self, our_outputs=None, true_outputs=None, input_batch=None):
    #     """
    #     In our case, true_outputs should be the remaining_input field.
    #     Our_outputs is the softmax output.
    #     input_batch is the ones that you wanna zero. That's important, because otherwise you see the
    #     predictions from the ones you knew about.

    #     But, it's still weird that it messed it up so grandly... not sure what happened there.
    #     """
    #     assert our_outputs is not None
    #     assert true_outputs is not None
    #     assert input_batch is not None

    #     print("Hello from the new _return_ndcg_given_args!!!")

    #     return tf.py_func(self._return_ndcg_given_args_TRUE, [our_outputs, true_outputs, input_batch], tf.float32)


    #     # return tf.py_func(NDCG_binary_at_k_batch, [our_outputs, true_outputs, 100, input_batch],
    #     #                   tf.float32)

    # def _get_good_indices(self, true_outputs, input_batch):
    #     # I want to choose randomly...

    #     # Assuming they're all zeros and ones...
    #     diff = true_outputs - input_batch


    #     num_goods = np.sum(diff, axis=0)


    #     with_ones_first = np.argsort(-diff, axis=-1)


    #     all_first_elements = with_ones_first[:,0]

    #     return all_first_elements


    # def _get_bad_indices(self, true_outputs, input_batch):
    #     with_zeros_first = np.argsort(true_outputs)

    #     first_bad_elements = with_zeros_first[:,0:99]

    #     return first_bad_elements

    # def _return_ndcg_given_args_TRUE(self, our_outputs=None, true_outputs=None, input_batch=None):
    #     # We need to get good_indices and bad_indices.

    #     good_indices = self._get_good_indices(true_outputs, input_batch)
    #     bad_indices = self._get_bad_indices(true_outputs, input_batch)

    #     return self._py_func_return_ndcg10_given_args(our_outputs, good_indices, bad_indices)

    # #     pass



    def funky_magic(self, outputs, good_indices, bad_indices):
        # Here, we're going to pass in the outputs, the good list of single indexes, and the list of bad indices.
        # We'll re-form this stuff, and then have the proper inputs to NDCG.
        # The "GOOD" target will ALWAYS go first.


        filtered_targets = [[1.0] + [0.0]*99 for _ in range(len(outputs))]

        filtered_outputs = []
        for output, index, b_is in zip(outputs, good_indices, bad_indices):
            filtered_output = []
            filtered_output.append(output[index])
            for bi in b_is:
                filtered_output.append(output[bi])
            filtered_outputs.append(filtered_output)

        filtered_targets = np.asarray(filtered_targets)
        filtered_outputs = np.asarray(filtered_outputs)
        
        assert filtered_targets.shape == filtered_outputs.shape

        # assert set(np.unique(filtered_targets)) == set([0.0, 1.0])

        return filtered_outputs, filtered_targets
    
    def _py_func_return_ndcg10_given_args(self, outputs, good_indices, bad_indices):
        filtered_outputs, filtered_targets  = self.funky_magic(outputs, good_indices, bad_indices)
        ndcg10 = NDCG_binary_at_k_batch(filtered_outputs, filtered_targets, k=10)
        return ndcg10

    def _py_func_return_recall10_given_args(self, outputs, good_indices, bad_indices):
        filtered_outputs, filtered_targets  = self.funky_magic(outputs, good_indices, bad_indices)
        recall10 = Recall_at_k_batch(filtered_outputs, filtered_targets, k=10)
        return recall10
    
    
    def _test_return_ndcg10_given_args(self, outputs, good_indices, bad_indices):
        return tf.py_func(self._py_func_return_ndcg10_given_args, [outputs, good_indices, bad_indices], tf.float32)
    
    def _test_return_recall10_given_args(self, outputs, good_indices, bad_indices):
        return tf.py_func(self._py_func_return_recall10_given_args, [outputs, good_indices, bad_indices], tf.float32)
        

    def create_validation_ops(self):
        """Where did I leave off? I figured out that I actually can just do it the regular way, until I need to
        construct the validation ops, which I'll do my way. That's actually great news.
        
        So, I need to overwrite create_validation_ops, which is the only place to use heldout_batch. And now,
        it should be easy.

        So, this function won't be so terrible now. I do need to call ndcg and recall to make them my targets, but
        that's not so bad. Also, I think I'll stick with using 100 as a metric, because we're filtering
        down either way. So it's never going to be the same.
        """

        vad_true_ndcg = self._test_return_ndcg10_given_args(self.prediction, self.good_indices, self.bad_indices)
        vad_true_recall = self._test_return_recall10_given_args(self.prediction, self.good_indices, self.bad_indices)

        self.vad_true_ndcg = vad_true_ndcg
        self.vad_true_recall = vad_true_recall





class NewMultiDAE(FunkyMagicMixin, OldMultiDAE):
    def __init__(
            self,
            batch_of_users,
            good_indices,
            bad_indices,
            input_dim=None,
            batch_size=100,
            evaluation_metric='NDCG',
            actor_reg_loss_scaler=1e-4,
            ac_reg_loss_scaler=0.0,
            lr_actor=1e-3,
            lr_critic=1e-4,
            lr_ac=2e-6,
            **kwargs):

        local_variables = locals()
        local_variables.pop('kwargs')
        self._set_locals(local_variables)

        self.build_graph()
        self.saver = tf.train.Saver()


class NewMultiVAE(FunkyMagicMixin, MultiVAE):
    def __init__(
            self,
            batch_of_users,
            good_indices,
            bad_indices,
            input_dim=None,
            batch_size=100,
            evaluation_metric='NDCG',
            actor_reg_loss_scaler=1e-4,
            ac_reg_loss_scaler=0.0,
            lr_actor=1e-3,
            lr_critic=1e-4,
            lr_ac=2e-6,
            **kwargs):

        local_variables = locals()
        local_variables.pop('kwargs')
        self._set_locals(local_variables)

        self.build_graph()
        self.saver = tf.train.Saver()


class NewVariationalWarpEncoder(FunkyMagicMixin, VariationalWarpEncoder):

    def __init__(
            self,
            batch_of_users,
            good_indices,
            bad_indices,
            input_dim=None,
            #  anneal_cap=0.2,
            #  epochs_to_anneal_over=50,
            # error_vector_limit=100,
            evaluation_metric='NDCG',
            batch_size=500,
            lr_actor=1e-3,
            lr_critic=1e-4,
            lr_ac=2e-6,
            # ac_reg_loss_scaler=1.0, # It's already scaled...
            ac_reg_loss_scaler=0.0,  #Just to be ...safe.
            actor_reg_loss_scaler=1e-4,
            **kwargs):
        """
        I'll do a better job about defining the inputs here.
        """
        local_variables = locals()
        local_variables.pop('kwargs')
        self._set_locals(local_variables)

        self.build_graph()
        self.saver = tf.train.Saver()

def _print(toprint, verbose):
    if verbose:
        print(toprint)


def calc_kl_scaler_by_batch(batch_num, min_kl, max_kl, batches_to_anneal_over):
    kl_scaler = (1.0 * batch_num) / batches_to_anneal_over
    kl_scaler = min(kl_scaler, max_kl)
    return kl_scaler


def train(
        small_dataset,
        n_items,
        constructor,
        actor_reg_loss_scaler=0.0,
        n_epochs_pred_only=0,
        n_epochs_ac_only=0,
        n_epochs_pred_and_ac=0,
        break_early=False,
        lr_actor=1e-3,
        lr_critic=1e-4,
        lr_ac=2e-6,
        batch_size=100,
        min_kl=0.0,
        max_kl=0.2,
        batches_to_anneal_over=200000,
        verbose=False):
    """Okay, where do we leave off? In a pretty, pretty good place.
    I just need to write out pretty much the same training function as
    before, but set up the data differently"""

    print('boom, top of train')

    np.random.seed(98765)
    tf.set_random_seed(98765)
    n_epochs = n_epochs_pred_only + n_epochs_ac_only + n_epochs_pred_and_ac

    tf.reset_default_graph()


    tf_dataset = csr_to_tfdataset(small_dataset, batch_size)
    # tf_dataset = dense_csr_to_tfdataset(small_dataset, batch_size)

    print(tf_dataset.output_types, tf_dataset.output_shapes)

    data_iterator = tf.data.Iterator.from_structure(tf_dataset.output_types,
                                                    tf_dataset.output_shapes)

    batch_of_users, good_indices, bad_indices = data_iterator.get_next()

    training_init_op = data_iterator.make_initializer(tf_dataset)


    # constructor = NewMultiDAE
    # constructor = NewMultiVAE

    
    model = constructor(batch_of_users, good_indices, bad_indices,
                input_dim=n_items,
                batch_size=batch_size,
                lr_actor=lr_actor,
                lr_critic=lr_critic,
                lr_ac=lr_ac,
                actor_reg_loss_scaler=actor_reg_loss_scaler,
    )

    print(model)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        best_score = -np.inf
        best_recall = -np.inf
        best_epoch = 0

        batches_seen = 0
        for epoch in range(n_epochs):
            # if log_critic_training_error:
            #     training_critic_error = []
            training_phase = None
            print("Starting epoch {}".format(epoch))
            try:
                sess.run(training_init_op)
                # print("initialized training.")
                mean_critic_errors = []
                mean_true_ndcg = []
                bnum = 0
                while True:
                    batches_seen += 1
                    bnum += 1
                    _print(bnum, verbose)
                    if break_early:
                        if bnum >= 25:
                            print('breaking early')
                            break


                    kl_scaler = calc_kl_scaler_by_batch(batches_seen, min_kl, max_kl,
                                                        batches_to_anneal_over)


                    feed_dict = {  #model.input_ph: X, #TODO: THESE WILL CHANGE...
                        model.keep_prob_ph: 0.5,
                        model.kl_loss_scaler: kl_scaler,
                        # model.kl_loss_scaler: calc_kl_scaler(epoch, min_kl, max_kl, epochs_to_anneal_over),
                        model.stddev_effect_on_latent_dim_scaler: 1.0,
                        model.train_batch_norm: False,  #Even though it's the default...
                        model.epoch: epoch,
                    }

                    to_run = {}

                    to_run['true_ndcg'] = model.true_ndcg
                    to_run['mean_critic_error'] = model.mean_critic_error
                    to_run['critic_regularization_loss'] = model.critic_regularization_loss
                    to_run['ac_input'] = model.ac_input
                    to_run['critic_output'] = model.critic_output

                    # if log_critic_training_error:
                    #     to_run['training_critic_error'] = model.critic_error
                    feed_dict[model.train_batch_norm] = True

                    t = time.time()
                    if epoch < n_epochs_pred_only:
                        # print("ACTOR ONLY")
                        training_phase = "ACTOR"
                        feed_dict[model.train_batch_norm] = True
                        to_run['_'] = model.actor_train_op
                        sess_return = sess.run(to_run, feed_dict=feed_dict)
                        _print(
                            "Time taken for n_epochs_pred_only batch: {}".format(time.time() - t),
                            verbose)
                        feed_dict[model.train_batch_norm] = False
                    elif n_epochs_pred_only <= epoch < (n_epochs_pred_only + n_epochs_ac_only):
                        # print("CRITIC ONLY")
                        training_phase = "CRITIC"
                        to_run['_'] = model.critic_train_op
                        sess_return = sess.run(to_run, feed_dict=feed_dict)
                        _print("Time taken for n_epochs_ac_only batch: {}".format(time.time() - t),
                               verbose)
                        # import ipdb; ipdb.set_trace()
                    else:
                        # print("BOTH TRAINING")
                        training_phase = "AC"
                        to_run['_'] = model.ac_train_op
                        to_run['__'] = model.critic_train_op
                        sess_return = sess.run(to_run, feed_dict=feed_dict)
                        _print("Time taken for n_epochs_pred_ac batch: {}".format(time.time() - t),
                               verbose)

                    # print("TRAINING NDCG: {}".format(sess_return['true_ndcg'].mean()))
                    # print("TRAINING Critic Error: {}".format(sess_return['mean_critic_error'].mean()))
                    # print("CRITIC REG LOSS: {}".format(sess_return['critic_regularization_loss'].mean()))
                    # print("CRITIC_OUTPUT: {}".format(sess_return['critic_output'].mean()))

                    mean_critic_errors.append(sess_return['mean_critic_error'])
                    mean_true_ndcg.append(sess_return['true_ndcg'])

            except tf.errors.OutOfRangeError:
                print("{} batches in total".format(bnum))
                actual_mean_critic_error = np.asarray(mean_critic_errors).mean()
                print("Mean Critic Error for Training: {}".format(actual_mean_critic_error))
                actual_mean_true_ndcg = np.asarray(mean_true_ndcg).mean()
                print("Mean True NDCG, calculated our way... {}".format(actual_mean_true_ndcg))
                pass
                # print("Epoch Training Done")

            try:
                sess.run(training_init_op)
                # print("initialized training for validation.")
                bnum = 0
                ndcgs = []
                recalls = []
                while True:
                    batches_seen += 1
                    bnum += 1
                    _print(bnum, verbose)
                    if break_early:
                        if bnum >= 25:
                            print('breaking early')
                            break


                    # kl_scaler = calc_kl_scaler_by_batch(batches_seen, min_kl, max_kl,
                    #                                     batches_to_anneal_over)


                    feed_dict = {  #model.input_ph: X, #TODO: THESE WILL CHANGE...
                        # model.keep_prob_ph: 0.5,
                        model.keep_prob_ph : 1.0,
                        # model.kl_loss_scaler: kl_scaler,
                        # model.kl_loss_scaler: calc_kl_scaler(epoch, min_kl, max_kl, epochs_to_anneal_over),
                        model.stddev_effect_on_latent_dim_scaler: 0.0,
                        # model.stddev_effect_on_latent_dim_scaler: 1.0,
                        model.train_batch_norm: False,  #Even though it's the default...
                        model.epoch: epoch,
                    }

                    to_run = {'vad_true_ndcg': model.vad_true_ndcg, 'vad_true_recall': model.vad_true_recall}

                    result = sess.run(to_run, feed_dict=feed_dict)

                    ndcgs.append(result['vad_true_ndcg'].mean())
                    recalls.append(result['vad_true_recall'].mean())



                    # print(result['vad_true_ndcg'].mean())

                    # if log_critic_training_error:
                    #     to_run['training_critic_error'] = model.critic_error

                    # to_run

                    # t = time.time()
                    # if epoch < n_epochs_pred_only:
                    #     training_phase = "ACTOR"
                    #     feed_dict[model.train_batch_norm] = True
                    #     to_run['_'] = model.actor_train_op
                    #     sess_return = sess.run(to_run, feed_dict=feed_dict)
                    #     _print(
                    #         "Time taken for n_epochs_pred_only batch: {}".format(time.time() - t),
                    #         verbose)
                    #     feed_dict[model.train_batch_norm] = False
                    # elif n_epochs_pred_only <= epoch < (n_epochs_pred_only + n_epochs_ac_only):
                    #     training_phase = "CRITIC"
                    #     to_run['_'] = model.critic_train_op
                    #     sess_return = sess.run(to_run, feed_dict=feed_dict)
                    #     _print("Time taken for n_epochs_ac_only batch: {}".format(time.time() - t),
                    #            verbose)
                    # else:
                    #     training_phase = "AC"
                    #     to_run['_'] = model.ac_train_op
                    #     to_run['__'] = model.critic_train_op
                    #     sess_return = sess.run(to_run, feed_dict=feed_dict)
                    #     _print("Time taken for n_epochs_pred_ac batch: {}".format(time.time() - t),
                    #            verbose)

            except tf.errors.OutOfRangeError:
                # print("Testing Done")
                mean_ndcg = np.asarray(ndcgs).mean()
                print("NDCG MEAN: {}".format(mean_ndcg))
                mean_recall = np.asarray(recalls).mean()
                print("recall MEAN: {}".format(mean_recall))
                if mean_ndcg > best_score:
                    best_epoch = epoch
                    best_score = mean_ndcg
                if mean_recall > best_recall:
                    best_recall = mean_recall

        print("All done! Best score achieved: {}".format(best_score))
        print("All done! Best recall achieved: {}".format(best_recall))
        print("Best NDCG score happended at epoch {}".format(best_epoch))

if __name__ == '__main__':

    train(
        his_dataset,
        3706,
        NewMultiDAE,
        batch_size=151,
        actor_reg_loss_scaler=2e-5,
        n_epochs_pred_only=50,
        n_epochs_ac_only=50,
        n_epochs_pred_and_ac=50,
        verbose=False,
        )

