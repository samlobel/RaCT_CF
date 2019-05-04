import sys
import os
UTILS_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'utils')
sys.path.insert(1, UTILS_DIR)

from data_loaders import load_train_data, load_tr_te_data, get_batch_from_list, get_num_items

from data_loaders import train_dataset, tr_te_dataset

import tensorflow as tf

import os

# from models import MultiVAE, MultiDAE, WeightedMatrixFactorization, WarpEncoder, WMFVAE
from models import MultiVAE, ProperlyShapedMultiDAE, WeightedMatrixFactorization, WarpEncoder, VWMF, GaussianVAE

import numpy as np

import json

from evaluation_functions import NDCG_binary_at_k_batch, Recall_at_k_batch, average_precision_at_k_batch


def get_test_data(data_subdir):
    data_dir = os.path.join('.', 'data', data_subdir)
    pro_dir = os.path.join(data_dir, 'pro_sg')

    n_items = get_num_items(pro_dir)

    test_tr, test_te = load_tr_te_data(
        os.path.join(pro_dir, 'test_tr.csv'), os.path.join(pro_dir, 'test_te.csv'), n_items)

    return {
        'n_items': n_items,
        'test_tr': test_tr,
        'test_te': test_te,
    }


def test_on_specific_model(model_path, only_actor=False, model_type=MultiVAE):
    """This is a function I'll feel okay messing aronud with. I'll generally use it
    to get data from the validation dataset when I need to."""

    batch_size = 500

    data_dict = get_test_data('ml-20m')
    # data_dict = get_test_data('msd')

    n_items = data_dict['n_items']
    test_tr = data_dict['test_tr']
    test_te = data_dict['test_te']

    testing_dataset = tr_te_dataset(test_tr, test_te, batch_size)

    data_iterator = tf.data.Iterator.from_structure(testing_dataset.output_types,
                                                    testing_dataset.output_shapes)

    batch_of_users, heldout_batch = data_iterator.get_next()

    testing_init_op = data_iterator.make_initializer(testing_dataset)

    model = model_type(batch_of_users, heldout_batch, input_dim=n_items)
    

    with tf.Session() as sess:
        if only_actor:
            model.actor_saver.restore(sess, '{}/model'.format(model_path))
            init = tf.variables_initializer(model.non_actor_restore_variables)
            sess.run(init)
        else:
            model.saver.restore(sess, '{}/model'.format(model_path))
        sess.run(testing_init_op)

        ndcg200_list = []
        ndcg100_list = []
        ndcg50_list = []
        ndcg20_list = []
        ndcg5_list = []
        ndcg3_list = []
        ndcg1_list = []

        items_per_user = []

        try:
            # for bnum, st_idx in enumerate(range(0, N_test, batch_size_test)):
            bnum = 0
            while True:
                bnum += 1
                print('test batch {}.'.format(bnum))

                feed_dict = {
                    model.keep_prob_ph: 1.0,
                    model.stddev_effect_on_latent_dim_scaler: 0.0,
                    model.kl_loss_scaler: 0.0,
                }

                pred_val, batch_of_users, heldout_batch, _ = sess.run(
                    [model.prediction, model.batch_of_users, model.heldout_batch, model.vad_true_ndcg],
                    feed_dict=feed_dict)

                ndcg200_list.append(NDCG_binary_at_k_batch(pred_val, heldout_batch, k=200, input_batch=batch_of_users))
                ndcg100_list.append(NDCG_binary_at_k_batch(pred_val, heldout_batch, k=100, input_batch=batch_of_users))
                ndcg50_list.append(NDCG_binary_at_k_batch(pred_val, heldout_batch, k=50, input_batch=batch_of_users))
                ndcg20_list.append(NDCG_binary_at_k_batch(pred_val, heldout_batch, k=20, input_batch=batch_of_users))
                ndcg5_list.append(NDCG_binary_at_k_batch(pred_val, heldout_batch, k=5, input_batch=batch_of_users))
                ndcg3_list.append(NDCG_binary_at_k_batch(pred_val, heldout_batch, k=3, input_batch=batch_of_users))
                ndcg1_list.append(NDCG_binary_at_k_batch(pred_val, heldout_batch, k=1, input_batch=batch_of_users))

                all_users = batch_of_users + heldout_batch
                items_per_single_user = np.sum(all_users, axis=1)

                items_per_user.append(items_per_single_user)

        except tf.errors.OutOfRangeError:
            print("Testing done. That broke it out of the loop.")
        
        print("Break!")

        print("NDCG")

        ndcg200_list = np.concatenate(ndcg200_list)
        ndcg100_list = np.concatenate(ndcg100_list)
        ndcg50_list = np.concatenate(ndcg50_list)
        ndcg20_list = np.concatenate(ndcg20_list)
        ndcg5_list = np.concatenate(ndcg5_list)
        ndcg3_list = np.concatenate(ndcg3_list)
        ndcg1_list = np.concatenate(ndcg1_list)


        ndcg200 = ndcg200_list.mean()
        ndcg100 = ndcg100_list.mean()
        ndcg50 = ndcg50_list.mean()
        ndcg20 = ndcg20_list.mean()
        ndcg5 = ndcg5_list.mean()
        ndcg3 = ndcg3_list.mean()
        ndcg1 = ndcg1_list.mean()

        print("USED MODEL: {}".format(model_path))

        print("Test NDCG@200=%.5f (%.5f)" % (np.mean(ndcg200_list),
                                            np.std(ndcg200_list) / np.sqrt(len(ndcg200_list))))
        print("Test NDCG@100=%.5f (%.5f)" % (np.mean(ndcg100_list),
                                            np.std(ndcg100_list) / np.sqrt(len(ndcg100_list))))
        print("Test NDCG@50=%.5f (%.5f)" % (np.mean(ndcg50_list),
                                            np.std(ndcg50_list) / np.sqrt(len(ndcg50_list))))
        print("Test NDCG@20=%.5f (%.5f)" % (np.mean(ndcg20_list),
                                            np.std(ndcg20_list) / np.sqrt(len(ndcg20_list))))
        print("Test NDCG@5=%.5f (%.5f)" % (np.mean(ndcg5_list),
                                            np.std(ndcg5_list) / np.sqrt(len(ndcg5_list))))
        print("Test NDCG@3=%.5f (%.5f)" % (np.mean(ndcg3_list),
                                            np.std(ndcg3_list) / np.sqrt(len(ndcg3_list))))
        print("Test NDCG@1=%.5f (%.5f)" % (np.mean(ndcg1_list),
                                            np.std(ndcg1_list) / np.sqrt(len(ndcg1_list))))

if __name__ == '__main__':
    # TODO: This should beuse CLI.
    test_on_specific_model("200_epochs_HIS_KL_annealing", only_actor=True, model_type=MultiVAE)
