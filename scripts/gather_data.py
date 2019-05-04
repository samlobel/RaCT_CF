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

    # model = MultiVAE(batch_of_users, heldout_batch,  input_dim=n_items)
    model = model_type(batch_of_users, heldout_batch, input_dim=n_items)
    # model = MultiDAE(batch_of_users, heldout_batch, input_dim=n_items)
    # model = WeightedMatrixFactorization(batch_of_users, heldout_batch, input_dim=n_items)
    # model = WarpEncoder(batch_of_users, heldout_batch, input_dim=n_items)
    

    with tf.Session() as sess:
        if only_actor:
            model.actor_saver.restore(sess, '{}/model'.format(model_path))
            init = tf.variables_initializer(model.non_actor_restore_variables)
            sess.run(init)
        else:
            model.saver.restore(sess, '{}/model'.format(model_path))
        sess.run(testing_init_op)

        all_ndcgs = []
        all_ces = []
        all_critic_outputs = []



        ndcg100_list = []
        recall50_list = []
        recall20_list = []

        # ndcg100_list = []
        # ap100_list = []
        # recall100_list = []

        # ndcg20_list = []
        # ndcg5_list = []
        # ndcg200_list = []
        # ndcg2_list = []
        # ndcg1_list = []

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

                # ndcg100, ce, critic_output, batch_of_users, heldout_batch = sess.run([model.vad_true_ndcg, model.vad_actor_error, model.vad_critic_output, model.batch_of_users, model.heldout_batch], feed_dict=feed_dict)

                pred_val, batch_of_users, heldout_batch, _ = sess.run(
                    [model.prediction, model.batch_of_users, model.heldout_batch, model.vad_true_ndcg],
                    feed_dict=feed_dict)

                # import ipdb; ipdb.set_trace()

                # ndcg100_list.append(NDCG_binary_at_k_batch(pred_val, heldout_batch, k=100, input_batch=batch_of_users))

                recall50_list.append(Recall_at_k_batch(pred_val, heldout_batch, k=50, input_batch=batch_of_users))
                recall20_list.append(Recall_at_k_batch(pred_val, heldout_batch, k=20, input_batch=batch_of_users))
                ndcg100_list.append(NDCG_binary_at_k_batch(pred_val, heldout_batch, k=100, input_batch=batch_of_users))


                # ap100_list.append(average_precision_at_k_batch(pred_val, heldout_batch, k=100, input_batch=batch_of_users))
                # recall100_list.append(Recall_at_k_batch(pred_val, heldout_batch, k=100, input_batch=batch_of_users))

                # ndcg20_list.append(NDCG_binary_at_k_batch(pred_val, heldout_batch, k=20, input_batch=batch_of_users))
                # ndcg5_list.append(NDCG_binary_at_k_batch(pred_val, heldout_batch, k=5, input_batch=batch_of_users))
                # ndcg200_list.append(NDCG_binary_at_k_batch(pred_val, heldout_batch, k=200, input_batch=batch_of_users))
                # ndcg2_list.append(NDCG_binary_at_k_batch(pred_val, heldout_batch, k=2, input_batch=batch_of_users))
                # ndcg1_list.append(NDCG_binary_at_k_batch(pred_val, heldout_batch, k=1, input_batch=batch_of_users))
                

                all_users = batch_of_users + heldout_batch
                items_per_single_user = np.sum(all_users, axis=1)

                # items_per_user.append(items_per_single_user)

                # all_ndcgs.append(ndcg100)
                # all_ces.append(ce)
                # all_critic_outputs.append(critic_output)

                # ndcg100, ap100, recall100 = sess.run(
                #     [model.vad_true_ndcg, model.vad_true_ap, model.vad_true_recall],
                #     feed_dict=feed_dict)

                # ndcg100_list.append(ndcg100)
                # ap100_list.append(ap100)
                # recall100_list.append(recall100)

        except tf.errors.OutOfRangeError:
            print("Testing done. That broke it out of the loop.")
        
        print("Break!")

        print("NDCG")

        recall20_list = np.concatenate(recall20_list)
        recall50_list = np.concatenate(recall50_list)
        

        ndcg100_list = np.concatenate(ndcg100_list)
        # ap100_list = np.concatenate(ap100_list)
        # recall100_list = np.concatenate(recall100_list)

        # ndcg20_list = np.concatenate(ndcg20_list)
        # ndcg5_list = np.concatenate(ndcg5_list)
        # ndcg200_list = np.concatenate(ndcg200_list)        
        # ndcg2_list = np.concatenate(ndcg2_list)
        # ndcg1_list = np.concatenate(ndcg1_list)


        # items_per_user = np.concatenate(items_per_user)

        # print("Used model: {}".format(model_path))

        print("Test NDCG@100=%.5f (%.5f)" % (np.mean(ndcg100_list),
                                            np.std(ndcg100_list) / np.sqrt(len(ndcg100_list))))
        print("Test RECALL@50=%.5f (%.5f)" % (np.mean(recall50_list),
                                            np.std(recall50_list) / np.sqrt(len(recall50_list))))
        print("Test Recall@20=%.5f (%.5f)" % (np.mean(recall20_list),
                                            np.std(recall20_list) / np.sqrt(len(recall20_list))))
        
        user_data = {}

        user_data['all_ndcgs'] = np.concatenate(all_ndcgs).tolist()
        user_data['all_ces'] = np.concatenate(all_ces).tolist()
        user_data['all_critic_outputs'] = np.concatenate(all_critic_outputs).tolist()
        user_data['all_num_items_per_user'] = np.concatenate(items_per_user).tolist()

        print(user_data)


        # with open("paper_plots/correlation_between_ndcg_nll_and_criticout/data/gaussian_vae_data.json", "w") as f:
        #     f.write(json.dumps(user_data))


        # all_ndcgs = np.concatenate(ndcg100_list).tolist()

        # user_data['all_ndcg_100s'] = ndcg100_list.tolist()
        # user_data['all_ap_100s'] = ap100_list.tolist()
        # user_data['all_recall_100s'] = recall100_list.tolist()

        # user_data['all_ndcg_20s'] = ndcg20_list.tolist()
        # user_data['all_ndcg_5s'] = ndcg5_list.tolist()
        # user_data['all_ndcg_200s'] = ndcg200_list.tolist()
        # user_data['all_ndcg_2s'] = ndcg2_list.tolist()
        # user_data['all_ndcg_1s'] = ndcg1_list.tolist()

        # user_data['items_per_user'] = items_per_user.tolist()

        # print("Writing")
        # # with open("paper_plots/histogram_of_ndcg_before_and_after_critic/data/after_critic_data.json", "w") as f:
        # with open("paper_plots/correlation_between_different_evaluation_measures/data/before_critic_data_no_sampling.json", "w") as f:
        #     f.write(json.dumps(user_data))



        # print(np.concatenate(ndcg100_list).mean())
        # print(np.concatenate(recall50_list).mean())
        # print(np.concatenate(recall20_list).mean())
        # all_ndcgs = np.concatenate(all_ndcgs).tolist()
        # all_ces = np.concatenate(all_ces).tolist()
        # all_critic_outputs = np.concatenate(all_critic_outputs).tolist()

        # data = {
        #     'all_ndcgs' : all_ndcgs,
        #     'all_ces' : all_ces,
        #     'all_critic_outputs' : all_critic_outputs,
        # }

        # with open("paper_plots/correlation_between_ndcg_nll_and_criticout/data/data.json", "w") as f:
        #     f.write(json.dumps(data))

        # import ipdb ; ipdb.set_trace()
        # print('boom baby')





if __name__ == '__main__':
    # TODO: Make this CLI
    test_on_specific_model('200_epochs_HIS_KL_annealing', only_actor=True, model_type=MultiVAE)
