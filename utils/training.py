import os
import shutil
import numpy as np
from scipy import sparse
import random
import time
from collections import defaultdict
import json

import matplotlib.pyplot as plt

import pandas as pd

import tensorflow as tf

np.random.seed(98765)
tf.set_random_seed(98765)

# from tensorflow.python import debug as tf_debug

from data_loaders import load_train_data, load_tr_te_data, get_num_items

from data_loaders import train_dataset, tr_te_dataset

from evaluation_functions import NDCG_binary_at_k_batch, Recall_at_k_batch, average_precision_at_k_batch

from models import (MultiVAE, WarpEncoder, WeightedMatrixFactorization, ProperlyShapedMultiDAE,
                    VWMF, GaussianVAE, MultiVAEWithPhase4WARP, LambdaRankEncoder, MultiVAEWithPhase4LambdaRank)


def write_plotting_data(arch_str, data_obj):
    plotting_dir = os.path.join('..', 'plotting_data', arch_str)
    if not os.path.isdir(plotting_dir):
        os.makedirs(plotting_dir)
    write_loc = os.path.join(plotting_dir, "data.json")
    json_data = json.dumps(data_obj, indent=4)
    print("Writing")
    with open(write_loc, "w") as f:
        f.write(json_data)
    print("Written")


def get_data(data_subdir):
    data_dir = os.path.join('..', 'data', data_subdir)
    pro_dir = os.path.join(data_dir, 'pro_sg')

    n_items = get_num_items(pro_dir)

    train_data = load_train_data(os.path.join(pro_dir, 'train.csv'), n_items)
    vad_data_tr, vad_data_te = load_tr_te_data(
        os.path.join(pro_dir, 'validation_tr.csv'), os.path.join(pro_dir, 'validation_te.csv'),
        n_items)
    return {
        'n_items': n_items,
        'train_data': train_data,
        'vad_data_tr': vad_data_tr,
        'vad_data_te': vad_data_te,
    }


def get_test_data(data_subdir):
    data_dir = os.path.join('..', 'data', data_subdir)
    pro_dir = os.path.join(data_dir, 'pro_sg')

    n_items = get_num_items(pro_dir)

    test_tr, test_te = load_tr_te_data(
        os.path.join(pro_dir, 'test_tr.csv'), os.path.join(pro_dir, 'test_te.csv'), n_items)

    return {
        'n_items': n_items,
        'test_tr': test_tr,
        'test_te': test_te,
    }


def make_arch_string(ordered_arg_names=[], args_to_ignore=[], **kwargs):
    starting_args = [str(kwargs.pop(arg_key)) for arg_key in ordered_arg_names]

    for arg_key in args_to_ignore:
        kwargs.pop(arg_key, None)

    remaining_tuples = kwargs.items()
    sorted_remaining_tuples = list(sorted(remaining_tuples))
    remaining_folder_names = ["{}:{}".format(k, v) for k, v in sorted_remaining_tuples]

    all_folders = starting_args + remaining_folder_names
    folder = os.path.join(*all_folders)

    return folder


def get_model(batch_of_users, heldout_batch, model_class=None, **kwargs):
    """
    At some point, this will do some intense importlib nonsense. For now, we'll just import MultiVAE.
    """
    # assert model_class in ['multi_vae', 'warp_encoder', 'wmf', 'new_multi_dae', 'variational_wmf', 'gaussian_vae']

    if model_class == 'multi_vae':
        constructor = MultiVAE
    elif model_class == 'warp_encoder':
        constructor = WarpEncoder
    elif model_class == 'wmf':
        constructor = WeightedMatrixFactorization
    elif model_class == 'new_multi_dae':
        constructor = ProperlyShapedMultiDAE
    elif model_class == 'variational_wmf':
        constructor = VWMF
    elif model_class == 'gaussian_vae':
        constructor = GaussianVAE
    elif model_class == 'phase_4_warp':
        constructor = MultiVAEWithPhase4WARP
    elif model_class == 'lambdarank_actor':
        constructor = LambdaRankEncoder
    elif model_class == 'phase_4_lambdarank':
        constructor = MultiVAEWithPhase4LambdaRank
    else:
        raise Exception("IllegalModelName: {}".format(model_class))
    model = constructor(batch_of_users, heldout_batch, **kwargs)
    return model


def make_summary_writer(arch_str):
    """
    Note that now the arch_str includes the data_subdir.
    NOTE: this must happen after the model is made.
    """
    log_dir = os.path.join('..', 'logging', arch_str)
    print("log directory: %s" % log_dir)
    if os.path.exists(log_dir):
        print("Existing logs found, removing to start anew")
        shutil.rmtree(log_dir)
    summary_writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())
    return summary_writer


def make_checkpoint_dir(arch_str):
    chkpt_dir = os.path.join('..', 'checkpoints', arch_str)
    if not os.path.isdir(chkpt_dir):
        os.makedirs(chkpt_dir)
    return chkpt_dir


def make_validation_logging():
    ndcg_vad_var = tf.Variable(0.0)
    critic_error_vad_var = tf.Variable(0.0)

    ndcg_vad_summary = tf.summary.scalar('ndcg_at_k_validation', ndcg_vad_var)

    critic_error_vad_summary = tf.summary.scalar('critic_error_validation', critic_error_vad_var)

    summary_vars = [ndcg_vad_summary, critic_error_vad_summary]

    validation_logging = tf.summary.merge(summary_vars)

    return {
        'ndcg_vad_var': ndcg_vad_var,
        'critic_error_vad_var': critic_error_vad_var,
        'validation_logging': validation_logging,

    }

def calc_kl_scaler_by_batch(batch_num, min_kl, max_kl, batches_to_anneal_over):
    kl_scaler = (1.0 * batch_num) / batches_to_anneal_over
    kl_scaler = min(kl_scaler, max_kl)
    return kl_scaler


def _print(toprint, verbose):
    if verbose:
        print(toprint)


def train(model_class=None,
          data_subdir='ml-20m',
          evaluation_metric='NDCG',
          n_epochs_pred_only=0,
          n_epochs_ac_only=0,
          n_epochs_pred_and_ac=0,
          n_epochs_second_pred=0,
          batch_size=500,
          break_early=False,
          batches_to_anneal_over=200000,
          min_kl=0.0,
          max_kl=0.2,
          epochs_to_anneal_over=50,
          logging_frequency=25,
          path_to_save_actor=None,
          path_to_save_last_actor=None,
          restore_trained_actor_path=None,
          verbose=False,
          actor_reg_loss_scaler=1e-4,
          ac_reg_loss_scaler=0.0,
          positive_weights=2.0,
          omit_num_seen_from_critic=False,
          omit_num_unseen_from_critic=False,
          log_critic_training_error=False,
          version_tag="",
          #   tf_random_seed=None
          ):

    # print("Just setting batches_to_anneal_over to his value.")

    # if tf_random_seed is not None:
    #     print("Setting tf random seed to something different, so we can get a mean/std")
    #     tf.set_random_seed(tf_random_seed)
    #     print('tf random seed set to {}'.format(tf_random_seed))

    assert data_subdir in ['ml-20m', 'netflix-prize', 'msd']
    assert model_class in ['multi_vae', 'warp_encoder', 'wmf', 'new_multi_dae', 'variational_wmf', 'gaussian_vae', 'phase_4_warp', 'lambdarank_actor', 'phase_4_lambdarank']

    train_args = locals()
    arch_str = make_arch_string(['data_subdir', 'model_class'], [
        'path_to_save_actor', 'path_to_save_last_actor', 'logging_frequency',
        'log_critic_training_error'
    ], **train_args)

    np.random.seed(98765)
    tf.set_random_seed(98765)
    n_epochs = n_epochs_pred_only + n_epochs_ac_only + n_epochs_pred_and_ac + n_epochs_second_pred

    tf.reset_default_graph()

    data_dict = get_data(data_subdir)
    n_items = data_dict['n_items']
    train_data = data_dict['train_data']
    vad_data_tr = data_dict['vad_data_tr']
    vad_data_te = data_dict['vad_data_te']

    training_dataset = train_dataset(train_data, batch_size)
    validation_dataset = tr_te_dataset(vad_data_tr, vad_data_te, batch_size)

    data_iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                                    training_dataset.output_shapes)

    batch_of_users, heldout_batch = data_iterator.get_next()

    training_init_op = data_iterator.make_initializer(training_dataset)
    validation_init_op = data_iterator.make_initializer(validation_dataset)

    train_args['input_dim'] = n_items
    model = get_model(batch_of_users, heldout_batch, **train_args)

    summary_writer = make_summary_writer(arch_str)
    chkpt_dir = make_checkpoint_dir(arch_str)

    # ndcg_vad_var, ap_vad_var, recall_vad_var, critic_error_vad_var, validation_logging = make_validation_logging(
    # )
    validation_ops = make_validation_logging()

    plot_data_obj = {}

    with tf.Session() as sess:
        if restore_trained_actor_path is not None:
            print("Restoring Actor!")
            model.actor_saver.restore(sess, '{}/model'.format(restore_trained_actor_path))
            # init = tf.variables_initializer(model.non_actor_restore_variables)
            non_actor_variables = set(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)).difference(model.actor_restore_variables)
            init = tf.variables_initializer(non_actor_variables)
            # init = tf.variables_initializer( tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES).difference())
            sess.run(init)
        else:
            print("Not Restoring Actor.")
            init = tf.global_variables_initializer()
            sess.run(init)
        best_score = -np.inf

        batches_seen = 0
        for epoch in range(n_epochs):
            if log_critic_training_error:
                training_critic_error = []
            training_phase = None
            print("Starting epoch {}".format(epoch))
            try:
                sess.run(training_init_op)
                print("initialized training.")
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

                    if log_critic_training_error:
                        to_run['training_critic_error'] = model.critic_error

                    t = time.time()
                    if epoch < n_epochs_pred_only:
                        training_phase = "ACTOR"
                        feed_dict[model.train_batch_norm] = True
                        to_run['_'] = model.actor_train_op
                        sess_return = sess.run(to_run, feed_dict=feed_dict)
                        _print(
                            "Time taken for n_epochs_pred_only batch: {}".format(time.time() - t),
                            verbose)
                        feed_dict[model.train_batch_norm] = False
                    elif n_epochs_pred_only <= epoch < (n_epochs_pred_only + n_epochs_ac_only):
                        training_phase = "CRITIC"
                        to_run['_'] = model.critic_train_op
                        sess_return = sess.run(to_run, feed_dict=feed_dict)
                        _print("Time taken for n_epochs_ac_only batch: {}".format(time.time() - t),
                               verbose)
                    elif (n_epochs_pred_only + n_epochs_ac_only) <= epoch < (n_epochs_pred_only + n_epochs_ac_only + n_epochs_pred_and_ac):
                        training_phase = "AC"
                        to_run['_'] = model.ac_train_op
                        to_run['__'] = model.critic_train_op
                        sess_return = sess.run(to_run, feed_dict=feed_dict)
                        _print("Time taken for n_epochs_pred_ac batch: {}".format(time.time() - t),
                               verbose)
                    else:
                        training_phase = "ALTERNATIVE_ERROR"
                        to_run['_'] = model.second_actor_train_op
                        sess_return = sess.run(to_run, feed_dict=feed_dict)
                        _print("Time taken for n_epochs_second_pred batch: {}".format(time.time() - t),
                               verbose)

                    if log_critic_training_error:
                        training_critic_error.append(sess_return['training_critic_error'])

                    if batches_seen % logging_frequency == 0:
                        t = time.time()
                        print("Logging for batch {}".format(bnum))
                        # print("Logging for batch {}. KL Scalar: {}.".format(bnum, kl_scaler))
                        summary_train = sess.run(model.all_summaries, feed_dict=feed_dict)
                        # summary_writer.add_summary(summary_train, global_step=epoch * num_batches_total + bnum)
                        summary_writer.add_summary(summary_train, global_step=batches_seen)
                        #At the end of an epoch I would like to see whats going on...
                        summary_writer.flush()
                        _print("Time to log: {}".format(time.time() - t), verbose)

            except tf.errors.OutOfRangeError:
                print("Epoch Training Done")

            print("On to validaiton...")
            try:
                sess.run(validation_init_op)
                bnum = 0
                # one_epoch_ndcg_vad, one_epoch_recall50_vad, one_epoch_recall20_vad, one_epoch_critic_error = [], [], [], []
                # one_epoch_ndcg_vad, one_epoch_critic_error = [], []
                one_epoch_ndcg_vad, one_epoch_score, one_epoch_critic_error = [], [], []

                # one_epoch_ndcg_vad = []
                # one_epoch_ap_vad = []
                # one_epoch_recall_vad = []
                # one_epoch_critic_error = []

                # one_epoch_ndcg_at_200_vad = []
                # one_epoch_ndcg_at_50_vad = []
                # one_epoch_ndcg_at_20_vad = []
                # one_epoch_ndcg_at_5_vad = []

                while True:
                    bnum += 1
                    _print(bnum, verbose)
                    if break_early:
                        if bnum >= 25:
                            print('breaking early')
                            break

                    feed_dict = {  # model.input_ph : X,
                        model.stddev_effect_on_latent_dim_scaler: 0.0,
                        model.train_batch_norm: False,  #Even though it's the default...
                        # model.remaining_input : next_element[1], #This is the important part!
                    }

                    t = time.time()


                    """Another way of doing it START"""
                    vad_true_evaluation_metric, vad_true_ndcg, vad_critic_error = sess.run(
                        [
                            model.vad_true_evaluation_metric, model.vad_true_ndcg, model.vad_critic_error
                        ],
                        feed_dict=feed_dict)
                    one_epoch_ndcg_vad.append(vad_true_ndcg)
                    one_epoch_score.append(vad_true_evaluation_metric)
                    one_epoch_critic_error.append(vad_critic_error)
                    """Another way of doing it END"""

                    # """Another way of doing it START"""
                    # vad_true_ndcg, vad_critic_error = sess.run(
                    #     [
                    #         model.vad_true_ndcg, model.vad_critic_error
                    #     ],
                    #     feed_dict=feed_dict)
                    # one_epoch_ndcg_vad.append(vad_true_ndcg)
                    # one_epoch_critic_error.append(vad_critic_error)
                    # """Another way of doing it END"""

                    """One way of doing it START"""
                    # pred_val, batch_of_users, heldout_batch, critic_error_vad = sess.run(
                    #     [
                    #         model.prediction, model.batch_of_users, model.heldout_batch,
                    #         model.vad_critic_error
                    #     ],
                    #     feed_dict=feed_dict)
                    # one_epoch_ndcg_vad.append(
                    #     NDCG_binary_at_k_batch(
                    #         pred_val, heldout_batch, k=100, input_batch=batch_of_users))
                    # one_epoch_critic_error.append(critic_error_vad)
                    """One way of doing it END"""


                    # one_epoch_recall50_vad.append(Recall_at_k_batch(pred_val, heldout_batch, k=50, input_batch=batch_of_users))
                    # one_epoch_recall20_vad.append(Recall_at_k_batch(pred_val, heldout_batch, k=20, input_batch=batch_of_users))

                    # vad_true_ndcg, vad_true_ap, vad_true_recall, vad_critic_error = \
                    #     sess.run([model.vad_true_ndcg, model.vad_true_ap, model.vad_true_recall, model.vad_critic_error], feed_dict=feed_dict)
                    #     'vad_true_ndcg': model.vad_true_ndcg,
                    #     'vad_true_ap': model.vad_true_ap,
                    #     'vad_true_recall': model.vad_true_recall,
                    #     'vad_critic_error': model.vad_critic_error,

                    #     'vad_true_ndcg_at_200': model.vad_true_ndcg_at_200,
                    #     'vad_true_ndcg_at_50': model.vad_true_ndcg_at_50,
                    #     'vad_true_ndcg_at_20': model.vad_true_ndcg_at_20,
                    #     'vad_true_ndcg_at_5': model.vad_true_ndcg_at_5,
                    # }, feed_dict=feed_dict)
                    # vad_true_ndcg, vad_true_ap, vad_true_recall, vad_critic_error = \
                    #     sess.run([model.vad_true_ndcg, model.vad_true_ap, model.vad_true_recall, model.vad_critic_error], feed_dict=feed_dict)

                    # one_epoch_ndcg_vad.append(vad_true_ndcg)
                    # one_epoch_ap_vad.append(vad_true_ap)
                    # one_epoch_recall_vad.append(vad_true_recall)
                    # one_epoch_critic_error.append(vad_critic_error)

                    # one_epoch_ndcg_vad.append(validation_results['vad_true_ndcg'])
                    # one_epoch_ap_vad.append(validation_results['vad_true_ap'])
                    # one_epoch_recall_vad.append(validation_results['vad_true_recall'])
                    # one_epoch_critic_error.append(validation_results['vad_critic_error'])

                    # one_epoch_ndcg_at_200_vad.append(validation_results['vad_true_ndcg_at_200'])
                    # one_epoch_ndcg_at_50_vad.append(validation_results['vad_true_ndcg_at_50'])
                    # one_epoch_ndcg_at_20_vad.append(validation_results['vad_true_ndcg_at_20'])
                    # one_epoch_ndcg_at_5_vad.append(validation_results['vad_true_ndcg_at_5'])

                    _print("Time to do one vad batch: {}".format(time.time() - t), verbose)

            except tf.errors.OutOfRangeError:
                print("Validation Done")

            one_epoch_score = np.concatenate(one_epoch_score).mean()
            one_epoch_ndcg_vad = np.concatenate(one_epoch_ndcg_vad).mean()
            # one_epoch_recall50_vad = np.concatenate(one_epoch_recall50_vad).mean()
            # one_epoch_recall20_vad = np.concatenate(one_epoch_recall20_vad).mean()

            one_epoch_critic_error = np.concatenate(one_epoch_critic_error).mean()

            if log_critic_training_error:
                average_training_critic_error = np.concatenate(training_critic_error).mean()
            # one_epoch_ap_vad = np.concatenate(one_epoch_ap_vad).mean()
            # one_epoch_recall_vad = np.concatenate(one_epoch_recall_vad).mean()

            # one_epoch_ndcg_at_200_vad = np.concatenate(one_epoch_ndcg_at_200_vad).mean()
            # one_epoch_ndcg_at_50_vad = np.concatenate(one_epoch_ndcg_at_50_vad).mean()
            # one_epoch_ndcg_at_20_vad = np.concatenate(one_epoch_ndcg_at_20_vad).mean()
            # one_epoch_ndcg_at_5_vad = np.concatenate(one_epoch_ndcg_at_5_vad).mean()

            if training_phase not in plot_data_obj:
                plot_data_obj[training_phase] = defaultdict(list)

            # This tolist is because np float32 isn't json-serializeable.
            plot_data_obj[training_phase]['ndcg'].append(one_epoch_ndcg_vad.tolist())

            # This is because I want to track the DCG for this one...
            plot_data_obj[training_phase]['evaluation_score'].append(one_epoch_score.tolist())


            # plot_data_obj[training_phase]['recall50'].append(one_epoch_recall50_vad.tolist())
            # plot_data_obj[training_phase]['recall20'].append(one_epoch_recall20_vad.tolist())
            # plot_data_obj[training_phase]['ap'].append(one_epoch_ap_vad.tolist())
            # plot_data_obj[training_phase]['recall'].append(one_epoch_recall_vad.tolist())
            plot_data_obj[training_phase]['validation_critic_error'].append(
                one_epoch_critic_error.tolist())
            if log_critic_training_error:
                plot_data_obj[training_phase]['training_critic_error'].append(
                    average_training_critic_error.tolist())

            # plot_data_obj[training_phase]['ndcg'].append(one_epoch_ndcg_vad.tolist())
            # plot_data_obj[training_phase]['ap'].append(one_epoch_ap_vad.tolist())
            # plot_data_obj[training_phase]['recall'].append(one_epoch_recall_vad.tolist())

            # plot_data_obj[training_phase]['ndcg_at_200'].append(one_epoch_ndcg_at_200_vad.tolist())
            # plot_data_obj[training_phase]['ndcg_at_50'].append(one_epoch_ndcg_at_50_vad.tolist())
            # plot_data_obj[training_phase]['ndcg_at_20'].append(one_epoch_ndcg_at_20_vad.tolist())
            # plot_data_obj[training_phase]['ndcg_at_5'].append(one_epoch_ndcg_at_5_vad.tolist())

            _print(plot_data_obj, verbose)

            # note that it includes reg...

            # one_epoch_critic_acc_loss = np.concatenate(one_epoch_critic_acc_loss).mean()
            # one_epoch_ndcg_train = np.concatenate(one_epoch_ndcg_train).mean()

            # merged_valid_val = sess.run(ndcg_logging, feed_dict={ndcgs_train_var: one_epoch_ndcg_train,
            #                                                      ndcgs_vad_var: one_epoch_ndcg_vad})

            # I only feel the need to log ndcg, because we know it's representative
            merged_validation_logging = sess.run(
                validation_ops['validation_logging'],
                feed_dict={
                    validation_ops['ndcg_vad_var']:
                    one_epoch_ndcg_vad,
                    # validation_ops['ap_vad_var']: one_epoch_ap_vad,
                    # validation_ops['recall_vad_var']: one_epoch_recall_vad,
                    validation_ops['critic_error_vad_var']:
                    one_epoch_critic_error,

                    # validation_ops['ndcg_at_200_vad_var']: one_epoch_ndcg_at_200_vad,
                    # validation_ops['ndcg_at_50_vad_var']: one_epoch_ndcg_at_50_vad,
                    # validation_ops['ndcg_at_20_vad_var']: one_epoch_ndcg_at_20_vad,
                    # validation_ops['ndcg_at_5_vad_var']: one_epoch_ndcg_at_5_vad,
                })

            # merged_validation_logging = sess.run(
            #     validation_logging,
            #     feed_dict={
            #         ndcg_vad_var: one_epoch_ndcg_vad,
            #         ap_vad_var: one_epoch_ap_vad,
            #         recall_vad_var: one_epoch_recall_vad,
            #         critic_error_vad_var: one_epoch_critic_error,
            #     })

            summary_writer.add_summary(merged_validation_logging, epoch)
            summary_writer.flush()  #At the end of an epoch I would like to see whats going on...

            print("NGCD VAD: {}".format(one_epoch_ndcg_vad))
            # print("AP VAD: {}".format(one_epoch_ap_vad))
            # print("RECALL VAD: {}".format(one_epoch_recall_vad))

            # if evaluation_metric == "NDCG":
            #     score = one_epoch_ndcg_vad
            # # elif evaluation_metric == "AP":
            # #     score = one_epoch_ap_vad
            # # elif evaluation_metric == "RECALL":
            # #     score = one_epoch_recall_vad
            # elif evaluation_metric == 'NDCG_AT_200':
            #     score = one_epoch_ndcg_at_200_vad
            # elif evaluation_metric == 'NDCG_AT_5':
            #     score = one_epoch_ndcg_at_5_vad
            # elif evaluation_metric == 'NDCG_AT_3':
            #     score = one_epoch_ndcg_at_3_vad
            # elif evaluation_metric == 'NDCG_AT_1':
            #     score = one_epoch_ndcg_at_1_vad
            # else:
            #     raise Exception("Should not get here.")

            score = one_epoch_score

            if score > best_score:
                print("new best on metric {}. Was {}, now {}. Saving".format(
                    evaluation_metric, best_score, score))
                model.saver.save(sess, '{}/model'.format(chkpt_dir))
                best_score = score
                if path_to_save_actor is not None:
                    print("Saving actor as well.")
                    model.actor_saver.save(sess, '{}/model'.format(path_to_save_actor))

            if epoch == n_epochs_pred_only - 1:
                if path_to_save_last_actor is not None:
                    print(
                        "Epoch {}. This is the last actor that was trained by the prediction model. Saving it so we can have pretty data continuity in the graph..."
                        .format(epoch))
                    model.actor_saver.save(sess, '{}/model'.format(path_to_save_last_actor))

            # This is so I can recover the data from partial runs if I need to, in case one is taking far too long.
            write_plotting_data(os.path.join("TEMP", arch_str), plot_data_obj)
    print("Wow, it feels good to be down here. Done for real")
    write_plotting_data(arch_str, plot_data_obj)


def test(
        model_class=None,
        data_subdir='ml-20m',
        evaluation_metric='NDCG',
        n_epochs_pred_only=0,
        n_epochs_ac_only=0,
        n_epochs_pred_and_ac=0,
        n_epochs_second_pred=0,
        batch_size=500,
        break_early=False,
        min_kl=0.0,
        max_kl=0.2,
        epochs_to_anneal_over=50,
        batches_to_anneal_over=200000,
        #   logging_frequency=25,
        #   path_to_save_actor=None,
        #   path_to_save_last_actor=None,
        restore_trained_actor_path=None,
        verbose=False,
        actor_reg_loss_scaler=1e-4,
        ac_reg_loss_scaler=0.0,
        positive_weights=2.0,
        omit_num_seen_from_critic=False,
        omit_num_unseen_from_critic=False,
        version_tag=""):

    assert data_subdir in ['ml-20m', 'netflix-prize', 'msd']
    assert model_class in ['multi_vae', 'warp_encoder', 'wmf', 'new_multi_dae', 'variational_wmf', 'gaussian_vae', 'phase_4_warp', 'lambdarank_actor', 'phase_4_lambdarank']

    train_args = locals()
    arch_str = make_arch_string(
        ['data_subdir', 'model_class'],
        ['path_to_save_actor', 'path_to_save_last_actor', 'logging_frequency'], **train_args)

    chkpt_dir = make_checkpoint_dir(arch_str)

    np.random.seed(98765)
    tf.set_random_seed(98765)

    tf.reset_default_graph()

    data_dict = get_test_data(data_subdir)

    n_items = data_dict['n_items']
    test_tr = data_dict['test_tr']
    test_te = data_dict['test_te']

    testing_dataset = tr_te_dataset(test_tr, test_te, batch_size)

    data_iterator = tf.data.Iterator.from_structure(testing_dataset.output_types,
                                                    testing_dataset.output_shapes)

    batch_of_users, heldout_batch = data_iterator.get_next()

    testing_init_op = data_iterator.make_initializer(testing_dataset)

    train_args['input_dim'] = n_items
    model = get_model(batch_of_users, heldout_batch, **train_args)

    print("Test!")

    # ndcg100_list, ap100_list, recall100_list = [], [], []
    # ndcg100_list, recall50_list, recall20_list = [], [], []
    ndcg100_list, recall50_list, recall20_list, ndcg200_list, ndcg5_list, ndcg3_list, ndcg1_list = [], [], [], [], [], [], []

    dcg100_list = []

    with tf.Session() as sess:
        model.saver.restore(sess, '{}/model'.format(chkpt_dir))
        sess.run(testing_init_op)
        try:
            # for bnum, st_idx in enumerate(range(0, N_test, batch_size_test)):
            while True:
                # print("Testing on batch {}".format(bnum))
                print('test batch.')
                # end_idx = min(st_idx + batch_size_test, N_test)
                # X = test_data_tr[idxlist_test[st_idx:end_idx]]
                # if sparse.isspmatrix(X):
                #     X = X.toarray()
                # X = X.astype('float32')

                feed_dict = {
                    # model.input_ph: X,
                    model.keep_prob_ph: 1.0,
                    model.stddev_effect_on_latent_dim_scaler: 0.0,
                }

                # pred_val, batch_of_users, heldout_batch, _ = sess.run(
                #     [model.prediction, model.batch_of_users, model.heldout_batch, model.vad_true_ndcg],
                #     feed_dict=feed_dict)

                pred_val, batch_of_users, heldout_batch = sess.run(
                    [model.prediction, model.batch_of_users, model.heldout_batch],
                    feed_dict=feed_dict)

                ndcg100_list.append(
                    NDCG_binary_at_k_batch(
                        pred_val, heldout_batch, k=100, input_batch=batch_of_users))
                recall50_list.append(
                    Recall_at_k_batch(pred_val, heldout_batch, k=50, input_batch=batch_of_users))
                recall20_list.append(
                    Recall_at_k_batch(pred_val, heldout_batch, k=20, input_batch=batch_of_users))
                ndcg200_list.append(
                    NDCG_binary_at_k_batch(
                        pred_val, heldout_batch, k=200, input_batch=batch_of_users))
                ndcg5_list.append(
                    NDCG_binary_at_k_batch(
                        pred_val, heldout_batch, k=5, input_batch=batch_of_users))
                ndcg3_list.append(
                    NDCG_binary_at_k_batch(
                        pred_val, heldout_batch, k=3, input_batch=batch_of_users))
                ndcg1_list.append(
                    NDCG_binary_at_k_batch(
                        pred_val, heldout_batch, k=1, input_batch=batch_of_users))
                
                dcg100_list.append(
                    NDCG_binary_at_k_batch(
                        pred_val, heldout_batch, k=100, input_batch=batch_of_users, normalize=False))

                # recall50_list.append(Recall_at_k_batch())

                # pred_val, heldout_batch, _ = sess.run(
                #     [model.vad_logits_out, model.heldout_batch, model.true_vad_ndcg],
                #     feed_dict=feed_dict)

                # ndcg100, ap100, recall100 = sess.run(
                #     [model.vad_true_ndcg, model.vad_true_ap, model.vad_true_recall],
                #     feed_dict=feed_dict)

                # NOTE: I don't know why you need to fetch the true_ndcg_vad. Maybe it's something about
                # graph_replace, that you need to get all of them together?
                # pred_val = sess.run(model.logits_out, feed_dict=feed_dict)
                # pred_val[X.nonzero()] = -np.inf
                # ndcg100_list.append(ndcg100)
                # ap100_list.append(ap100)
                # recall100_list.append(recall100)

                # n100_list.append(NDCG_binary_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=100))
                # r20_list.append(Recall_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=20))
                # r50_list.append(Recall_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=50))
        except tf.errors.OutOfRangeError:
            print("Testing done. That broke it out of the loop.")

    ndcg100_list = np.concatenate(ndcg100_list)
    # ap100_list = np.concatenate(ap100_list)
    # recall100_list = np.concatenate(recall100_list)

    # In[64]:
    print("Test UNNORMALIZED DCG@100=%.5f (%.5f)" % (np.mean(dcg100_list),
                                         np.std(dcg100_list) / np.sqrt(len(dcg100_list))))
    print("Test NDCG@100=%.5f (%.5f)" % (np.mean(ndcg100_list),
                                         np.std(ndcg100_list) / np.sqrt(len(ndcg100_list))))
    print("Test Recall@50=%.5f (%.5f)" % (np.mean(recall50_list),
                                          np.std(recall50_list) / np.sqrt(len(recall50_list))))
    print("Test Recall@020=%.5f (%.5f)" % (np.mean(recall20_list),
                                           np.std(recall20_list) / np.sqrt(len(recall20_list))))

    print("Test NDCG@0200=%.5f (%.5f)" % (np.mean(ndcg200_list),
                                           np.std(ndcg200_list) / np.sqrt(len(ndcg200_list))))
    print("Test NDCG@5=%.5f (%.5f)" % (np.mean(ndcg5_list),
                                           np.std(ndcg5_list) / np.sqrt(len(ndcg5_list))))
    print("Test NDCG@3=%.5f (%.5f)" % (np.mean(ndcg3_list),
                                           np.std(ndcg3_list) / np.sqrt(len(ndcg3_list))))
    print("Test NDCG@1=%.5f (%.5f)" % (np.mean(ndcg1_list),
                                           np.std(ndcg1_list) / np.sqrt(len(ndcg1_list))))


    import json
    with open("../TEST_RESULTS.txt", "a") as f:
        f.write("\n\n")
        f.write(json.dumps(train_args) + "\n")
        f.write("Test NDCG@100=%.5f (%.5f)\n" % (np.mean(ndcg100_list),
                                                 np.std(ndcg100_list) / np.sqrt(len(ndcg100_list))))
        f.write("Test Recall@50=%.5f (%.5f)\n" %
                (np.mean(recall50_list), np.std(recall50_list) / np.sqrt(len(recall50_list))))
        f.write("Test Recall@20=%.5f (%.5f)\n" %
                (np.mean(recall20_list), np.std(recall20_list) / np.sqrt(len(recall20_list))))

        f.write("Test NDCG@0200=%.5f (%.5f)\n" % (np.mean(ndcg200_list),
                                            np.std(ndcg200_list) / np.sqrt(len(ndcg200_list))))
        f.write("Test NDCG@5=%.5f (%.5f)\n" % (np.mean(ndcg5_list),
                                            np.std(ndcg5_list) / np.sqrt(len(ndcg5_list))))
        f.write("Test NDCG@3=%.5f (%.5f)\n" % (np.mean(ndcg3_list),
                                            np.std(ndcg3_list) / np.sqrt(len(ndcg3_list))))
        f.write("Test NDCG@1=%.5f (%.5f)\n" % (np.mean(ndcg1_list),
                                            np.std(ndcg1_list) / np.sqrt(len(ndcg1_list))))
