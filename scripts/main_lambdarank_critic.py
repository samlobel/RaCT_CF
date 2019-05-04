import sys
import os
UTILS_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'utils')
sys.path.insert(1, UTILS_DIR)

from training import train, test
import tensorflow as tf

if __name__ == '__main__':

    # It's going to use the pre-trained model.

    DEFAULT_KWARGS = {
        'data_subdir': 'msd',
        'model_class': 'phase_4_lambdarank',
        # model_class: "multi_vae",
        # model_class: 'warp_encoder',
        # 'n_epochs_pred_only': 1,
        'n_epochs_pred_only': 0,
        'n_epochs_ac_only': 0,
        'n_epochs_pred_and_ac': 0,
        'n_epochs_second_pred': 25,
        'epochs_to_anneal_over': 100,
        # 'min_kl': 0.0001,
        'max_kl': 0.2,
        'ac_reg_loss_scaler': 0.0,
        'actor_reg_loss_scaler': 1e-5,
        # 'positive_weights': 5,
        # 'evaluation_metric': 'AP',
        'evaluation_metric': "NDCG",
        'logging_frequency': 50,
        # 'logging_frequency': 5,
        # 'logging_frequency': 25,
        # 'logging_frequency': 50,
        # 'logging_frequency': 50,
        # 'batch_size': 500,
        # 'batch_size': 5,
        # 'break_early': True,
        # 'break_early': False,
        'verbose': True,
        # 'path_to_save_actor': "best_ndcg_trained_150_epochs",
        # 'path_to_save_last_actor': "last_actor_after_150_trained_epochs",
        'version_tag': "PHASE_4_LAMBDARANK",
        'path_to_save_actor': "MSD_BEST_ACTOR_FROM_LAMBDARANK_PHASE_4",
        'restore_trained_actor_path': "MSD_BEST_ACTOR_FROM_WARP_CRITIC_ONLY_PHASE_1_FULL"
    }

    # jUST PHASE 4 THIS TIME
    FIRST_TRAIN_KWARGS = dict(DEFAULT_KWARGS)

    # NOW FOR TESTING
    FIRST_TEST_KWARGS = dict(DEFAULT_KWARGS)
    del FIRST_TEST_KWARGS['logging_frequency']
    del FIRST_TEST_KWARGS['path_to_save_actor']

    train(**FIRST_TRAIN_KWARGS)
    tf.reset_default_graph()
    test(**FIRST_TEST_KWARGS)
    exit()
