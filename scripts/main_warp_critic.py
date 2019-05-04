import sys
import os
UTILS_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'utils')
sys.path.insert(1, UTILS_DIR)

from training import train, test
import tensorflow as tf

if __name__ == '__main__':

    DEFAULT_KWARGS = {
        'model_class': 'phase_4_warp',
        # model_class: "multi_vae",
        # model_class: 'warp_encoder',
        # 'n_epochs_pred_only': 1,
        'n_epochs_pred_only': 150,
        'n_epochs_ac_only': 0,
        'n_epochs_pred_and_ac': 0,
        'n_epochs_second_pred': 0,
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
        # 'batch_size': 25,
        # 'break_early': True,
        # 'break_early': False,
        'verbose': False,
        # 'path_to_save_actor': "best_ndcg_trained_150_epochs",
        # 'path_to_save_last_actor': "last_actor_after_150_trained_epochs",
        'version_tag': "PHASE_4_WARP",
        'path_to_save_actor': "BEST_ACTOR_FROM_WARP_CRITIC_ONLY_PHASE_1_FULL",
        # 'restore_trained_actor_path': "BEST_WARP_RUN_15_EPOCHS_TRUTHFUL_LOSS"
    }

    # First, we do phase 1
    FIRST_TRAIN_KWARGS = dict(DEFAULT_KWARGS)

    # Then, we test phase 1
    FIRST_TEST_KWARGS = dict(DEFAULT_KWARGS)
    del FIRST_TEST_KWARGS['logging_frequency']
    del FIRST_TEST_KWARGS['path_to_save_actor']

    # Then, we only do phase 4
    SECOND_TRAIN_KWARGS = dict(DEFAULT_KWARGS)
    SECOND_TRAIN_KWARGS['n_epochs_pred_only'] = 0
    # SECOND_TRAIN_KWARGS['n_epochs_second_pred'] = 1
    SECOND_TRAIN_KWARGS['n_epochs_second_pred'] = 50
    SECOND_TRAIN_KWARGS['restore_trained_actor_path'] = FIRST_TRAIN_KWARGS['path_to_save_actor']
    SECOND_TRAIN_KWARGS['path_to_save_actor'] = 'BEST_ACTOR_FROM_WARP_CRITIC_PHASE_4_FULL'

    # Then, we test phase 4
    SECOND_TEST_KWARGS = dict(SECOND_TRAIN_KWARGS)
    print(SECOND_TEST_KWARGS)
    del SECOND_TEST_KWARGS['logging_frequency']
    del SECOND_TEST_KWARGS['path_to_save_actor']

    # Finally, we do the remainder of phase 1.
    THIRD_TRAIN_KWARGS = dict(DEFAULT_KWARGS)
    # THIRD_TRAIN_KWARGS['n_epochs_pred_only']=1
    THIRD_TRAIN_KWARGS['n_epochs_pred_only']=50
    THIRD_TRAIN_KWARGS['restore_trained_actor_path'] = FIRST_TRAIN_KWARGS['path_to_save_actor']
    THIRD_TRAIN_KWARGS['path_to_save_actor'] = 'BEST_ACTOR_FROM_WARP_CRITIC_AFTER_ADDITIONAL_PHASE_1'

    THIRD_TEST_KWARGS = dict(THIRD_TRAIN_KWARGS)
    del THIRD_TEST_KWARGS['logging_frequency']
    del THIRD_TEST_KWARGS['path_to_save_actor']


    train(**FIRST_TRAIN_KWARGS)
    tf.reset_default_graph()
    test(**FIRST_TEST_KWARGS)
    tf.reset_default_graph()
    train(**SECOND_TRAIN_KWARGS)
    tf.reset_default_graph()
    test(**SECOND_TEST_KWARGS)
    tf.reset_default_graph()
    train(**THIRD_TRAIN_KWARGS)
    tf.reset_default_graph()
    test(**THIRD_TEST_KWARGS)
    tf.reset_default_graph()
    
    exit()

    # print('\n\n\nchange from multivae!!!!!!\n\n\n')
    print("First, train just the actor")
    train(
        # model_class="wmf",
        model_class='phase_4_warp',
        # model_class="multi_vae",
        # model_class='warp_encoder',
        n_epochs_pred_only=2,
        n_epochs_ac_only=0,
        n_epochs_pred_and_ac=0,
        n_epochs_second_pred=0,
        epochs_to_anneal_over=100,
        # min_kl=0.0001,
        max_kl=0.2,
        ac_reg_loss_scaler=0.0,
        actor_reg_loss_scaler=1e-5,
        # positive_weights=5,
        # evaluation_metric='AP',
        evaluation_metric="NDCG",
        # logging_frequency=50,
        logging_frequency=5,
        # logging_frequency=25,
        # logging_frequency=50,
        # logging_frequency=50,
        # batch_size=500,
        batch_size=25,
        break_early=True,
        # break_early=False,
        verbose=False,
        # path_to_save_actor="best_ndcg_trained_150_epochs",
        # path_to_save_last_actor="last_actor_after_150_trained_epochs",
        version_tag="PHASE_4_WARP",
        path_to_save_actor="BEST_ACTOR_FROM_WARP_CRITIC_ONLY_PHASE_1",
        # restore_trained_actor_path="BEST_WARP_RUN_15_EPOCHS_TRUTHFUL_LOSS"
    )

    print("Testing time!")
    tf.reset_default_graph()

    test(
        # model_class="wmf",
        model_class='phase_4_warp',
        # model_class="multi_vae",
        # model_class='warp_encoder',
        n_epochs_pred_only=2,
        n_epochs_ac_only=0,
        n_epochs_pred_and_ac=0,
        n_epochs_second_pred=0,
        epochs_to_anneal_over=100,
        # min_kl=0.0001,
        max_kl=0.2,
        ac_reg_loss_scaler=0.0,
        actor_reg_loss_scaler=1e-5,
        # positive_weights=5,
        # evaluation_metric='AP',
        evaluation_metric="NDCG",
        # logging_frequency=50,
        # logging_frequency=5,
        # logging_frequency=25,
        # logging_frequency=50,
        # logging_frequency=50,
        # batch_size=500,
        batch_size=25,
        break_early=True,
        # break_early=False,
        verbose=False,
        # path_to_save_actor="best_ndcg_trained_150_epochs",
        # path_to_save_last_actor="last_actor_after_150_trained_epochs",
        version_tag="PHASE_4_WARP",
        # path_to_save_actor="BEST_ACTOR_FROM_WARP_CRITIC_ONLY_PHASE_1",
        # restore_trained_actor_path="BEST_WARP_RUN_15_EPOCHS_TRUTHFUL_LOSS"
    )


    print("Training with the pretrained actor")
    tf.reset_default_graph()
    train(
        # model_class="wmf",
        model_class='phase_4_warp',
        # model_class="multi_vae",
        # model_class='warp_encoder',
        n_epochs_pred_only=2,
        n_epochs_ac_only=0,
        n_epochs_pred_and_ac=0,
        n_epochs_second_pred=0,
        epochs_to_anneal_over=100,
        # min_kl=0.0001,
        max_kl=0.2,
        ac_reg_loss_scaler=0.0,
        actor_reg_loss_scaler=1e-5,
        # positive_weights=5,
        # evaluation_metric='AP',
        evaluation_metric="NDCG",
        # logging_frequency=50,
        logging_frequency=5,
        # logging_frequency=25,
        # logging_frequency=50,
        # logging_frequency=50,
        # batch_size=500,
        batch_size=25,
        break_early=True,
        # break_early=False,
        verbose=False,
        # path_to_save_actor="best_ndcg_trained_150_epochs",
        # path_to_save_last_actor="last_actor_after_150_trained_epochs",
        version_tag="PHASE_4_WARP",
        # path_to_save_actor="BEST_ACTOR_FROM_WARP_CRITIC_ONLY_PHASE_1",
        restore_trained_actor_path="BEST_ACTOR_FROM_WARP_CRITIC_ONLY_PHASE_1"
    )

    print("Testing!")
    tf.reset_default_graph()
    test(
        # model_class="wmf",
        model_class='phase_4_warp',
        # model_class="multi_vae",
        # model_class='warp_encoder',
        n_epochs_pred_only=2,
        n_epochs_ac_only=0,
        n_epochs_pred_and_ac=0,
        n_epochs_second_pred=0,
        epochs_to_anneal_over=100,
        # min_kl=0.0001,
        max_kl=0.2,
        ac_reg_loss_scaler=0.0,
        actor_reg_loss_scaler=1e-5,
        # positive_weights=5,
        # evaluation_metric='AP',
        evaluation_metric="NDCG",
        # logging_frequency=50,
        logging_frequency=5,
        # logging_frequency=25,
        # logging_frequency=50,
        # logging_frequency=50,
        # batch_size=500,
        batch_size=25,
        break_early=True,
        # break_early=False,
        verbose=False,
        # path_to_save_actor="best_ndcg_trained_150_epochs",
        # path_to_save_last_actor="last_actor_after_150_trained_epochs",
        version_tag="PHASE_4_WARP",
        # path_to_save_actor="BEST_ACTOR_FROM_WARP_CRITIC_ONLY_PHASE_1",
        restore_trained_actor_path="BEST_ACTOR_FROM_WARP_CRITIC_ONLY_PHASE_1"
    )
    