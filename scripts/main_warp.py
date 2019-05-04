import sys
import os
UTILS_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'utils')
sys.path.insert(1, UTILS_DIR)

from training import train, test

if __name__ == '__main__':

    """
    NOTE: This takes roughly 30 minutes per epoch with a good GPU
    """


    train(
        model_class='warp_encoder',
        n_epochs_pred_only=0,
        n_epochs_ac_only=10,
        n_epochs_pred_and_ac=10,
        epochs_to_anneal_over=100,
        # min_kl=0.0001,
        max_kl=0.0,
        ac_reg_loss_scaler=0.0,
        actor_reg_loss_scaler=1e-5,
        # positive_weights=5,
        # evaluation_metric='AP',
        evaluation_metric="NDCG",
        logging_frequency=25,
        # logging_frequency=50,
        # logging_frequency=50,
        batch_size=500,
        # batch_size=25,
        break_early=False,
        verbose=False,
        # path_to_save_actor="best_ndcg_trained_150_epochs",
        # path_to_save_last_actor="last_actor_after_150_trained_epochs",
        version_tag="WARP_WITH_CRITIC",
        # path_to_save_actor="BEST_WARP_RUN_15_EPOCHS_TRUTHFUL_LOSS",
        restore_trained_actor_path="BEST_WARP_RUN_15_EPOCHS_TRUTHFUL_LOSS"
    )

    print("On to testing.")

    test(
        # model_class="wmf",
        # model_class='multi_vae',
        model_class='warp_encoder',
        n_epochs_pred_only=0,
        n_epochs_ac_only=10,
        n_epochs_pred_and_ac=10,
        epochs_to_anneal_over=100,
        # min_kl=0.0001,
        max_kl=0.0,
        ac_reg_loss_scaler=0.0,
        actor_reg_loss_scaler=1e-5,
        # positive_weights=5,
        # evaluation_metric='AP',
        evaluation_metric="NDCG",
        # logging_frequency=25,
        # logging_frequency=50,
        # logging_frequency=50,
        batch_size=500,
        # batch_size=25,
        break_early=False,
        verbose=False,
        # path_to_save_actor="best_ndcg_trained_150_epochs",
        # path_to_save_last_actor="last_actor_after_150_trained_epochs",
        version_tag="WARP_WITH_CRITIC",
        # path_to_save_actor="BEST_WARP_RUN_15_EPOCHS_TRUTHFUL_LOSS",
        restore_trained_actor_path="BEST_WARP_RUN_15_EPOCHS_TRUTHFUL_LOSS"
    )

    exit()

