import sys
import os
UTILS_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'utils')
sys.path.insert(1, UTILS_DIR)

from training import train, test

if __name__ == '__main__':
    train(
        # model_class="wmf",
        model_class='lambdarank_actor',
        # model_class='warp_encoder',
        n_epochs_pred_only=10,
        n_epochs_ac_only=0,
        n_epochs_pred_and_ac=0,
        epochs_to_anneal_over=100,
        # min_kl=0.0001,
        max_kl=0.2,
        ac_reg_loss_scaler=0.0,
        actor_reg_loss_scaler=1e-5,
        # positive_weights=5,
        # evaluation_metric='AP',
        evaluation_metric="NDCG",
        logging_frequency=50,
        # logging_frequency=25,
        # logging_frequency=50,
        # logging_frequency=50,
        # batch_size=500,
        # batch_size=10,
        # batch_size=25,
        # batch_size=500,
        # break_early=True,
        # # break_early=False,
        # verbose=False,
        verbose=True,
        # path_to_save_actor="best_ndcg_trained_150_epochs",
        # path_to_save_last_actor="last_actor_after_150_trained_epochs",
        version_tag="10_epochs_of_lambdarank",
        # path_to_save_actor="BEST_WARP_RUN_15_EPOCHS_TRUTHFUL_LOSS",
        # restore_trained_actor_path="BEST_WARP_RUN_15_EPOCHS_TRUTHFUL_LOSS"
    )
