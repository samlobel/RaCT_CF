import sys
import os
UTILS_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'utils')
sys.path.insert(1, UTILS_DIR)

from training import train, test

if __name__ == '__main__':

    BREAK_EARLY = False
    BATCH_SIZE = 500

    cutoff = "NDCG_AT_200"

    for evaluation_metric in ["NDCG_AT_200", "NDCG_AT_5", "NDCG_AT_3", "NDCG_AT_1"]:

        train(
            model_class='multi_vae',
            # data_subdir=data_subdir,
            n_epochs_pred_only=0,
            n_epochs_ac_only=50,
            n_epochs_pred_and_ac=100,
            max_kl=0.2,
            ac_reg_loss_scaler=0.0,
            actor_reg_loss_scaler=0.01,
            evaluation_metric=evaluation_metric,
            logging_frequency=50,
            batch_size=BATCH_SIZE,
            break_early=BREAK_EARLY,
            verbose=False,
            version_tag="DIFFERENT_CUTOFFS",
            # path_to_save_actor=actor_path,
            restore_trained_actor_path="200_epochs_HIS_KL_annealing",
            log_critic_training_error=False,
        )

        print("Now, hopefully on to testing...")

        test(
            model_class='multi_vae',
            # data_subdir=data_subdir,
            n_epochs_pred_only=0,
            n_epochs_ac_only=50,
            n_epochs_pred_and_ac=100,
            max_kl=0.2,
            ac_reg_loss_scaler=0.0,
            actor_reg_loss_scaler=0.01,
            evaluation_metric=evaluation_metric,
            batch_size=BATCH_SIZE,
            break_early=BREAK_EARLY,
            verbose=False,
            version_tag="DIFFERENT_CUTOFFS",
            restore_trained_actor_path="200_epochs_HIS_KL_annealing",
        )

    exit("Bye bye now")
