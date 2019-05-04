import sys
import os
UTILS_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'utils')
sys.path.insert(1, UTILS_DIR)

from training import train, test

if __name__ == '__main__':

    BREAK_EARLY = False
    BATCH_SIZE = 500

    for data_subdir in ['ml-20m', 'netflix-prize', 'msd']:
        actor_path = "VAE_ACTOR_TRAIN_{}".format(data_subdir)
        train(
            model_class='multi_vae',
            data_subdir=data_subdir,
            n_epochs_pred_only=100,
            n_epochs_ac_only=0,
            n_epochs_pred_and_ac=0,
            max_kl=0.2,
            ac_reg_loss_scaler=0.0,
            actor_reg_loss_scaler=0.01,
            evaluation_metric="NDCG",
            logging_frequency=50,
            batch_size=BATCH_SIZE,
            break_early=BREAK_EARLY,
            verbose=False,
            version_tag="FULL_RUN_ON_OTHER_DATASETS",
            path_to_save_actor=actor_path,
            log_critic_training_error=False,
        )

        print("Now, hopefully on to testing...")

        test(
            model_class='multi_vae',
            data_subdir=data_subdir,
            n_epochs_pred_only=100,
            n_epochs_ac_only=0,
            n_epochs_pred_and_ac=0,
            max_kl=0.2,
            ac_reg_loss_scaler=0.0,
            actor_reg_loss_scaler=0.01,
            evaluation_metric="NDCG",
            batch_size=BATCH_SIZE,
            break_early=BREAK_EARLY,
            verbose=False,
            version_tag="FULL_RUN_ON_OTHER_DATASETS",
        )

        print("On to round 2! Now we'll do the critic.")

        train(
            model_class='multi_vae',
            data_subdir=data_subdir,
            n_epochs_pred_only=0,
            n_epochs_ac_only=50,
            n_epochs_pred_and_ac=50,
            max_kl=0.2,
            ac_reg_loss_scaler=0.0,
            actor_reg_loss_scaler=0.01,
            evaluation_metric="NDCG",
            logging_frequency=50,
            batch_size=BATCH_SIZE,
            break_early=BREAK_EARLY,
            verbose=False,
            version_tag="FULL_RUN_ON_OTHER_DATASETS",
            restore_trained_actor_path=actor_path,
        )

        print("Now, hopefully on to testing...")

        test(
            model_class='multi_vae',
            data_subdir=data_subdir,
            n_epochs_pred_only=0,
            n_epochs_ac_only=50,
            n_epochs_pred_and_ac=50,
            max_kl=0.2,
            ac_reg_loss_scaler=0.0,
            actor_reg_loss_scaler=0.01,
            evaluation_metric="NDCG",
            batch_size=BATCH_SIZE,
            break_early=BREAK_EARLY,
            verbose=False,
            version_tag="FULL_RUN_ON_OTHER_DATASETS",
            restore_trained_actor_path=actor_path,
        )


    print("Bye bye")
    exit()
