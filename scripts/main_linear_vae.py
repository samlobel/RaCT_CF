import sys
import os
UTILS_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'utils')
sys.path.insert(1, UTILS_DIR)


from training import train, test

if __name__ == '__main__':

    BREAK_EARLY = False
    BATCH_SIZE = 500

    # actor_path = "WMFVAE_ACTOR_TRAIN_{}".format(data_subdir)
    actor_path = "CORRECT_VWMF_ACTOR"
    train(
        model_class='variational_wmf',
        # data_subdir=data_subdir,
        n_epochs_pred_only=200,
        n_epochs_ac_only=0,
        n_epochs_pred_and_ac=0,
        max_kl=0.05,
        ac_reg_loss_scaler=0.0,
        actor_reg_loss_scaler=0.0001,
        evaluation_metric="NDCG",
        logging_frequency=50,
        batch_size=BATCH_SIZE,
        break_early=BREAK_EARLY,
        verbose=False,
        positive_weights=5.0,
        version_tag="VWMF_JUST_ACTOR",
        path_to_save_actor=actor_path,
        log_critic_training_error=False,
    )

    print("Now, hopefully on to testing...")

    test(
        model_class='variational_wmf',
        # data_subdir=data_subdir,
        n_epochs_pred_only=200,
        n_epochs_ac_only=0,
        n_epochs_pred_and_ac=0,
        max_kl=0.05,
        ac_reg_loss_scaler=0.0,
        actor_reg_loss_scaler=0.0001,
        evaluation_metric="NDCG",
        batch_size=BATCH_SIZE,
        break_early=BREAK_EARLY,
        verbose=False,
        positive_weights=5.0,
        version_tag="VWMF_JUST_ACTOR",
    )

    print("On to round 2! Now we'll do the critic.")

    train(
        model_class='variational_wmf',
        # data_subdir=data_subdir,
        n_epochs_pred_only=0,
        n_epochs_ac_only=50,
        n_epochs_pred_and_ac=50,
        max_kl=0.05,
        ac_reg_loss_scaler=0.0,
        actor_reg_loss_scaler=0.0001,
        evaluation_metric="NDCG",
        logging_frequency=50,
        batch_size=BATCH_SIZE,
        break_early=BREAK_EARLY,
        verbose=False,
        positive_weights=5.0,
        version_tag="VWMF_WITH_CRITIC",
        restore_trained_actor_path=actor_path,
    )

    print("Now, hopefully on to testing...")

    test(
        model_class='variational_wmf',
        # data_subdir=data_subdir,
        n_epochs_pred_only=0,
        n_epochs_ac_only=50,
        n_epochs_pred_and_ac=50,
        max_kl=0.05,
        ac_reg_loss_scaler=0.0,
        actor_reg_loss_scaler=0.0001,
        evaluation_metric="NDCG",
        batch_size=BATCH_SIZE,
        break_early=BREAK_EARLY,
        verbose=False,
        positive_weights=5.0,
        version_tag="VWMF_WITH_CRITIC",
        restore_trained_actor_path=actor_path,
    )


    print("Bye bye")
    exit()


    # train(
    #     # model_class="wmf",
    #     # model_class='multi_dae',
    #     model_class="wmf_vae",
    #     # model_class='warp_encoder',
    #     n_epochs_pred_only=200,
    #     n_epochs_ac_only=0,
    #     n_epochs_pred_and_ac=0,
    #     # epochs_to_anneal_over=100,
    #     # min_kl=0.0001,
    #     max_kl=0.05,
    #     # ac_reg_loss_scaler=0.0005,
    #     ac_reg_loss_scaler=0.0,
    #     # actor_reg_loss_scaler=0.00001,
    #     # actor_reg_loss_scaler=0.01,
    #     positive_weights=5.0,
    #     # evaluation_metric='AP',
    #     evaluation_metric="NDCG",
    #     # logging_frequency=25,
    #     # logging_frequency=50,
    #     logging_frequency=50,
    #     batch_size=500,
    #     # batch_size=25,
    #     break_early=False,
    #     verbose=False,
    #     # path_to_save_actor="best_ndcg_trained_150_epochs",
    #     # path_to_save_last_actor="last_actor_after_150_trained_epochs",
    #     version_tag="FULL_WMFVAE_RUN_JUST_ACTOR",
    #     path_to_save_actor="200_EPOCHS_WMFVAE_AT_0.05_KL_JUST_ACTOR",
    #     # path_to_save_last_actor="LAST_ACTOR_OF_200_epochs_HIS_KL_annealing",
    #     # restore_trained_actor_path="200_epochs_HIS_DAE",
    # )

    # train(
    #     # model_class="wmf",
    #     # model_class='multi_dae',
    #     model_class="wmf_vae",
    #     # model_class='warp_encoder',
    #     n_epochs_pred_only=0,
    #     n_epochs_ac_only=50,
    #     n_epochs_pred_and_ac=100,
    #     # epochs_to_anneal_over=100,
    #     # min_kl=0.0001,
    #     max_kl=0.05,
    #     # ac_reg_loss_scaler=0.0005,
    #     ac_reg_loss_scaler=0.0,
    #     # actor_reg_loss_scaler=0.00001,
    #     # actor_reg_loss_scaler=0.01,
    #     positive_weights=5.0,
    #     # evaluation_metric='AP',
    #     evaluation_metric="NDCG",
    #     # logging_frequency=25,
    #     # logging_frequency=50,
    #     logging_frequency=50,
    #     batch_size=500,
    #     # batch_size=25,
    #     break_early=False,
    #     verbose=False,
    #     # path_to_save_actor="best_ndcg_trained_150_epochs",
    #     # path_to_save_last_actor="last_actor_after_150_trained_epochs",
    #     version_tag="FULL_WMFVAE_RUN_WITH_CRITIC",
    #     # path_to_save_actor="200_EPOCHS_WMFVAE_AT_0.05_KL",
    #     # path_to_save_last_actor="LAST_ACTOR_OF_200_epochs_HIS_KL_annealing",
    #     restore_trained_actor_path="200_EPOCHS_WMFVAE_AT_0.05_KL_JUST_ACTOR",
    # )


    # print("Now that we've done the thing we really care about, let's have some fun with hyperparameters")

    # for max_kl in [0.4, 0.2, 0.1, 0.01]:

    #     train(
    #         # model_class="wmf",
    #         # model_class='multi_dae',
    #         model_class="wmf_vae",
    #         # model_class='warp_encoder',
    #         n_epochs_pred_only=100,
    #         n_epochs_ac_only=0,
    #         n_epochs_pred_and_ac=0,
    #         # epochs_to_anneal_over=100,
    #         # min_kl=0.0001,
    #         max_kl=max_kl,
    #         # ac_reg_loss_scaler=0.0005,
    #         ac_reg_loss_scaler=0.0,
    #         # actor_reg_loss_scaler=0.00001,
    #         actor_reg_loss_scaler=0.01,
    #         positive_weights=5.0,
    #         # evaluation_metric='AP',
    #         evaluation_metric="NDCG",
    #         # logging_frequency=25,
    #         # logging_frequency=50,
    #         logging_frequency=50,
    #         batch_size=500,
    #         # batch_size=25,
    #         break_early=False,
    #         verbose=False,
    #         # path_to_save_actor="best_ndcg_trained_150_epochs",
    #         # path_to_save_last_actor="last_actor_after_150_trained_epochs",
    #         version_tag="TESTING_HYPERPARAMETERS",
    #         # path_to_save_actor="200_epochs_HIS_DAE",
    #         # path_to_save_last_actor="LAST_ACTOR_OF_200_epochs_HIS_KL_annealing",
    #         # restore_trained_actor_path="200_epochs_HIS_DAE",
    #     )
    #     print("On to the next one...")

    # exit()
    # print("Now, hopefully on to testing...")

    # test(
    #     # model_class="wmf",
    #     model_class='multi_dae',
    #     # model_class='warp_encoder',
    #     n_epochs_pred_only=10,
    #     n_epochs_ac_only=0,
    #     n_epochs_pred_and_ac=0,
    #     # epochs_to_anneal_over=100,
    #     # min_kl=0.0001,
    #     # max_kl=0.2,
    #     # ac_reg_loss_scaler=0.0,
    #     # actor_reg_loss_scaler=0.01,
    #     positive_weights=5.0,
    #     # evaluation_metric='AP',
    #     evaluation_metric="NDCG",
    #     # logging_frequency=25,
    #     # logging_frequency=50,
    #     # logging_frequency=50,
    #     batch_size=500,
    #     # batch_size=25,
    #     break_early=False,
    #     verbose=False,
    #     # path_to_save_actor="best_ndcg_trained_150_epochs",
    #     # path_to_save_last_actor="last_actor_after_150_trained_epochs",
    #     version_tag="TRAINING_DAE",
    #     # path_to_save_actor="200_epochs_HIS_DAE",
    #     # restore_trained_actor_path="200_epochs_HIS_DAE",
    # )

    # print("On to round 2! Now we'll do the critic.")

    # train(
    #     # model_class="wmf",
    #     model_class='multi_dae',
    #     # model_class='warp_encoder',
    #     n_epochs_pred_only=0,
    #     n_epochs_ac_only=50,
    #     n_epochs_pred_and_ac=100,
    #     # epochs_to_anneal_over=100,
    #     # min_kl=0.0001,
    #     max_kl=0.2,
    #     # ac_reg_loss_scaler=0.0005,
    #     ac_reg_loss_scaler=0.0,
    #     # actor_reg_loss_scaler=0.00001,
    #     actor_reg_loss_scaler=0.01,
    #     # positive_weights=positive_weights,
    #     # evaluation_metric='AP',
    #     evaluation_metric="NDCG",
    #     # logging_frequency=25,
    #     # logging_frequency=50,
    #     logging_frequency=50,
    #     batch_size=500,
    #     # batch_size=25,
    #     break_early=False,
    #     verbose=False,
    #     # path_to_save_actor="best_ndcg_trained_150_epochs",
    #     # path_to_save_last_actor="last_actor_after_150_trained_epochs",
    #     version_tag="TRAINING_DAE",
    #     # path_to_save_actor="200_epochs_HIS_DAE",
    #     # path_to_save_last_actor="LAST_ACTOR_OF_200_epochs_HIS_KL_annealing",
    #     restore_trained_actor_path="200_epochs_HIS_DAE",
    # )

    # print("Now, hopefully on to testing...")

    # test(
    #     # model_class="wmf",
    #     model_class='multi_dae',
    #     # model_class='warp_encoder',
    #     n_epochs_pred_only=0,
    #     n_epochs_ac_only=50,
    #     n_epochs_pred_and_ac=100,
    #     # epochs_to_anneal_over=100,
    #     # min_kl=0.0001,
    #     max_kl=0.2,
    #     ac_reg_loss_scaler=0.0,
    #     actor_reg_loss_scaler=0.01,
    #     # positive_weights=positive_weights,
    #     # evaluation_metric='AP',
    #     evaluation_metric="NDCG",
    #     # logging_frequency=25,
    #     # logging_frequency=50,
    #     # logging_frequency=50,
    #     batch_size=500,
    #     # batch_size=25,
    #     break_early=False,
    #     verbose=False,
    #     # path_to_save_actor="best_ndcg_trained_150_epochs",
    #     # path_to_save_last_actor="last_actor_after_150_trained_epochs",
    #     version_tag="TRAINING_DAE",
    #     # path_to_save_actor="200_epochs_HIS_DAE",
    #     restore_trained_actor_path="200_epochs_HIS_DAE",
    # )

    # print("Bye bye")
    # exit()
