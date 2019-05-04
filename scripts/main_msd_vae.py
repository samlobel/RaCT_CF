import sys
import os
UTILS_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'utils')
sys.path.insert(1, UTILS_DIR)

from training import train, test

if __name__ == '__main__':

    BREAK_EARLY = False
    BATCH_SIZE = 500

    print("For this one, I just want to try out a different KL or two. Because we didn't get the competitive results we asked for the other way.")

    for data_subdir in ['msd']:
        actor_path = "VAE_ACTOR_TRAIN_{}_KL=0.1".format(data_subdir)
        train(
            model_class='multi_vae',
            data_subdir=data_subdir,
            n_epochs_pred_only=100,
            n_epochs_ac_only=0,
            n_epochs_pred_and_ac=0,
            # max_kl=0.2,
            max_kl=0.1,
            ac_reg_loss_scaler=0.0,
            actor_reg_loss_scaler=0.01,
            evaluation_metric="NDCG",
            logging_frequency=200,
            batch_size=BATCH_SIZE,
            break_early=BREAK_EARLY,
            verbose=False,
            version_tag="FULL_RUN_ON_MSD_ONLY",
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
            # max_kl=0.2,
            max_kl=0.1,
            ac_reg_loss_scaler=0.0,
            actor_reg_loss_scaler=0.01,
            evaluation_metric="NDCG",
            batch_size=BATCH_SIZE,
            break_early=BREAK_EARLY,
            verbose=False,
            version_tag="FULL_RUN_ON_MSD_ONLY",
        )

        print("On to round 2! Now we'll do the critic.")

        train(
            model_class='multi_vae',
            data_subdir=data_subdir,
            n_epochs_pred_only=0,
            n_epochs_ac_only=50,
            n_epochs_pred_and_ac=50,
            # max_kl=0.2,
            max_kl=0.1,
            ac_reg_loss_scaler=0.0,
            actor_reg_loss_scaler=0.01,
            evaluation_metric="NDCG",
            logging_frequency=200,
            batch_size=BATCH_SIZE,
            break_early=BREAK_EARLY,
            verbose=False,
            version_tag="FULL_RUN_ON_MSD_ONLY",
            restore_trained_actor_path=actor_path,
        )

        print("Now, hopefully on to testing...")

        test(
            model_class='multi_vae',
            data_subdir=data_subdir,
            n_epochs_pred_only=0,
            n_epochs_ac_only=50,
            n_epochs_pred_and_ac=50,
            # max_kl=0.2,
            max_kl=0.1,
            ac_reg_loss_scaler=0.0,
            actor_reg_loss_scaler=0.01,
            evaluation_metric="NDCG",
            batch_size=BATCH_SIZE,
            break_early=BREAK_EARLY,
            verbose=False,
            version_tag="FULL_RUN_ON_MSD_ONLY",
            restore_trained_actor_path=actor_path,
        )


    print("Bye bye")
    exit()

    # train(
    #     # model_class="wmf",
    #     model_class='multi_dae',
    #     # model_class='warp_encoder',
    #     n_epochs_pred_only=200,
    #     n_epochs_ac_only=0,
    #     n_epochs_pred_and_ac=0,
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
    #     path_to_save_actor="200_epochs_HIS_DAE",
    #     # path_to_save_last_actor="LAST_ACTOR_OF_200_epochs_HIS_KL_annealing",
    #     # restore_trained_actor_path="200_epochs_HIS_DAE",
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

    # for positive_weights in [2.0, 5.0, 10.0, 30.0, 50.0, 100.0]:
    #     train(
    #         model_class="wmf",
    #         # model_class='multi_vae',
    #         # model_class='warp_encoder',
    #         n_epochs_pred_only=50,
    #         n_epochs_ac_only=0,
    #         n_epochs_pred_and_ac=0,
    #         epochs_to_anneal_over=50,
    #         max_kl=0.2,
    #         ac_reg_loss_scaler=0.0005,
    #         actor_reg_loss_scaler=0.00001,
    #         positive_weights=positive_weights,
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
    #         version_tag="making_one_trained_on_each_eval_metric",
    #         # path_to_save_actor="test_actor_save",
    #         # restore_trained_actor_path="last_actor_after_150_trained_epochs"
    #     )
    # exit("Exiting gracefully!")

    # # train(
    # #     model_class='multi_vae',
    # #     # model_class='warp_encoder',
    # #     n_epochs_pred_only=0,
    # #     n_epochs_ac_only=50,
    # #     n_epochs_pred_and_ac=100,
    # #     epochs_to_anneal_over=50,
    # #     max_kl=0.2,
    # #     ac_reg_loss_scaler=0.0005,
    # #     # evaluation_metric='AP',
    # #     evaluation_metric="NDCG",
    # #     # logging_frequency=25,
    # #     # logging_frequency=50,
    # #     logging_frequency=50,
    # #     batch_size=500,
    # #     # batch_size=25,
    # #     break_early=False,
    # #     verbose=False,
    # #     # path_to_save_actor="best_ndcg_trained_150_epochs",
    # #     # path_to_save_last_actor="last_actor_after_150_trained_epochs",
    # #     version_tag="making_one_trained_on_each_eval_metric",
    # #     # path_to_save_actor="test_actor_save",
    # #     restore_trained_actor_path="last_actor_after_150_trained_epochs"
    # # )

    # print("Just for good measure, I'm going to run the test function too.")

    # test(
    #     model_class='multi_vae',
    #     # model_class='warp_encoder',
    #     n_epochs_pred_only=0,
    #     n_epochs_ac_only=50,
    #     n_epochs_pred_and_ac=100,
    #     epochs_to_anneal_over=50,
    #     max_kl=0.2,
    #     ac_reg_loss_scaler=0.0005,
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
    #     version_tag="making_one_trained_on_each_eval_metric",
    #     # path_to_save_actor="test_actor_save",
    #     restore_trained_actor_path="last_actor_after_150_trained_epochs"
    # )



    # exit("Bye bye now! I doubt it will make it here, but a man can dream.")

    # for ac_reg_loss_scaler in [0.0, 1e-3, 1e-2, 1e-1]:
    #     train(
    #         model_class='multi_vae',
    #         # model_class='warp_encoder',
    #         n_epochs_pred_only=0,
    #         n_epochs_ac_only=50,
    #         n_epochs_pred_and_ac=50,
    #         epochs_to_anneal_over=50,
    #         max_kl=0.2,
    #         # evaluation_metric='AP',
    #         evaluation_metric="NDCG",
    #         # logging_frequency=25,
    #         logging_frequency=50,
    #         batch_size=500,
    #         # batch_size=25,
    #         break_early=False,
    #         verbose=False,
    #         ac_reg_loss_scaler=ac_reg_loss_scaler,
    #         version_tag="hyperparameter_ac_reg",
    #         # path_to_save_actor="best_ndcg_trained_100_epochs",
    #         # path_to_save_last_actor="last_actor_after_100_trained_epochs",
    #         # path_to_save_actor="test_actor_save",
    #         restore_trained_actor_path="best_ndcg_trained_100_epochs",
    #     )

