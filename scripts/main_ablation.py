import sys
import os
UTILS_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'utils')
sys.path.insert(1, UTILS_DIR)

from training import train, test

if __name__ == '__main__':

    DEFAULT_KWARGS = {
            'model_class': 'multi_vae',
            'n_epochs_pred_only': 150,
            'n_epochs_ac_only': 50,
            'n_epochs_pred_and_ac': 50,
            'max_kl': 0.2,
            'ac_reg_loss_scaler': 0.0,
            'evaluation_metric': "NDCG",
            'logging_frequency': 50,
            'batch_size': 500,
            'break_early': False,
            'verbose': False,
            'version_tag': "ABLATION_STUDY",
    }

    for omit_seen, omit_unseen in [(True, True), (False, True), (True, False)]:
        KWARGS = dict(DEFAULT_KWARGS)
        KWARGS['omit_num_seen_from_critic'] = omit_seen
        KWARGS['omit_num_unseen_from_critic'] = omit_unseen
        KWARGS['path_to_save_actor']: "200_epochs_VAE_omit_seen_{}_omit_unseen_{}".format(omit_seen, omit_unseen),        

        train(
            **KWARGS
        )

        print(f"Finished training actor for omit_seen={omit_seen} and omit_unseen={omit_unseen}")
