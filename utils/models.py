import tensorflow.contrib.slim as slim
import tensorflow as tf

# from base_models import BaseModelWithDefaultCritic
from base_models import (BaseModel, CriticModelMixin, StochasticActorModelMixin,
                         PointEstimateActorModelMixin, LinearModelMixin,
                         StochasticLinearActorModelMixin, ProperlyShapedPointEstimateModelMixin)
# from warp_utils import ErrorVectorCreator
from warp_utils import ErrorVectorCreator
from lambdarank_utils import my_lambdarank


class OldMultiDAE(CriticModelMixin, PointEstimateActorModelMixin, BaseModel):

    """
    This implements the MultiDAE from https://github.com/dawenl/vae_cf
    """

    def __init__(
            self,
            batch_of_users,
            heldout_batch,
            input_dim=None,
            anneal_cap=0.2,
            epochs_to_anneal_over=50,
            batch_size=500,
            evaluation_metric='NDCG',
            lr_actor=1e-3,
            lr_critic=1e-4,
            lr_ac=2e-6,
            actor_reg_loss_scaler=1e-4,  #The default in training is 1e-4 so I'll leave it like that.
            ac_reg_loss_scaler=0.0,  # We'll increase it as need be.
            **kwargs):
        local_variables = locals()
        local_variables.pop('kwargs')
        self._set_locals(local_variables)

        self.build_graph()
        self.saver = tf.train.Saver()

    def construct_actor_error(self):
        # NOTE: Check the KL term later.
        # NOTE: This will give the wrong number for validation stuff, because batch_of_users isn't the right output!.

        self.actor_error_mask = tf.identity(self.batch_of_users)

        log_softmax = tf.nn.log_softmax(self.prediction)
        actor_error = -tf.reduce_sum(log_softmax * self.actor_error_mask, axis=-1)

        # This way, KL isn't factored into the critic at all. Which is probably what we want, although the AC should have it.
        # mean_actor_error = tf.reduce_mean(actor_error) + (
        #     self.kl_loss_scaler * self.actor_regularization_loss)
        mean_actor_error = tf.reduce_mean(actor_error) + (
            self.actor_reg_loss_scaler * self.actor_regularization_loss)

        self.actor_error = actor_error
        self.mean_actor_error = mean_actor_error

    def create_logging_ops(self):
        tf.summary.scalar('mean_actor_error', self.mean_actor_error)  # Includes KL term...
        tf.summary.scalar('actor_reg', self.actor_regularization_loss)
        tf.summary.scalar('ndcg@100', tf.reduce_mean(self.true_ndcg))
        # tf.summary.scalar('AP@100', tf.reduce_mean(self.true_ap))
        # tf.summary.scalar('Recall@100', tf.reduce_mean(self.true_recall))
        tf.summary.scalar('mean_critic_error', self.mean_critic_error)


class MultiVAE(CriticModelMixin, StochasticActorModelMixin, BaseModel):

    """This implements the MultaVAE from https://github.com/dawenl/vae_cf"""

    def __init__(
            self,
            batch_of_users,
            heldout_batch,
            input_dim=None,
            anneal_cap=0.2,
            epochs_to_anneal_over=50,
            batch_size=500,
            evaluation_metric='NDCG',
            lr_actor=1e-3,
            lr_critic=1e-4,
            lr_ac=2e-6,
            ac_reg_loss_scaler=0.0,  # We'll increase it as need be.
            omit_num_seen_from_critic=False,
            omit_num_unseen_from_critic=False,
            **kwargs):
        """
        I'll do a better job about defining the inputs here.
        """
        local_variables = locals()
        local_variables.pop('kwargs')
        self._set_locals(local_variables)

        self.build_graph()
        self.saver = tf.train.Saver()

        if hasattr(self, 'superclass_stuff'):
            self.superclass_stuff()


    def construct_actor_error(self):
        # NOTE: Check the KL term later.
        # NOTE: This will give the wrong number for validation stuff, because batch_of_users isn't the right output!.

        self.actor_error_mask = tf.identity(self.batch_of_users)

        log_softmax = tf.nn.log_softmax(self.prediction)
        actor_error = -tf.reduce_sum(log_softmax * self.actor_error_mask, axis=-1)

        # This way, KL isn't factored into the critic at all. Which is probably what we want, although the AC should have it.
        mean_actor_error = tf.reduce_mean(actor_error) + (
            self.kl_loss_scaler * self.actor_regularization_loss)

        self.actor_error = actor_error
        self.mean_actor_error = mean_actor_error

    def create_logging_ops(self):
        tf.summary.scalar('mean_actor_error', self.mean_actor_error)  # Includes KL term...
        tf.summary.scalar('actor_reg', self.actor_regularization_loss)
        tf.summary.scalar('ndcg@100', tf.reduce_mean(self.true_ndcg))
        # tf.summary.scalar('AP@100', tf.reduce_mean(self.true_ap))
        # tf.summary.scalar('Recall@100', tf.reduce_mean(self.true_recall))
        tf.summary.scalar('mean_critic_error', self.mean_critic_error)

        # tf.summary.scalar("NDCG@200", tf.reduce_mean(self.true_ndcg_at_200))
        # tf.summary.scalar("NDCG@50", tf.reduce_mean(self.true_ndcg_at_50))
        # tf.summary.scalar("NDCG@20", tf.reduce_mean(self.true_ndcg_at_20))
        # tf.summary.scalar("NDCG@5", tf.reduce_mean(self.true_ndcg_at_5))

class MultiVAEWithPhase4LambdaRank(MultiVAE):

    """After pre-training a MultiVAE, this allows you to fine-tune the results with the LambdaRank objective"""

    def superclass_stuff(self):
        self.create_second_actor_error()
        self.create_second_logging_ops()
        self.create_training_op_for_second_actor()

    def create_second_actor_error(self):
        self.second_actor_error_mask = tf.identity(self.batch_of_users)
        second_actor_error = my_lambdarank(self.prediction, self.second_actor_error_mask)
        second_actor_error = tf.reduce_sum(second_actor_error, axis=-1)
        mean_second_actor_error = tf.reduce_mean(second_actor_error)

        self.second_actor_error = second_actor_error
        self.mean_second_actor_error = mean_second_actor_error

    def create_training_op_for_second_actor(self):
        """As written now, must have same LR as original actor..."""
        train_op = tf.train.AdamOptimizer(self.lr_actor).minimize(
            self.mean_second_actor_error, var_list=self.actor_forward_variables)  #TODO: var_list part

        self.second_actor_train_op = train_op

    def create_second_logging_ops(self):
        tf.summary.scalar('mean_second_actor_error', self.mean_second_actor_error)  # Includes KL term...

        pass


class MultiVAEWithPhase4WARP(MultiVAE):

    """After pre-training a MultiVAE, this allows you to fine-tune the results with the WARP objective"""

    def superclass_stuff(self):
        self.create_second_actor_error()
        self.create_second_logging_ops()
        self.create_training_op_for_second_actor()

    def create_second_actor_error(self):
        self.second_actor_error_mask = tf.identity(self.batch_of_users)
        # self.error_vector_creator = ErrorVectorCreator(
        #     input_dim=self.input_dim, limit=self.error_vector_limit)
        self.second_error_vector_creator = ErrorVectorCreator(input_dim=self.input_dim)
        error_scaler = tf.py_func(self.second_error_vector_creator,
                                  [self.prediction, self.second_actor_error_mask], tf.float32)

        true_second_error = self.prediction * error_scaler

        self.second_actor_error = tf.reduce_sum(true_second_error, axis=-1)
        # self.mean_second_actor_error = tf.reduce_mean(
        #     self.second_actor_error) + (self.actor_reg_loss_scaler * self.actor_regularization_loss)
        self.mean_second_actor_error = tf.reduce_mean(self.second_actor_error)


    def create_training_op_for_second_actor(self):
        """As written now, must have same LR as original actor..."""
        train_op = tf.train.AdamOptimizer(self.lr_actor).minimize(
            self.mean_second_actor_error, var_list=self.actor_forward_variables)  #TODO: var_list part

        self.second_actor_train_op = train_op


    def create_second_logging_ops(self):
        tf.summary.scalar('mean_second_actor_error', self.mean_second_actor_error)  # Includes KL term...

        pass



class LambdaRankEncoder(CriticModelMixin, StochasticActorModelMixin, BaseModel):

    def __init__(
            self,
            batch_of_users,
            heldout_batch,
            input_dim=None,
            anneal_cap=0.2,
            epochs_to_anneal_over=50,
            batch_size=500,
            evaluation_metric='NDCG',
            lr_actor=1e-3,
            lr_critic=1e-4,
            lr_ac=2e-6,
            #  actor_reg_loss_scaler=1e-4, #This is the KL scaler... And it varies as you go. So annoying.
            #  ac_reg_loss_scaler=1e-4,
            #  ac_reg_loss_scaler=0.2, #This uses the KL loss on the AC training.
            ac_reg_loss_scaler=0.0,  # We'll increase it as need be.
            omit_num_seen_from_critic=False,
            omit_num_unseen_from_critic=False,
            **kwargs):
        local_variables = locals()
        local_variables.pop('kwargs')
        self._set_locals(local_variables)

        self.build_graph()
        self.saver = tf.train.Saver()

        if hasattr(self, 'superclass_stuff'):
            self.superclass_stuff()

    def construct_actor_error(self):
        # NOTE: Check the KL term later.
        # NOTE: This will give the wrong number for validation stuff, because batch_of_users isn't the right output!.


        self.actor_error_mask = tf.identity(self.batch_of_users)
        actor_error = my_lambdarank(self.prediction, self.actor_error_mask)
        actor_error = tf.reduce_sum(actor_error, axis=-1)

        mean_actor_error = tf.reduce_mean(actor_error) + (self.kl_loss_scaler * self.actor_regularization_loss)

        self.actor_error = actor_error
        self.mean_actor_error = mean_actor_error


    def create_logging_ops(self):
        tf.summary.scalar('mean_actor_error', self.mean_actor_error)  # Includes KL term...
        tf.summary.scalar('actor_reg', self.actor_regularization_loss)
        tf.summary.scalar('ndcg@100', tf.reduce_mean(self.true_ndcg))
        # tf.summary.scalar('AP@100', tf.reduce_mean(self.true_ap))
        # tf.summary.scalar('Recall@100', tf.reduce_mean(self.true_recall))
        tf.summary.scalar('mean_critic_error', self.mean_critic_error)


class WarpEncoder(CriticModelMixin, LinearModelMixin, BaseModel):

    def __init__(
            self,
            batch_of_users,
            heldout_batch,
            input_dim=None,
            #  anneal_cap=0.2,
            #  epochs_to_anneal_over=50,
            # error_vector_limit=100,
            evaluation_metric='NDCG',
            batch_size=500,
            lr_actor=1e-3,
            lr_critic=1e-4,
            lr_ac=2e-6,
            # ac_reg_loss_scaler=1.0, # It's already scaled...
            ac_reg_loss_scaler=0.0,  #Just to be ...safe.
            actor_reg_loss_scaler=1e-4,
            **kwargs):
        """
        I'll do a better job about defining the inputs here.
        """
        local_variables = locals()
        local_variables.pop('kwargs')
        self._set_locals(local_variables)

        self.build_graph()
        self.saver = tf.train.Saver()

    def construct_actor_error(self):
        self.actor_error_mask = tf.identity(self.batch_of_users)
        # self.error_vector_creator = ErrorVectorCreator(
        #     input_dim=self.input_dim, limit=self.error_vector_limit)
        self.error_vector_creator = ErrorVectorCreator(input_dim=self.input_dim)
        error_scaler = tf.py_func(self.error_vector_creator,
                                  [self.prediction, self.actor_error_mask], tf.float32)

        true_error = self.prediction * error_scaler

        self.actor_error = tf.reduce_sum(true_error, axis=-1)
        print("Shape of actor_error should be like 500: {}".format(self.actor_error.get_shape()))
        self.mean_actor_error = tf.reduce_mean(
            self.actor_error) + (self.actor_reg_loss_scaler * self.actor_regularization_loss)

    def create_logging_ops(self):
        tf.summary.scalar('mean_actor_error', self.mean_actor_error)  # Includes KL term...
        tf.summary.scalar('actor_reg', self.actor_regularization_loss)
        tf.summary.scalar('ndcg@100', tf.reduce_mean(self.true_ndcg))
        # tf.summary.scalar('AP@100', tf.reduce_mean(self.true_ap))
        # tf.summary.scalar('Recall@100', tf.reduce_mean(self.true_recall))
        tf.summary.scalar('mean_critic_error', self.mean_critic_error)  #Includes regularization.


class WeightedMatrixFactorization(CriticModelMixin, LinearModelMixin, BaseModel):

    def __init__(
            self,
            batch_of_users,
            heldout_batch,
            input_dim=None,
            #  anneal_cap=0.2,
            #  epochs_to_anneal_over=50,
            # error_vector_limit=100,
            positive_weights=2.0,
            evaluation_metric='NDCG',
            batch_size=500,
            lr_actor=1e-3,
            lr_critic=1e-4,
            lr_ac=2e-6,
            # ac_reg_loss_scaler=1.0, # It's already scaled...
            ac_reg_loss_scaler=0.0,  #Just to be ...safe.
            actor_reg_loss_scaler=1e-4,
            **kwargs):
        """
        Positive weights is how much bigger the positives are than the negatives.
        so, if it's 1, then the click-matrix will have 1 for negative, and 2 for positive.
        """
        local_variables = locals()
        assert positive_weights >= 2.0
        local_variables.pop('kwargs')
        self._set_locals(local_variables)

        self.build_graph()
        self.saver = tf.train.Saver()

    def construct_actor_error(self):
        self.actor_error_mask = tf.identity(self.batch_of_users)

        # the error mask is all 0 and 1s. So, I'll multiply it by positive_weights, and then add one.
        error_scaler = (self.actor_error_mask * (self.positive_weights - 1)) + 1

        square_difference = tf.square(self.actor_error_mask - self.prediction)
        true_error = error_scaler * square_difference

        self.actor_error = tf.reduce_sum(true_error, axis=-1)
        print("Shape of actor_error should be like 500: {}".format(self.actor_error.get_shape()))

        self.mean_actor_error = tf.reduce_mean(
            self.actor_error) + (self.actor_reg_loss_scaler * self.actor_regularization_loss)

    def create_logging_ops(self):
        tf.summary.scalar('mean_actor_error', self.mean_actor_error)  # Includes KL term...
        tf.summary.scalar('actor_reg', self.actor_regularization_loss)
        tf.summary.scalar('ndcg@100', tf.reduce_mean(self.true_ndcg))
        # tf.summary.scalar('AP@100', tf.reduce_mean(self.true_ap))
        # tf.summary.scalar('Recall@100', tf.reduce_mean(self.true_recall))
        tf.summary.scalar('mean_critic_error', self.mean_critic_error)  #Includes regularization.


class ProperlyShapedMultiDAE(CriticModelMixin, ProperlyShapedPointEstimateModelMixin, BaseModel):
    """This looks an awful lot like MultiDAE, except for the inheritances! It uses a more complicated base model."""

    def __init__(
            self,
            batch_of_users,
            heldout_batch,
            input_dim=None,
            anneal_cap=0.2,
            epochs_to_anneal_over=50,
            batch_size=500,
            evaluation_metric='NDCG',
            lr_actor=1e-3,
            lr_critic=1e-4,
            lr_ac=2e-6,
            actor_reg_loss_scaler=1e-4,  #The default in training is 1e-4 so I'll leave it like that.
            #  ac_reg_loss_scaler=1e-4,
            #  ac_reg_loss_scaler=0.2, #This uses the KL loss on the AC training.
            ac_reg_loss_scaler=0.0,  # We'll increase it as need be.
            **kwargs):
        local_variables = locals()
        local_variables.pop('kwargs')
        self._set_locals(local_variables)

        self.build_graph()
        self.saver = tf.train.Saver()

    def construct_actor_error(self):
        # NOTE: Check the KL term later.
        # NOTE: This will give the wrong number for validation stuff, because batch_of_users isn't the right output!.

        self.actor_error_mask = tf.identity(self.batch_of_users)

        log_softmax = tf.nn.log_softmax(self.prediction)
        actor_error = -tf.reduce_sum(log_softmax * self.actor_error_mask, axis=-1)

        # This way, KL isn't factored into the critic at all. Which is probably what we want, although the AC should have it.
        # mean_actor_error = tf.reduce_mean(actor_error) + (
        #     self.kl_loss_scaler * self.actor_regularization_loss)
        mean_actor_error = tf.reduce_mean(actor_error) + (
            self.actor_reg_loss_scaler * self.actor_regularization_loss)

        self.actor_error = actor_error
        self.mean_actor_error = mean_actor_error

    def create_logging_ops(self):
        tf.summary.scalar('mean_actor_error', self.mean_actor_error)  # Includes KL term...
        tf.summary.scalar('actor_reg', self.actor_regularization_loss)
        tf.summary.scalar('ndcg@100', tf.reduce_mean(self.true_ndcg))
        # tf.summary.scalar('AP@100', tf.reduce_mean(self.true_ap))
        # tf.summary.scalar('Recall@100', tf.reduce_mean(self.true_recall))
        tf.summary.scalar('mean_critic_error', self.mean_critic_error)

        # tf.summary.scalar("NDCG@200", tf.reduce_mean(self.true_ndcg_at_200))
        # tf.summary.scalar("NDCG@50", tf.reduce_mean(self.true_ndcg_at_50))
        # tf.summary.scalar("NDCG@20", tf.reduce_mean(self.true_ndcg_at_20))
        # tf.summary.scalar("NDCG@5", tf.reduce_mean(self.true_ndcg_at_5))


class GaussianVAE(CriticModelMixin, StochasticActorModelMixin, BaseModel):

    def __init__(
            self,
            batch_of_users,
            heldout_batch,
            input_dim=None,
            #  anneal_cap=0.2,
            #  epochs_to_anneal_over=50,
            # error_vector_limit=100,
            positive_weights=2.0,
            evaluation_metric='NDCG',
            batch_size=500,
            lr_actor=1e-3,
            lr_critic=1e-4,
            lr_ac=2e-6,
            # ac_reg_loss_scaler=1.0, # It's already scaled...
            ac_reg_loss_scaler=0.0,  #Just to be ...safe.
            actor_reg_loss_scaler=1e-4,
            **kwargs):
        """
        Positive weights is how much bigger the positives are than the negatives.
        so, if it's 1, then the click-matrix will have 1 for negative, and 2 for positive.
        """
        local_variables = locals()
        assert positive_weights >= 2.0
        local_variables.pop('kwargs')
        self._set_locals(local_variables)

        self.build_graph()
        self.saver = tf.train.Saver()

    def construct_actor_error(self):
        self.actor_error_mask = tf.identity(self.batch_of_users)

        # the error mask is all 0 and 1s. So, I'll multiply it by positive_weights, and then add one.
        error_scaler = (self.actor_error_mask * (self.positive_weights - 1)) + 1

        square_difference = tf.square(self.actor_error_mask - self.prediction)
        true_error = error_scaler * square_difference

        self.actor_error = tf.reduce_sum(true_error, axis=-1)
        print("Shape of actor_error should be like 500: {}".format(self.actor_error.get_shape()))

        self.mean_actor_error = tf.reduce_mean(
            self.actor_error) + (self.kl_loss_scaler * self.actor_regularization_loss)

    def create_logging_ops(self):
        tf.summary.scalar('mean_actor_error', self.mean_actor_error)  # Includes KL term...
        tf.summary.scalar('actor_reg', self.actor_regularization_loss)
        tf.summary.scalar('ndcg@100', tf.reduce_mean(self.true_ndcg))
        # tf.summary.scalar('AP@100', tf.reduce_mean(self.true_ap))
        # tf.summary.scalar('Recall@100', tf.reduce_mean(self.true_recall))
        tf.summary.scalar('mean_critic_error', self.mean_critic_error)  #Includes regularization.


class VWMF(CriticModelMixin, StochasticLinearActorModelMixin, BaseModel):

    def __init__(
            self,
            batch_of_users,
            heldout_batch,
            input_dim=None,
            #  anneal_cap=0.2,
            #  epochs_to_anneal_over=50,
            # error_vector_limit=100,
            positive_weights=2.0,
            evaluation_metric='NDCG',
            batch_size=500,
            lr_actor=1e-3,
            lr_critic=1e-4,
            lr_ac=2e-6,
            # ac_reg_loss_scaler=1.0, # It's already scaled...
            ac_reg_loss_scaler=0.0,  #Just to be ...safe.
            actor_reg_loss_scaler=1e-4,
            **kwargs):
        """
        Positive weights is how much bigger the positives are than the negatives.
        so, if it's 1, then the click-matrix will have 1 for negative, and 2 for positive.
        """
        local_variables = locals()
        assert positive_weights >= 2.0
        local_variables.pop('kwargs')
        self._set_locals(local_variables)

        self.build_graph()
        self.saver = tf.train.Saver()

    def construct_actor_error(self):
        self.actor_error_mask = tf.identity(self.batch_of_users)

        # the error mask is all 0 and 1s. So, I'll multiply it by positive_weights, and then add one.
        error_scaler = (self.actor_error_mask * (self.positive_weights - 1)) + 1

        square_difference = tf.square(self.actor_error_mask - self.prediction)
        true_error = error_scaler * square_difference

        self.actor_error = tf.reduce_sum(true_error, axis=-1)
        print("Shape of actor_error should be like 500: {}".format(self.actor_error.get_shape()))

        self.mean_actor_error = tf.reduce_mean(
            self.actor_error) + (self.kl_loss_scaler * self.actor_regularization_loss)

    def create_logging_ops(self):
        tf.summary.scalar('mean_actor_error', self.mean_actor_error)  # Includes KL term...
        tf.summary.scalar('actor_reg', self.actor_regularization_loss)
        tf.summary.scalar('ndcg@100', tf.reduce_mean(self.true_ndcg))
        # tf.summary.scalar('AP@100', tf.reduce_mean(self.true_ap))
        # tf.summary.scalar('Recall@100', tf.reduce_mean(self.true_recall))
        tf.summary.scalar('mean_critic_error', self.mean_critic_error)  #Includes regularization.


class VariationalWarpEncoder(CriticModelMixin, StochasticActorModelMixin, BaseModel):

    def __init__(
            self,
            batch_of_users,
            heldout_batch,
            input_dim=None,
            #  anneal_cap=0.2,
            #  epochs_to_anneal_over=50,
            # error_vector_limit=100,
            evaluation_metric='NDCG',
            batch_size=500,
            lr_actor=1e-3,
            lr_critic=1e-4,
            lr_ac=2e-6,
            # ac_reg_loss_scaler=1.0, # It's already scaled...
            ac_reg_loss_scaler=0.0,  #Just to be ...safe.
            actor_reg_loss_scaler=1e-4,
            **kwargs):
        """
        I'll do a better job about defining the inputs here.
        """
        local_variables = locals()
        local_variables.pop('kwargs')
        self._set_locals(local_variables)

        self.build_graph()
        self.saver = tf.train.Saver()

    def construct_actor_error(self):
        self.actor_error_mask = tf.identity(self.batch_of_users)
        # self.error_vector_creator = ErrorVectorCreator(
        #     input_dim=self.input_dim, limit=self.error_vector_limit)
        self.error_vector_creator = ErrorVectorCreator(input_dim=self.input_dim)
        error_scaler = tf.py_func(self.error_vector_creator,
                                  [self.prediction, self.actor_error_mask], tf.float32)

        true_error = self.prediction * error_scaler

        self.actor_error = tf.reduce_sum(true_error, axis=-1)
        print("Shape of actor_error should be like 500: {}".format(self.actor_error.get_shape()))
        self.mean_actor_error = tf.reduce_mean(
            self.actor_error) + (self.kl_loss_scaler * self.actor_regularization_loss)

    def create_logging_ops(self):
        tf.summary.scalar('mean_actor_error', self.mean_actor_error)  # Includes KL term...
        tf.summary.scalar('actor_reg', self.actor_regularization_loss)
        tf.summary.scalar('ndcg@100', tf.reduce_mean(self.true_ndcg))
        # tf.summary.scalar('AP@100', tf.reduce_mean(self.true_ap))
        # tf.summary.scalar('Recall@100', tf.reduce_mean(self.true_recall))
        tf.summary.scalar('mean_critic_error', self.mean_critic_error)  #Includes regularization.
